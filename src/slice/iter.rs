//! Iterator definitions for `SegmentedSlice`.
//!
//! This module contains iterator types for `SegmentedSlice`, including:
//! - Split iterators (`Split`, `SplitMut`, `SplitInclusive`, etc.)
//! - Mutable chunk iterators (`ChunksMut`, `ChunksExactMut`, etc.)

use allocator_api2::alloc::{Allocator, Global};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;

use crate::raw_vec::RawSegmentedVec;
use crate::slice::{SegmentedSlice, SegmentedSliceMut};

impl<'a, T, A: Allocator> IntoIterator for SegmentedSlice<'a, T, A> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for SegmentedSliceMut<'a, T, A> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T, A>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SliceIterMut::into_iter(self)
    }
}

// An iterator over the elements of a `SegmentedSlice`.
/// An iterator over the elements of a `SegmentedSlice`.
///
/// This iterator is optimized for sequential access by tracking pointers
/// directly instead of computing segment locations on each iteration.
pub struct SliceIter<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    /// Current pointer for forward iteration
    ptr: NonNull<T>,
    /// End of current segment for forward iteration
    seg_end: NonNull<T>,
    /// Current segment index for forward iteration
    seg: usize,
    /// Current pointer for backward iteration (points to next element to yield)
    back_ptr: NonNull<T>,
    /// Start of current segment for backward iteration
    back_seg_start: NonNull<T>,
    /// Current segment index for backward iteration
    back_seg: usize,
    /// Remaining element count (used for size_hint and termination)
    remaining: usize,
    _marker: PhantomData<&'a T>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for SliceIter<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

unsafe impl<T: Sync, A: Allocator + Sync> Sync for SliceIter<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Send for SliceIter<'_, T, A> {}

impl<'a, T, A: Allocator> SliceIter<'a, T, A> {
    /// Creates a new `SliceIter` from a buffer and index range.
    #[inline]
    pub(crate) fn new(slice: &SegmentedSlice<'a, T, A>) -> Self {
        let len = slice.len;

        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                buf: slice.buf,
                ptr: NonNull::dangling(),
                seg_end: NonNull::dangling(),
                seg: 0,
                back_ptr: NonNull::dangling(),
                back_seg_start: NonNull::dangling(),
                back_seg: 0,
                remaining: len,
                _marker: PhantomData,
            };
        }

        // SAFETY: buf is valid for the lifetime 'a
        let buf_ref = unsafe { slice.buf.as_ref() };

        // Compute forward iteration state
        let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        let start_ptr =
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(start_seg).add(start_offset)) };
        let start_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
        let start_seg_end =
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(start_seg).add(start_seg_cap)) };

        // Compute backward iteration state
        // Optimize: usage of cached fields from SegmentedSlice
        let back_ptr = unsafe { NonNull::new_unchecked(slice.end_ptr.as_ptr().sub(1)) };
        let back_seg = slice.end_seg;
        let back_seg_start = unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(back_seg)) };

        Self {
            buf: slice.buf,
            ptr: start_ptr,
            seg_end: start_seg_end,
            seg: start_seg,
            back_ptr,
            back_seg_start,
            back_seg,
            remaining: len,
            _marker: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'a, T, A> {
        if self.remaining == 0 {
            return SegmentedSlice::new(self.buf, 0, 0);
        }

        // Optimize: calculate offset from seg_end to avoid loading segment_ptr from memory.
        let start = if std::mem::size_of::<T>() == 0 {
            0
        } else {
            let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.seg);
            let dist_to_end =
                unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) as usize };
            let offset = seg_cap - dist_to_end;
            let segment_start = self.buf().segment_start_index(self.seg);
            segment_start + offset
        };

        // Construct SegmentedSlice directly to avoid re-calculating end location
        // end_ptr is exclusive, so it is back_ptr + 1
        let end_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().add(1)) };

        SegmentedSlice {
            buf: self.buf,
            start,
            len: self.remaining,
            end_ptr,
            end_seg: self.back_seg,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn buf(&self) -> &RawSegmentedVec<T, A> {
        // SAFETY: The pointer is valid for the lifetime 'a
        unsafe { self.buf.as_ref() }
    }
}

// iterator! {struct Iter -> *const T, &'a T, const, {/* no mut */}, as_ref, each_ref, {
//     fn is_sorted_by<F>(self, mut compare: F) -> bool
//     where
//         Self: Sized,
//         F: FnMut(&Self::Item, &Self::Item) -> bool,
//     {
//         self.as_slice().is_sorted_by(|a, b| compare(&a, &b))
//     }
// }}

impl<T, A: Allocator> Clone for SliceIter<'_, T, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            buf: self.buf,
            ptr: self.ptr,
            seg_end: self.seg_end,
            seg: self.seg,
            back_ptr: self.back_ptr,
            back_seg_start: self.back_seg_start,
            back_seg: self.back_seg,
            remaining: self.remaining,
            _marker: self._marker,
        }
    }
}

/// A mutable iterator over the elements of a `SegmentedSlice`.
///
/// This iterator is optimized for sequential access by tracking pointers
/// directly instead of computing segment locations on each iteration.
pub struct SliceIterMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    /// Current pointer for forward iteration
    ptr: NonNull<T>,
    /// End of current segment for forward iteration
    seg_end: NonNull<T>,
    /// Current segment index for forward iteration
    seg: usize,
    /// Current pointer for backward iteration (points to next element to yield)
    back_ptr: NonNull<T>,
    /// Start of current segment for backward iteration
    back_seg_start: NonNull<T>,
    /// Current segment index for backward iteration
    back_seg: usize,
    /// Remaining element count (used for size_hint and termination)
    remaining: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator> SliceIterMut<'a, T, A> {
    /// Creates a new `SliceIterMut` from a buffer and index range.
    #[inline]
    pub(crate) fn new(slice: &mut SegmentedSliceMut<'_, T, A>) -> Self {
        // Delegate to from_slice_mut by manually copying fields.
        // This is safe because we are just creating another view over the same data,
        // and SliceIterMut logic handles the lifetime safety.
        Self::into_iter(SegmentedSliceMut {
            buf: slice.buf,
            start: slice.start,
            len: slice.len(),
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
            _marker: PhantomData,
        })
    }

    /// Creates a new `SliceIterMut` that consumes the `SegmentedSliceMut`.
    #[inline]
    pub(crate) fn into_iter(slice: SegmentedSliceMut<'a, T, A>) -> Self {
        let len = slice.len();

        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                buf: slice.buf,
                ptr: NonNull::dangling(),
                seg_end: NonNull::dangling(),
                seg: 0,
                back_ptr: NonNull::dangling(),
                back_seg_start: NonNull::dangling(),
                back_seg: 0,
                remaining: len,
                _marker: PhantomData,
            };
        }

        // SAFETY: buf is valid for the lifetime 'a
        let buf_ref = unsafe { slice.buf.as_ref() };

        // Compute forward iteration state
        let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        let start_ptr =
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(start_seg).add(start_offset)) };
        let start_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
        let start_seg_end =
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(start_seg).add(start_seg_cap)) };

        // Compute backward iteration state
        // Optimize: usage of cached fields from SegmentedSliceMut
        let back_ptr = unsafe { NonNull::new_unchecked(slice.end_ptr.as_ptr().sub(1)) };
        let back_seg = slice.end_seg;
        let back_seg_start = unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(back_seg)) };

        Self {
            buf: slice.buf,
            ptr: start_ptr,
            seg_end: start_seg_end,
            seg: start_seg,
            back_ptr,
            back_seg_start,
            back_seg,
            remaining: len,
            _marker: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn into_slice(self) -> SegmentedSlice<'a, T, A> {
        if self.remaining == 0 {
            return SegmentedSlice::new(self.buf, 0, 0);
        }

        // Optimize: calculate offset from seg_end to avoid loading segment_ptr from memory.
        let start = if std::mem::size_of::<T>() == 0 {
            0
        } else {
            let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.seg);
            let dist_to_end =
                unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) as usize };
            let offset = seg_cap - dist_to_end;
            let segment_start = self.buf().segment_start_index(self.seg);
            segment_start + offset
        };

        // Construct SegmentedSlice directly to avoid re-calculating end location
        // end_ptr is exclusive, so it is back_ptr + 1
        let end_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().add(1)) };

        SegmentedSlice {
            buf: self.buf,
            start,
            len: self.remaining,
            end_ptr,
            end_seg: self.back_seg,
            _marker: PhantomData,
        }
    }
    #[inline]
    fn buf(&self) -> &RawSegmentedVec<T, A> {
        // SAFETY: The pointer is valid for the lifetime 'a
        unsafe { self.buf.as_ref() }
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [`SegmentedSlice`].
///
/// [`split`]: SegmentedSlice::split
pub struct Split<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(crate) slice: SegmentedSlice<'a, T, A>,
    pred: P,
    finished: bool,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> Split<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, pred: P) -> Self {
        Self {
            slice,
            pred,
            finished: false,
        }
    }

    /// Returns the remainder of the original slice that has not yet been yielded.
    #[inline]
    pub fn as_slice(&self) -> &SegmentedSlice<'a, T, A> {
        &self.slice
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for Split<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Find the position of the next matching element
        let mut idx = None;
        for (i, elem) in self.slice.iter().enumerate() {
            if (self.pred)(elem) {
                idx = Some(i);
                break;
            }
        }

        match idx {
            None => {
                self.finished = true;
                // Return the remaining slice
                let result = SegmentedSlice::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + self.slice.len,
                );
                self.slice = SegmentedSlice::new(
                    self.slice.buf,
                    self.slice.start + self.slice.len,
                    self.slice.start + self.slice.len,
                );
                Some(result)
            }
            Some(idx) => {
                let result =
                    SegmentedSlice::new(self.slice.buf, self.slice.start, self.slice.start + idx);
                // Skip the separator element
                self.slice = SegmentedSlice::new(
                    self.slice.buf,
                    self.slice.start + idx + 1,
                    self.slice.start + self.slice.len,
                );
                Some(result)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // At least 1, at most len + 1
            (1, Some(self.slice.len() + 1))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for Split<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Find the position of the last matching element
        let mut idx = None;
        for (i, elem) in self.slice.iter().enumerate().rev() {
            if (self.pred)(elem) {
                idx = Some(i);
                break;
            }
        }

        match idx {
            None => {
                self.finished = true;
                let result = SegmentedSlice::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + self.slice.len,
                );
                self.slice =
                    SegmentedSlice::new(self.slice.buf, self.slice.start, self.slice.start);
                Some(result)
            }
            Some(idx) => {
                let result = SegmentedSlice::new(
                    self.slice.buf,
                    self.slice.start + idx + 1,
                    self.slice.start + self.slice.len,
                );
                // Update slice to exclude the separator and everything after
                self.slice =
                    SegmentedSlice::new(self.slice.buf, self.slice.start, self.slice.start + idx);
                Some(result)
            }
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for Split<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function. Unlike `Split`, it contains the matched part as a terminator
/// of the subslice.
///
/// This struct is created by the [`split_inclusive`] method on [`SegmentedSlice`].
///
/// [`split_inclusive`]: SegmentedSlice::split_inclusive
pub struct SplitInclusive<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSlice<'a, T, A>,
    pred: P,
    finished: bool,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitInclusive<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for SplitInclusive<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Find the position of the next matching element
        let idx = self
            .slice
            .iter()
            .position(|x| (self.pred)(x))
            .map(|idx| idx + 1) // Include the separator
            .unwrap_or(self.slice.len());

        if idx == self.slice.len() {
            self.finished = true;
        }

        let result = SegmentedSlice::new(self.slice.buf, self.slice.start, self.slice.start + idx);
        self.slice = SegmentedSlice::new(
            self.slice.buf,
            self.slice.start + idx,
            self.slice.start + self.slice.len,
        );
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (1, Some(std::cmp::max(1, self.slice.len())))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitInclusive<'a, T, A, P> where
    P: FnMut(&T) -> bool
{
}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [`SegmentedSlice`].
///
/// [`rsplit`]: SegmentedSlice::rsplit
pub struct RSplit<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: Split<'a, T, A, P>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> RSplit<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, pred: P) -> Self {
        Self {
            inner: Split::new(slice, pred),
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for RSplit<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for RSplit<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplit<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn`] method on [`SegmentedSlice`].
///
/// [`splitn`]: SegmentedSlice::splitn
pub struct SplitN<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: Split<'a, T, A, P>,
    count: usize,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitN<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: Split::new(slice, pred),
            count: n,
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for SplitN<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.count {
            0 => None,
            1 => {
                self.count = 0;
                if self.inner.finished {
                    None
                } else {
                    self.inner.finished = true;
                    let slice = &self.inner.slice;
                    Some(SegmentedSlice::new(
                        slice.buf,
                        slice.start,
                        slice.start + slice.len,
                    ))
                }
            }
            _ => {
                self.count -= 1;
                self.inner.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.size_hint();
        (
            std::cmp::min(self.count, lower),
            Some(upper.map_or(self.count, |u| std::cmp::min(self.count, u))),
        )
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitN<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits, starting from the end.
///
/// This struct is created by the [`rsplitn`] method on [`SegmentedSlice`].
///
/// [`rsplitn`]: SegmentedSlice::rsplitn
pub struct RSplitN<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: RSplit<'a, T, A, P>,
    count: usize,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> RSplitN<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: RSplit::new(slice, pred),
            count: n,
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for RSplitN<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.count {
            0 => None,
            1 => {
                self.count = 0;
                if self.inner.inner.finished {
                    None
                } else {
                    self.inner.inner.finished = true;
                    let slice = &self.inner.inner.slice;
                    Some(SegmentedSlice::new(
                        slice.buf,
                        slice.start,
                        slice.start + slice.len,
                    ))
                }
            }
            _ => {
                self.count -= 1;
                self.inner.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.size_hint();
        (
            std::cmp::min(self.count, lower),
            Some(upper.map_or(self.count, |u| std::cmp::min(self.count, u))),
        )
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplitN<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks.
///
/// This struct is created by the [`chunks_mut`] method on [`SegmentedSlice`].
///
/// [`chunks_mut`]: SegmentedSlice::chunks_mut
pub struct ChunksMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    start: usize,
    end: usize,
    chunk_size: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> ChunksMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_end = std::cmp::min(self.start + self.chunk_size, self.end);
            let result = SegmentedSliceMut::new(self.buf, self.start, chunk_end);
            self.start = chunk_end;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start >= self.end {
            (0, Some(0))
        } else {
            let len = self.end - self.start;
            let n = len.div_ceil(self.chunk_size);
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for ChunksMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let len = self.end - self.start;
            let remainder = len % self.chunk_size;
            let chunk_size = if remainder != 0 {
                remainder
            } else {
                self.chunk_size
            };
            let chunk_start = self.end - chunk_size;
            let result = SegmentedSliceMut::new(self.buf, chunk_start, self.end);
            self.end = chunk_start;
            Some(result)
        }
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for ChunksMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for ChunksMut<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks of
/// exact size.
///
/// When the slice length is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the `into_remainder` function.
///
/// This struct is created by the [`chunks_exact_mut`] method on [`SegmentedSlice`].
///
/// [`chunks_exact_mut`]: SegmentedSlice::chunks_exact_mut
pub struct ChunksExactMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    start: usize,
    /// End of the exact chunks (excludes remainder)
    end: usize,
    /// End of the full slice (includes remainder)
    full_end: usize,
    chunk_size: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> ChunksExactMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_end = slice.start + len - rem;
        Self {
            buf: slice.buf,
            start: slice.start,
            end: exact_end,
            full_end: slice.start + len,
            chunk_size,
            _marker: PhantomData,
        }
    }

    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    #[inline]
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T, A> {
        SegmentedSliceMut::new(self.buf, self.end, self.full_end)
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksExactMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start + self.chunk_size > self.end {
            None
        } else {
            let chunk_end = self.start + self.chunk_size;
            let result = SegmentedSliceMut::new(self.buf, self.start, chunk_end);
            self.start = chunk_end;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.end - self.start) / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for ChunksExactMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start + self.chunk_size > self.end {
            None
        } else {
            let chunk_start = self.end - self.chunk_size;
            let result = SegmentedSliceMut::new(self.buf, chunk_start, self.end);
            self.end = chunk_start;
            Some(result)
        }
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for ChunksExactMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for ChunksExactMut<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) chunks of exact
/// size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact`]: SegmentedSlice::rchunks_exact
pub struct RChunksExact<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    /// Start of the exact chunks (excludes remainder)
    start: usize,
    end: usize,
    /// Start of the full slice (includes remainder)
    full_start: usize,
    chunk_size: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator + 'a> RChunksExact<'a, T, A> {
    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    pub fn remainder(&self) -> SegmentedSlice<'a, T, A> {
        SegmentedSlice::new(self.buf, self.full_start, self.start)
    }
}

impl<'a, T, A: Allocator + 'a> Clone for RChunksExact<'a, T, A> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf,
            start: self.start,
            end: self.end,
            full_start: self.full_start,
            chunk_size: self.chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> RChunksExact<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        Self {
            buf: slice.buf,
            start: exact_start,
            end: slice.start + len,
            full_start: slice.start,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunksExact<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.end - self.start < self.chunk_size {
            None
        } else {
            let chunk_start = self.end - self.chunk_size;
            let result = SegmentedSlice::new(self.buf, chunk_start, self.end);
            self.end = chunk_start;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.end - self.start) / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksExact<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end - self.start < self.chunk_size {
            None
        } else {
            let chunk_end = self.start + self.chunk_size;
            let result = SegmentedSlice::new(self.buf, self.start, chunk_end);
            self.start = chunk_end;
            Some(result)
        }
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for RChunksExact<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for RChunksExact<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks,
/// starting at the end of the slice.
///
/// This struct is created by the [`rchunks_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_mut`]: SegmentedSlice::rchunks_mut
pub struct RChunksMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    start: usize,
    end: usize,
    chunk_size: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> RChunksMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunksMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_start = self.end.saturating_sub(self.chunk_size).max(self.start);
            let result = SegmentedSliceMut::new(self.buf, chunk_start, self.end);
            self.end = chunk_start;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start >= self.end {
            (0, Some(0))
        } else {
            let len = self.end - self.start;
            let n = len.div_ceil(self.chunk_size);
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let len = self.end - self.start;
            let remainder = len % self.chunk_size;
            let chunk_size = if remainder != 0 {
                remainder
            } else {
                self.chunk_size
            };
            let chunk_end = self.start + chunk_size;
            let result = SegmentedSliceMut::new(self.buf, self.start, chunk_end);
            self.start = chunk_end;
            Some(result)
        }
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for RChunksMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for RChunksMut<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks of
/// exact size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact_mut`]: SegmentedSlice::rchunks_exact_mut
pub struct RChunksExactMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    start: usize,
    end: usize,
    full_start: usize,
    chunk_size: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> RChunksExactMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        Self {
            buf: slice.buf,
            start: exact_start,
            end: slice.start + len,
            full_start: slice.start,
            chunk_size,
            _marker: PhantomData,
        }
    }

    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T, A> {
        SegmentedSliceMut::new(self.buf, self.full_start, self.start)
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunksExactMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.end - self.start < self.chunk_size {
            None
        } else {
            let chunk_start = self.end - self.chunk_size;
            let result = SegmentedSliceMut::new(self.buf, chunk_start, self.end);
            self.end = chunk_start;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.end - self.start) / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksExactMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end - self.start < self.chunk_size {
            None
        } else {
            let chunk_end = self.start + self.chunk_size;
            let result = SegmentedSliceMut::new(self.buf, self.start, chunk_end);
            self.start = chunk_end;
            Some(result)
        }
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for RChunksExactMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for RChunksExactMut<'a, T, A> {}

impl<'a, T, A: Allocator> Iterator for SliceIter<'a, T, A> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            // SAFETY: ZSTs don't need valid memory, dangling pointer is fine
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means ptr is valid
        let result = unsafe { self.ptr.as_ref() };
        self.remaining -= 1;

        if self.remaining > 0 {
            // Advance pointer
            self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(1)) };

            // Check if we need to move to the next segment
            if self.ptr == self.seg_end && self.seg < self.back_seg {
                self.seg += 1;
                let seg_ptr = unsafe { self.buf().segment_ptr(self.seg) };
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.seg);
                self.ptr = unsafe { NonNull::new_unchecked(seg_ptr) };
                self.seg_end = unsafe { NonNull::new_unchecked(seg_ptr.add(seg_cap)) };
            }
        }

        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, A: Allocator> ExactSizeIterator for SliceIter<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for SliceIter<'_, T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for SliceIter<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means back_ptr is valid
        let result = unsafe { self.back_ptr.as_ref() };
        self.remaining -= 1;

        if self.remaining > 0 {
            // Check if we need to move to the previous segment
            if self.back_ptr == self.back_seg_start && self.back_seg > self.seg {
                self.back_seg -= 1;
                let seg_ptr = unsafe { self.buf().segment_ptr(self.back_seg) };
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.back_seg);
                self.back_seg_start = unsafe { NonNull::new_unchecked(seg_ptr) };
                self.back_ptr = unsafe { NonNull::new_unchecked(seg_ptr.add(seg_cap - 1)) };
            } else {
                self.back_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().sub(1)) };
            }
        }

        Some(result)
    }
}

impl<'a, T, A: Allocator> Iterator for SliceIterMut<'a, T, A> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            // SAFETY: ZSTs don't need valid memory, dangling pointer is fine
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means ptr is valid
        let result = unsafe { &mut *self.ptr.as_ptr() };
        self.remaining -= 1;

        if self.remaining > 0 {
            // Advance pointer
            self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(1)) };

            // Check if we need to move to the next segment
            if self.ptr == self.seg_end && self.seg < self.back_seg {
                self.seg += 1;
                let seg_ptr = unsafe { self.buf().segment_ptr(self.seg) };
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.seg);
                self.ptr = unsafe { NonNull::new_unchecked(seg_ptr) };
                self.seg_end = unsafe { NonNull::new_unchecked(seg_ptr.add(seg_cap)) };
            }
        }

        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, A: Allocator> ExactSizeIterator for SliceIterMut<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for SliceIterMut<'_, T, A> {}

impl<T, A: Allocator> DoubleEndedIterator for SliceIterMut<'_, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means back_ptr is valid
        let result = unsafe { &mut *self.back_ptr.as_ptr() };
        self.remaining -= 1;

        if self.remaining > 0 {
            // Check if we need to move to the previous segment
            if self.back_ptr == self.back_seg_start && self.back_seg > self.seg {
                self.back_seg -= 1;
                let seg_ptr = unsafe { self.buf().segment_ptr(self.back_seg) };
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.back_seg);
                self.back_seg_start = unsafe { NonNull::new_unchecked(seg_ptr) };
                self.back_ptr = unsafe { NonNull::new_unchecked(seg_ptr.add(seg_cap - 1)) };
            } else {
                self.back_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().sub(1)) };
            }
        }

        Some(result)
    }
}

/// An iterator over chunks of a `SegmentedSlice`.
pub struct Chunks<'a, T, A: Allocator + 'a = Global> {
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) chunk_size: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator> Chunks<'a, T, A> {
    #[inline]
    fn buf(&self) -> &RawSegmentedVec<T, A> {
        unsafe { self.buf.as_ref() }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for Chunks<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_end = std::cmp::min(self.start + self.chunk_size, self.end);
            let chunk_len = chunk_end - self.start;

            // Compute start optimization fields
            let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(self.start);
            let _start_ptr = unsafe { self.buf().segment_ptr(start_seg).add(start_offset) };

            // Compute end optimization fields
            let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_end);
            let end_ptr = if end_seg >= self.buf().segment_count() {
                let (last_seg, last_offset) = RawSegmentedVec::<T, A>::location(chunk_end - 1);
                end_seg = last_seg;
                unsafe {
                    NonNull::new_unchecked(self.buf().segment_ptr(last_seg).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked(self.buf().segment_ptr(end_seg).add(end_offset)) }
            };

            let chunk = SegmentedSlice {
                buf: self.buf,
                start: self.start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };
            self.start = chunk_end;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.start);
        let chunks = remaining.div_ceil(self.chunk_size);
        (chunks, Some(chunks))
    }
}

impl<T, A: Allocator> ExactSizeIterator for Chunks<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for Chunks<'_, T, A> {}

/// An iterator over exact-sized chunks of a `SegmentedSlice`.
impl<'a, T, A: Allocator + 'a> Chunks<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

pub struct ChunksExact<'a, T, A: Allocator + 'a = Global> {
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) remainder_start: usize,
    pub(crate) remainder_end: usize,
    pub(crate) chunk_size: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator> ChunksExact<'a, T, A> {
    #[inline]
    fn buf(&self) -> &RawSegmentedVec<T, A> {
        unsafe { self.buf.as_ref() }
    }

    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    pub fn remainder(&self) -> SegmentedSlice<'a, T, A> {
        let len = self.remainder_end - self.remainder_start;
        if len == 0 {
            SegmentedSlice {
                buf: self.buf,
                start: self.remainder_start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            }
        } else {
            // Compute start optimization fields
            let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(self.remainder_start);
            let _start_ptr = unsafe { self.buf().segment_ptr(start_seg).add(start_offset) };

            // Compute end optimization fields
            let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(self.remainder_end);
            let end_ptr = if end_seg >= self.buf().segment_count() {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T, A>::location(self.remainder_end - 1);
                end_seg = last_seg;
                unsafe {
                    NonNull::new_unchecked(self.buf().segment_ptr(last_seg).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked(self.buf().segment_ptr(end_seg).add(end_offset)) }
            };

            SegmentedSlice {
                buf: self.buf,
                start: self.remainder_start,
                len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            }
        }
    }
}

impl<'a, T, A: Allocator + 'a> ChunksExact<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_end = slice.start + len - rem;
        Self {
            buf: slice.buf,
            start: slice.start,
            end: exact_end,
            remainder_start: exact_end,
            remainder_end: slice.start + len,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksExact<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start + self.chunk_size > self.end {
            None
        } else {
            let chunk_end = self.start + self.chunk_size;

            // Compute start optimization fields
            let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(self.start);
            let _start_ptr = unsafe { self.buf().segment_ptr(start_seg).add(start_offset) };

            // Compute end optimization fields
            let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_end);
            let end_ptr = if end_seg >= self.buf().segment_count() {
                let (last_seg, last_offset) = RawSegmentedVec::<T, A>::location(chunk_end - 1);
                end_seg = last_seg;
                unsafe {
                    NonNull::new_unchecked(self.buf().segment_ptr(last_seg).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked(self.buf().segment_ptr(end_seg).add(end_offset)) }
            };

            let chunk = SegmentedSlice {
                buf: self.buf,
                start: self.start,
                len: self.chunk_size,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };
            self.start += self.chunk_size;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.start);
        let chunks = remaining / self.chunk_size;
        (chunks, Some(chunks))
    }
}

impl<T, A: Allocator> ExactSizeIterator for ChunksExact<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for ChunksExact<'_, T, A> {}

/// An iterator over chunks of a `SegmentedSlice`, starting from the end.
pub struct RChunks<'a, T, A: Allocator + 'a = Global> {
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) chunk_size: usize,
    pub(crate) _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator> RChunks<'a, T, A> {
    #[inline]
    fn buf(&self) -> &RawSegmentedVec<T, A> {
        unsafe { self.buf.as_ref() }
    }
}

impl<'a, T, A: Allocator + 'a> RChunks<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunks<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_start = if self.end - self.start >= self.chunk_size {
                self.end - self.chunk_size
            } else {
                self.start
            };
            let chunk_len = self.end - chunk_start;

            // Compute start optimization fields
            let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(chunk_start);
            let _start_ptr = unsafe { self.buf().segment_ptr(start_seg).add(start_offset) };

            // Compute end optimization fields
            let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(self.end);
            let end_ptr = if end_seg >= self.buf().segment_count() {
                let (last_seg, last_offset) = RawSegmentedVec::<T, A>::location(self.end - 1);
                end_seg = last_seg;
                unsafe {
                    NonNull::new_unchecked(self.buf().segment_ptr(last_seg).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked(self.buf().segment_ptr(end_seg).add(end_offset)) }
            };

            let chunk = SegmentedSlice {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };
            self.end = chunk_start;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.start);
        let chunks = remaining.div_ceil(self.chunk_size);
        (chunks, Some(chunks))
    }
}

impl<T, A: Allocator> ExactSizeIterator for RChunks<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for RChunks<'_, T, A> {}

/// An iterator over overlapping windows of a `SegmentedSlice`.
pub struct Windows<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSlice<'a, T, A>,
    pub(crate) size: NonZero<usize>,
}

impl<'a, T, A: Allocator + 'a> Iterator for Windows<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let size = self.size.get();
        if self.slice.len() < size {
            return None;
        }

        // Better:
        let window_end = self.slice.start + size;

        // Compute end optimization fields for the WINDOW
        // We need RawSegmentedVec access.
        let buf_ref = unsafe { self.slice.buf.as_ref() };

        let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(window_end);
        let end_ptr = if end_seg >= buf_ref.segment_count() {
            let (last_seg, last_offset) = RawSegmentedVec::<T, A>::location(window_end - 1);
            end_seg = last_seg;
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(last_seg).add(last_offset + 1)) }
        } else {
            unsafe { NonNull::new_unchecked(buf_ref.segment_ptr(end_seg).add(end_offset)) }
        };

        let window = SegmentedSlice {
            buf: self.slice.buf,
            start: self.slice.start,
            len: size,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        };

        // Advance the iterator
        self.slice.start += 1;
        self.slice.len -= 1;

        Some(window)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.len() < self.size.get() {
            (0, Some(0))
        } else {
            let windows = self.slice.len() - self.size.get() + 1;
            (windows, Some(windows))
        }
    }
}

impl<T, A: Allocator> ExactSizeIterator for Windows<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for Windows<'_, T, A> {}

unsafe impl<T: Send, A: Allocator + Send> Send for SliceIterMut<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for SliceIterMut<'_, T, A> {}

/// An iterator over subslices separated by elements that match a predicate function.
pub struct SplitMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSliceMut<'a, T, A>,
    pred: P,
    finished: bool,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitMut<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for SplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Find the position of the next matching element
        let mut idx = None;
        // SegmentedSliceMut::iter() returns mutable references, but we only need to read for predicate.
        // But predicate takes &T. So &mut T -> &T coercion works? Yes.
        // Wait, iter() consumes self? No, iter() on reference.
        // But self.slice is owned SegmentedSliceMut.
        // We can create iterator from reference to self.slice.
        for (i, elem) in self.slice.iter().enumerate() {
            if (self.pred)(elem) {
                idx = Some(i);
                break;
            }
        }

        match idx {
            None => {
                self.finished = true;
                // Return the remaining slice
                let result = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + self.slice.len,
                );
                // Set slice to empty
                self.slice = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start + self.slice.len,
                    self.slice.start + self.slice.len,
                );
                Some(result)
            }
            Some(idx) => {
                let result = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + idx,
                );
                // Skip the separator element
                self.slice = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start + idx + 1,
                    self.slice.start + self.slice.len,
                );
                Some(result)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // At least 1, at most len + 1
            (1, Some(self.slice.len() + 1))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for SplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Find the position of the last matching element
        let mut idx = None;
        for (i, elem) in self.slice.iter().enumerate().rev() {
            if (self.pred)(elem) {
                idx = Some(i);
                break;
            }
        }

        match idx {
            None => {
                self.finished = true;
                let result = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + self.slice.len,
                );
                self.slice =
                    SegmentedSliceMut::new(self.slice.buf, self.slice.start, self.slice.start);
                Some(result)
            }
            Some(idx) => {
                let result = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start + idx + 1,
                    self.slice.start + self.slice.len,
                );
                // Update slice to exclude the separator and everything after
                self.slice = SegmentedSliceMut::new(
                    self.slice.buf,
                    self.slice.start,
                    self.slice.start + idx,
                );
                Some(result)
            }
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitMut<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over non-overlapping windows of a slice.
pub struct ArrayWindows<'a, T: 'a, const N: usize> {
    slice: SegmentedSlice<'a, T>,
}

impl<'a, T: 'a, const N: usize> Iterator for ArrayWindows<'a, T, N> {
    type Item = &'a [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// An iterator over subslices separated by elements that match a predicate function.
pub struct ChunkBy<'a, T, F> {
    slice: SegmentedSlice<'a, T>,
    pred: F,
}

impl<'a, T, F> Iterator for ChunkBy<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// An iterator over mutable subslices separated by elements that match a predicate function.
pub struct ChunkByMut<'a, T, F> {
    slice: SegmentedSliceMut<'a, T>,
    pred: F,
}

impl<'a, T, F> Iterator for ChunkByMut<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct SplitInclusiveMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSliceMut<'a, T, A>,
    pred: P,
    finished: bool,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitInclusiveMut<'a, T, A, P> {
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for SplitInclusiveMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

pub struct RSplitMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: SplitMut<'a, T, A, P>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> RSplitMut<'a, T, A, P> {
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, pred: P) -> Self {
        Self {
            inner: SplitMut::new(slice, pred),
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for RSplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
