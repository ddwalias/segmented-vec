//! Iterator definitions for `SegmentedSlice`.
//!
//! This module contains iterator types for `SegmentedSlice`, including:
//! - Split iterators (`Split`, `SplitMut`, `SplitInclusive`, etc.)
//! - Mutable chunk iterators (`ChunksMut`, `ChunksExactMut`, etc.)

use allocator_api2::alloc::{Allocator, Global};
use std::cmp;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
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

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
#[doc(hidden)]
pub(super) trait SplitIter: DoubleEndedIterator {
    /// Marks the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [`SegmentedSlice`].
///
/// [`split`]: SegmentedSlice::split
#[derive(Debug, Clone)]
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

        let mut iter = self.slice.iter();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                return self.finish();
            }

            // SAFETY: iter.remaining > 0 means ptr is valid
            let element = unsafe { &*iter.ptr.as_ptr() };

            if (self.pred)(element) {
                // Match found!
                let result = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start,
                    len: consumed,
                    end_ptr: iter.ptr, // Reuse the pointer we stopped at
                    end_seg: iter.seg, // Reuse the segment index
                    _marker: PhantomData,
                };

                // Advance past the separator
                iter.next();

                // Update slice to start after the separator
                self.slice = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start + consumed + 1,
                    len: iter.remaining, // Only remains what's left in the iterator
                    end_ptr: self.slice.end_ptr, // Original end is preserved
                    end_seg: self.slice.end_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }

            iter.next();
            consumed += 1;
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

        let mut iter = self.slice.iter();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                return self.finish();
            }

            // SAFETY: iter.remaining > 0 means back_ptr is valid
            let element = unsafe { &*iter.back_ptr.as_ptr() };

            if (self.pred)(element) {
                let result_start = self.slice.start + iter.remaining;

                let result = SegmentedSlice {
                    buf: self.slice.buf,
                    start: result_start,
                    len: consumed,
                    end_ptr: self.slice.end_ptr,
                    end_seg: self.slice.end_seg,
                    _marker: PhantomData,
                };

                // Capture separator pointer/seg BEFORE advancing
                let separator_ptr = iter.back_ptr;
                let separator_seg = iter.back_seg;

                iter.next_back();

                self.slice = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start,
                    len: iter.remaining,
                    end_ptr: separator_ptr,
                    end_seg: separator_seg,
                    _marker: PhantomData,
                };
                return Some(result);
            }

            iter.next_back();
            consumed += 1;
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> SplitIter for Split<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSlice<'a, T, A>> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(self.slice)
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
#[derive(Debug, Clone)]
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

        let mut iter = self.slice.iter();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                if self.finished || self.slice.is_empty() {
                    self.finished = true;
                    return None;
                } else {
                    self.finished = true;
                    return Some(self.slice);
                }
            }

            // SAFETY: iter.remaining > 0 means ptr is valid
            let element = unsafe { &*iter.ptr.as_ptr() };
            let matched = (self.pred)(element);

            iter.next();
            consumed += 1;

            if matched {
                // Match found! Include the separator in the result.
                let result = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start,
                    len: consumed,
                    end_ptr: iter.ptr, // ptr is already past the separator
                    end_seg: iter.seg,
                    _marker: PhantomData,
                };

                // Update slice to start after the separator
                self.slice = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start + consumed,
                    len: iter.remaining,
                    end_ptr: self.slice.end_ptr,
                    end_seg: self.slice.end_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (1, Some(self.slice.len()))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for SplitInclusive<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut iter = self.slice.iter();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                if self.finished || self.slice.is_empty() {
                    self.finished = true;
                    return None;
                } else {
                    self.finished = true;
                    return Some(self.slice);
                }
            }

            // SAFETY: iter.remaining > 0 means back_ptr is valid
            let element = unsafe { &*iter.back_ptr.as_ptr() };
            let matched = (self.pred)(element);

            if matched && consumed > 0 {
                let result_start = self.slice.start + iter.remaining;
                let result_len = consumed;

                let result = SegmentedSlice {
                    buf: self.slice.buf,
                    start: result_start,
                    len: result_len,
                    end_ptr: self.slice.end_ptr,
                    end_seg: self.slice.end_seg,
                    _marker: PhantomData,
                };

                // Update self.slice to contain the left part (including the separator)
                // The separator is at `iter.back_ptr`.
                // We want the end to be `iter.back_ptr` + 1 (exclusive).
                // Which is exactly `result_start` in pointer terms? No.
                // `result_start` calculation `self.slice.start + iter.remaining` matches the *count*.

                // We need `end_ptr` for the stored slice.
                // `end_ptr` should be S+1. S is at `iter.back_ptr`.
                let new_end_ptr = unsafe { NonNull::new_unchecked(iter.back_ptr.as_ptr().add(1)) };
                // `end_seg` is `iter.back_seg`.

                self.slice = SegmentedSlice {
                    buf: self.slice.buf,
                    start: self.slice.start,
                    len: iter.remaining,
                    end_ptr: new_end_ptr,
                    end_seg: iter.back_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }

            iter.next_back();
            consumed += 1;
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitInclusive<'a, T, A, P> where
    P: FnMut(&T) -> bool
{
}

/// An iterator over subslices separated by elements that match a predicate function.
#[derive(Debug)]
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

        // Capture fields before borrowing slice mutably for replace and loop
        let slice_buf = self.slice.buf;
        let slice_start = self.slice.start;
        let slice_len = self.slice.len;
        let slice_end_ptr = self.slice.end_ptr;
        let slice_end_seg = self.slice.end_seg;

        // Take the slice to avoid mutable borrow conflict during iteration
        let mut slice = std::mem::replace(
            &mut self.slice,
            SegmentedSliceMut {
                buf: slice_buf,
                start: slice_start + slice_len,
                len: 0,
                end_ptr: NonNull::dangling(), // Temporary, will be overwritten if match
                end_seg: 0,
                _marker: PhantomData,
            },
        );

        let mut iter = slice.iter_mut();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                // iter is exhausted, slice is fully consumed.
                // We return the remaining slice via finish().
                self.slice = slice;
                return self.finish();
            }

            // SAFETY: iter.remaining > 0 means iter.ptr is valid
            let element = unsafe { &*iter.ptr.as_ptr() };

            if (self.pred)(element) {
                // Match found.
                // Reconstruct result and remaining parts.

                let result = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start,
                    len: consumed,
                    end_ptr: iter.ptr,
                    end_seg: iter.seg,
                    _marker: PhantomData,
                };

                iter.next(); // Skip separator

                // Put remaining part back into self.slice
                self.slice = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start + consumed + 1,
                    len: iter.remaining,
                    end_ptr: slice_end_ptr,
                    end_seg: slice_end_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }

            iter.next();
            consumed += 1;
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

        // Capture fields before borrowing slice mutably
        let slice_buf = self.slice.buf;
        let slice_start = self.slice.start;
        let slice_end_ptr = self.slice.end_ptr;
        let slice_end_seg = self.slice.end_seg;

        // Take the slice to avoid mutable borrow conflict
        let mut slice = std::mem::replace(
            &mut self.slice,
            SegmentedSliceMut {
                buf: slice_buf,
                start: slice_start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            },
        );

        let mut iter = slice.iter_mut();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                self.slice = slice;
                return self.finish();
            }

            // SAFETY: iter.remaining > 0 means back_ptr is valid
            let element = unsafe { &*iter.back_ptr.as_ptr() };

            if (self.pred)(element) {
                let result_start = slice_start + iter.remaining;

                let result = SegmentedSliceMut {
                    buf: slice_buf,
                    start: result_start,
                    len: consumed,
                    end_ptr: slice_end_ptr,
                    end_seg: slice_end_seg,
                    _marker: PhantomData,
                };

                // Capture separator pointer/seg BEFORE advancing
                let separator_ptr = iter.back_ptr;
                let separator_seg = iter.back_seg;

                iter.next_back();

                // Put remaining part back into self.slice
                self.slice = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start,
                    len: iter.remaining,
                    end_ptr: separator_ptr,
                    end_seg: separator_seg,
                    _marker: PhantomData,
                };
                return Some(result);
            }

            iter.next_back();
            consumed += 1;
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> SplitIter for SplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSliceMut<'a, T, A>> {
        if self.finished {
            None
        } else {
            self.finished = true;
            // Capture fields before borrowing slice mutably
            let slice_buf = self.slice.buf;
            let slice_start = self.slice.start;

            Some(std::mem::replace(
                &mut self.slice,
                SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start,
                    len: 0,
                    end_ptr: NonNull::dangling(),
                    end_seg: 0,
                    _marker: PhantomData,
                },
            ))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitMut<'a, T, A, P> where P: FnMut(&T) -> bool {}

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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Capture fields before borrowing slice mutably for replace and loop
        let slice_buf = self.slice.buf;
        let slice_start = self.slice.start;
        let slice_len = self.slice.len;
        let slice_end_ptr = self.slice.end_ptr;
        let slice_end_seg = self.slice.end_seg;

        // Take the slice to avoid mutable borrow conflict during iteration
        let mut slice = std::mem::replace(
            &mut self.slice,
            SegmentedSliceMut {
                buf: slice_buf,
                start: slice_start + slice_len,
                len: 0,
                end_ptr: NonNull::dangling(), // Temporary, will be overwritten if match
                end_seg: 0,
                _marker: PhantomData,
            },
        );

        let mut iter = slice.iter_mut();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                if self.finished || slice.is_empty() {
                    self.finished = true;
                    return None;
                } else {
                    self.finished = true;
                    return Some(slice);
                }
            }

            // SAFETY: iter.remaining > 0 means iter.ptr is valid
            let element = unsafe { &*iter.ptr.as_ptr() };
            let matched = (self.pred)(element);

            iter.next();
            consumed += 1;

            if matched {
                // Match found! Include the separator in the result.
                let result = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start,
                    len: consumed,
                    end_ptr: iter.ptr, // ptr is already past the separator
                    end_seg: iter.seg,
                    _marker: PhantomData,
                };

                // Update slice to start after the separator
                self.slice = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start + consumed,
                    len: iter.remaining,
                    end_ptr: slice_end_ptr,
                    end_seg: slice_end_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (1, Some(self.slice.len()))
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for SplitInclusiveMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Capture fields before borrowing slice mutably
        let slice_buf = self.slice.buf;
        let slice_start = self.slice.start;
        let slice_end_ptr = self.slice.end_ptr;
        let slice_end_seg = self.slice.end_seg;

        // Take the slice to avoid mutable borrow conflict
        let mut slice = std::mem::replace(
            &mut self.slice,
            SegmentedSliceMut {
                buf: slice_buf,
                start: slice_start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            },
        );

        let mut iter = slice.iter_mut();
        let mut consumed = 0;

        loop {
            if iter.remaining == 0 {
                if self.finished || slice.is_empty() {
                    self.finished = true;
                    return None;
                } else {
                    self.finished = true;
                    return Some(slice);
                }
            }

            // SAFETY: iter.remaining > 0 means back_ptr is valid
            let element = unsafe { &*iter.back_ptr.as_ptr() };
            let matched = (self.pred)(element);

            if matched && consumed > 0 {
                let result_start = slice_start + iter.remaining;
                let result_len = consumed;

                let result = SegmentedSliceMut {
                    buf: slice_buf,
                    start: result_start,
                    len: result_len,
                    end_ptr: slice_end_ptr,
                    end_seg: slice_end_seg,
                    _marker: PhantomData,
                };

                // Capture separator seg BEFORE advancing
                let separator_seg = iter.back_seg;

                // The separator terminates the previous chunk (the one on the left).
                // So the slice we store in `self.slice` ends AFTER the separator.
                // Wait. `end_ptr` should be S+1. S is at `iter.back_ptr`.
                let new_end_ptr = unsafe { NonNull::new_unchecked(iter.back_ptr.as_ptr().add(1)) };

                self.slice = SegmentedSliceMut {
                    buf: slice_buf,
                    start: slice_start,
                    len: iter.remaining,
                    end_ptr: new_end_ptr,
                    end_seg: separator_seg,
                    _marker: PhantomData,
                };

                return Some(result);
            }

            iter.next_back();
            consumed += 1;
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitInclusiveMut<'a, T, A, P> where
    P: FnMut(&T) -> bool
{
}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [`SegmentedSlice`].
///
/// [`rsplit`]: SegmentedSlice::rsplit
#[derive(Debug, Clone)]
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

impl<'a, T, A: Allocator + 'a, P> SplitIter for RSplit<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSlice<'a, T, A>> {
        self.inner.finish()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplit<'a, T, A, P> where P: FnMut(&T) -> bool {}

#[derive(Debug)]
pub struct RSplitMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: SplitMut<'a, T, A, P>,
}

impl<'a, T, A: Allocator + 'a, P> SplitIter for RSplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSliceMut<'a, T, A>> {
        self.inner.finish()
    }
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, A: Allocator + 'a, P> DoubleEndedIterator for RSplitMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplitMut<'a, T, A, P> where P: FnMut(&T) -> bool
{}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
struct GenericSplitN<I> {
    iter: I,
    count: usize,
}

impl<T, I: SplitIter<Item = T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => {
                self.count -= 1;
                self.iter.finish()
            }
            _ => {
                self.count -= 1;
                self.iter.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper_opt) = self.iter.size_hint();
        (
            cmp::min(self.count, lower),
            Some(upper_opt.map_or(self.count, |upper| cmp::min(self.count, upper))),
        )
    }
}

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
    inner: GenericSplitN<Split<'a, T, A, P>>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitN<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: Split::new(slice, pred),
                count: n,
            },
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
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
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
    inner: GenericSplitN<RSplit<'a, T, A, P>>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> RSplitN<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: RSplit::new(slice, pred),
                count: n,
            },
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
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplitN<'a, T, A, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [`SegmentedSliceMut`].
///
/// [`splitn_mut`]: SegmentedSliceMut::splitn_mut
pub struct SplitNMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<SplitMut<'a, T, A, P>>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> SplitNMut<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: SplitMut::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for SplitNMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for SplitNMut<'a, T, A, P> where P: FnMut(&T) -> bool
{}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits, starting from the end.
///
/// This struct is created by the [`rsplitn_mut`] method on [`SegmentedSliceMut`].
///
/// [`rsplitn_mut`]: SegmentedSliceMut::rsplitn_mut
pub struct RSplitNMut<'a, T, A: Allocator + 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<RSplitMut<'a, T, A, P>>,
}

impl<'a, T, A: Allocator + 'a, P: FnMut(&T) -> bool> RSplitNMut<'a, T, A, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: RSplitMut::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, A: Allocator + 'a, P> Iterator for RSplitNMut<'a, T, A, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, A: Allocator + 'a, P> FusedIterator for RSplitNMut<'a, T, A, P> where
    P: FnMut(&T) -> bool
{
}

/// An iterator over overlapping windows of a `SegmentedSlice`.
#[derive(Clone, Debug)]
pub struct Windows<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSlice<'a, T, A>,
    pub(crate) size: NonZeroUsize,
    // Cached end location for the current window.
    pub(crate) window_end_ptr: NonNull<T>,
    // Current segment index for window end pointer
    pub(crate) window_end_seg: usize,
    // End of current segment for window end pointer
    pub(crate) window_end_seg_end: NonNull<T>,
}

impl<'a, T: 'a, A: Allocator + 'a> Windows<'a, T, A> {
    #[inline]
    pub(super) fn new(slice: SegmentedSlice<'a, T, A>, size: NonZeroUsize) -> Self {
        let size_val = size.get();
        if slice.len < size_val {
            return Self {
                slice,
                size,
                window_end_ptr: NonNull::dangling(),
                window_end_seg: 0,
                window_end_seg_end: NonNull::dangling(),
            };
        }

        let end_index = slice.start + size_val;
        // This `location` call happens once upon creation.
        let (mut window_end_seg, mut window_end_offset) =
            RawSegmentedVec::<T, A>::location(end_index);

        let buf = unsafe { slice.buf.as_ref() };
        if window_end_seg >= buf.segment_count() {
            // If we point past the last segment, backtrack to the end of the previous segment.
            // This happens when the window ends exactly at the capacity limit.
            if window_end_seg > 0 {
                window_end_seg -= 1;
                window_end_offset = RawSegmentedVec::<T, A>::segment_capacity(window_end_seg);
            }
        }

        let window_end_ptr = unsafe {
            let ptr = buf.segment_ptr(window_end_seg).add(window_end_offset);
            NonNull::new_unchecked(ptr)
        };

        let window_end_seg_end = unsafe {
            if window_end_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(window_end_seg);
                NonNull::new_unchecked(buf.segment_ptr(window_end_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            slice,
            size,
            window_end_ptr,
            window_end_seg,
            window_end_seg_end,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for Windows<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let size = self.size.get();
        if self.slice.len() < size {
            return None;
        }

        // Construct the window slice using cached end pointers.
        // We use struct initialization directly to bypass `SegmentedSlice::new` which recalculates.
        let window = SegmentedSlice {
            buf: self.slice.buf,
            start: self.slice.start,
            len: size,
            end_ptr: self.window_end_ptr,
            end_seg: self.window_end_seg,
            _marker: PhantomData,
        };

        // Advance start
        self.slice.start += 1;
        self.slice.len -= 1;

        // Advance window_end
        // Logic: if window_end_ptr == window_end_seg_end, move to next segment.

        let buf = unsafe { self.slice.buf.as_ref() };

        // Non-ZST logic
        if core::mem::size_of::<T>() > 0 {
            // Check if we reached the boundary of the current segment
            if self.window_end_ptr == self.window_end_seg_end {
                // Move to next segment
                self.window_end_seg += 1;

                if self.window_end_seg < buf.segment_count() {
                    let ptr = unsafe { buf.segment_ptr(self.window_end_seg) };
                    // Set ptr to start of next segment
                    self.window_end_ptr = unsafe { NonNull::new_unchecked(ptr) };
                    let cap = RawSegmentedVec::<T, A>::segment_capacity(self.window_end_seg);
                    self.window_end_seg_end = unsafe { NonNull::new_unchecked(ptr.add(cap)) };
                } else {
                    // Should be covered by slice.len check, but for safety:
                    self.window_end_ptr = NonNull::dangling();
                    self.window_end_seg_end = NonNull::dangling();
                }
            }

            // Now advance pointer by 1
            // If we just switched segments, we are at start. Adding 1 moves to index 1.
            // This is correct (see thought process).
            self.window_end_ptr =
                unsafe { NonNull::new_unchecked(self.window_end_ptr.as_ptr().add(1)) };
        } else {
            let end_index = self.slice.start + self.size.get();
            let (seg, _) = RawSegmentedVec::<T, A>::location(end_index);
            self.window_end_seg = seg;
        }

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

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let size = self.size.get();
        if self.slice.len() < size {
            return None;
        }
        let windows_count = self.slice.len() - size + 1;
        if n >= windows_count {
            // Consumed all
            self.slice.start += windows_count;
            self.slice.len -= windows_count;
            return None;
        }

        // Advance by n
        self.slice.start += n;
        self.slice.len -= n;

        // Recalculate cached window_end fields
        let end_index = self.slice.start + size;
        let (mut seg, mut offset) = RawSegmentedVec::<T, A>::location(end_index);

        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                offset = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }

        self.window_end_seg = seg;

        let ptr = unsafe { buf.segment_ptr(seg).add(offset) };
        self.window_end_ptr = unsafe { NonNull::new_unchecked(ptr) };

        self.window_end_seg_end = unsafe {
            if seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                NonNull::new_unchecked(buf.segment_ptr(seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for Windows<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let size = self.size.get();
        if self.slice.len() < size {
            return None;
        }

        // The last window starts at `len - size`.
        // Relative to `self.slice.start`:
        // start_index = self.slice.start + (self.slice.len() - size);
        let start_index = self.slice.start + self.slice.len() - size;
        let end_index = start_index + size; // = self.slice.start + self.slice.len()

        let window = SegmentedSlice::new(self.slice.buf, start_index, end_index);

        // Shrink the slice from the end
        self.slice.len -= 1;

        Some(window)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if self.slice.len() < self.size.get() {
            return None;
        }
        let windows_count = self.slice.len() - self.size.get() + 1;
        if n >= windows_count {
            // Consumed all
            self.slice.len -= windows_count;
            return None;
        }

        // Decrement len by n
        self.slice.len -= n;

        self.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for Windows<'_, T, A> {}
impl<T, A: Allocator> FusedIterator for Windows<'_, T, A> {}

/// An iterator over chunks of a `SegmentedSlice`.
#[derive(Debug, Clone)]
pub struct Chunks<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSlice<'a, T, A>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T, A: Allocator + 'a> Chunks<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                slice,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
            };
        }

        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        let buf = unsafe { slice.buf.as_ref() };

        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }

        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };

        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            slice,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for Chunks<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = std::cmp::min(self.slice.len, self.chunk_size);

            let chunk_start = self.slice.start;
            let chunk_end_idx = chunk_start + chunk_len;

            // Use cached start info
            let chunk_start_ptr = self.start_ptr;
            let chunk_start_seg = self.start_seg;

            // Identify chunk end pointer and segment.
            // If the chunk is contained within the current segment:
            let (chunk_end_ptr, chunk_end_seg) = unsafe {
                let end_ptr = chunk_start_ptr.as_ptr().add(chunk_len);
                if end_ptr <= self.start_seg_end.as_ptr() {
                    (NonNull::new_unchecked(end_ptr), chunk_start_seg)
                } else {
                    // It crosses segment boundary.
                    // Instead of full location(), we can compute it since we know it's relatively small cross.
                    // But for simplicity and correctness first, we'll use location() or a helper.
                    // Actually, let's just use self.slice.buf.ptr_at(chunk_end_idx) which is optimized.
                    let buf = self.slice.buf.as_ref();
                    let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_end_idx);
                    let (seg, ptr) = if seg >= buf.segment_count() {
                        let last_seg = seg - 1;
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                        (last_seg, buf.segment_ptr(last_seg).add(cap))
                    } else {
                        (seg, buf.segment_ptr(seg).add(off))
                    };
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let chunk = SegmentedSlice {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: chunk_end_ptr,
                end_seg: chunk_end_seg,
                _marker: PhantomData,
            };

            // Update iterator state
            self.slice.start += chunk_len;
            self.slice.len -= chunk_len;

            // Update start cache for next call
            if self.slice.len > 0 {
                self.start_ptr = chunk_end_ptr;
                self.start_seg = chunk_end_seg;
                // If we crossed segments, update the segment end too
                if chunk_end_seg != chunk_start_seg {
                    unsafe {
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(chunk_end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            self.slice.buf.as_ref().segment_ptr(chunk_end_seg).add(cap),
                        );
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let chunks = self.slice.len.div_ceil(self.chunk_size);
        (chunks, Some(chunks))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.checked_mul(self.chunk_size)?;
        if skip >= self.slice.len {
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            return None;
        }

        self.slice.start += skip;
        self.slice.len -= skip;

        // Re-initialize cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.slice.start);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for Chunks<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let remainder = self.slice.len % self.chunk_size;
            let chunk_len = if remainder != 0 {
                remainder
            } else {
                self.chunk_size
            };

            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let chunk = SegmentedSlice {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.slice.end_ptr,
                end_seg: self.slice.end_seg,
                _marker: PhantomData,
            };

            self.slice.len -= chunk_len;

            // Update self.slice.end_ptr and end_seg for next calls.
            if self.slice.len > 0 {
                unsafe {
                    let buf = self.slice.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_start);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.slice.end_ptr = NonNull::new_unchecked(ptr);
                        self.slice.end_seg = seg;
                    }
                }
            } else {
                self.slice.end_ptr = NonNull::dangling();
                // start_ptr was already handled in next() if it reached 0
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(chunk)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.slice.end_ptr = NonNull::dangling();
            return None;
        }

        let total_chunks = self.count();
        let index_from_start = total_chunks - 1 - n;
        let relative_start = index_from_start * self.chunk_size;

        let chunk_len = std::cmp::min(self.chunk_size, self.slice.len - relative_start);
        let chunk_start_abs = self.slice.start + relative_start;
        let chunk_end_abs = chunk_start_abs + chunk_len;

        self.slice.len = relative_start;

        // Re-initialize end cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(chunk_end_abs);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for Chunks<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for Chunks<'_, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks.
///
/// This struct is created by the [`chunks_mut`] method on [`SegmentedSlice`].
///
/// [`chunks_mut`]: SegmentedSlice::chunks_mut
pub struct ChunksMut<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSliceMut<'a, T, A>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T, A: Allocator + 'a> ChunksMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                slice,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            slice,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = std::cmp::min(self.chunk_size, self.slice.len);
            let chunk_start = self.slice.start;

            // Use cached start info
            let start_ptr = self.start_ptr;
            let start_seg = self.start_seg;

            // Identify chunk end pointer and segment.
            let (end_ptr, end_seg) = unsafe {
                let end_ptr = start_ptr.as_ptr().add(chunk_len);
                if end_ptr <= self.start_seg_end.as_ptr() {
                    (NonNull::new_unchecked(end_ptr), start_seg)
                } else {
                    // It crosses segment boundary.
                    let chunk_end_idx = chunk_start + chunk_len;
                    let buf = self.slice.buf.as_ref();
                    let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_end_idx);
                    let (seg, ptr) = if seg >= buf.segment_count() {
                        let last_seg = seg - 1;
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                        (last_seg, buf.segment_ptr(last_seg).add(cap))
                    } else {
                        (seg, buf.segment_ptr(seg).add(off))
                    };
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSliceMut {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            // Update iterator state
            self.slice.start += chunk_len;
            self.slice.len -= chunk_len;

            // Update start cache for next call
            if self.slice.len > 0 {
                self.start_ptr = end_ptr;
                self.start_seg = end_seg;
                // If we crossed segments, update the segment end too
                if end_seg != start_seg {
                    unsafe {
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            self.slice.buf.as_ref().segment_ptr(end_seg).add(cap),
                        );
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.len == 0 {
            (0, Some(0))
        } else {
            let n = self.slice.len.div_ceil(self.chunk_size);
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.checked_mul(self.chunk_size)?;
        if skip >= self.slice.len {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            return None;
        }

        self.slice.start += skip;
        self.slice.len -= skip;

        // Re-initialize start cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.slice.start);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };

        self.next()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for ChunksMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let remainder = self.slice.len % self.chunk_size;
            let chunk_len = if remainder != 0 {
                remainder
            } else {
                self.chunk_size
            };

            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let result = SegmentedSliceMut {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.slice.end_ptr,
                end_seg: self.slice.end_seg,
                _marker: PhantomData,
            };

            self.slice.len -= chunk_len;

            // Update self.slice.end_ptr and end_seg for next calls.
            if self.slice.len > 0 {
                unsafe {
                    let buf = self.slice.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_start);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.slice.end_ptr = NonNull::new_unchecked(ptr);
                        self.slice.end_seg = seg;
                    }
                }
            } else {
                self.slice.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.slice.end_ptr = NonNull::dangling();
            return None;
        }

        let skip_len = n * self.chunk_size;
        self.slice.len -= skip_len;

        // Re-initialize end cache
        let (mut seg, mut off) =
            RawSegmentedVec::<T, A>::location(self.slice.start + self.slice.len);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for ChunksMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for ChunksMut<'a, T, A> {}
unsafe impl<T, A: Allocator> Send for ChunksMut<'_, T, A> where T: Send {}
unsafe impl<T, A: Allocator> Sync for ChunksMut<'_, T, A> where T: Sync {}

#[derive(Debug, Clone)]
pub struct ChunksExact<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSlice<'a, T, A>,
    pub(crate) rem: SegmentedSlice<'a, T, A>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T, A: Allocator + 'a> ChunksExact<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem_len = len % chunk_size;
        let exact_len = len - rem_len;

        let rem = slice.sub_slice(exact_len..len);
        let exact_slice = slice.sub_slice(0..exact_len);

        if exact_len == 0 {
            return Self {
                slice: exact_slice,
                rem,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            slice: exact_slice,
            rem,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
        }
    }

    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    pub fn remainder(&self) -> SegmentedSlice<'a, T, A> {
        self.rem
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksExact<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start;

            // Use cached start info
            let start_ptr = self.start_ptr;
            let start_seg = self.start_seg;

            // Identify chunk end pointer and segment.
            let (end_ptr, end_seg) = unsafe {
                let end_ptr = start_ptr.as_ptr().add(chunk_len);
                if end_ptr <= self.start_seg_end.as_ptr() {
                    (NonNull::new_unchecked(end_ptr), start_seg)
                } else {
                    // It crosses segment boundary.
                    let chunk_end_idx = chunk_start + chunk_len;
                    let buf = self.slice.buf.as_ref();
                    let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_end_idx);
                    let (seg, ptr) = if seg >= buf.segment_count() {
                        let last_seg = seg - 1;
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                        (last_seg, buf.segment_ptr(last_seg).add(cap))
                    } else {
                        (seg, buf.segment_ptr(seg).add(off))
                    };
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSlice {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            // Update iterator state
            self.slice.start += chunk_len;
            self.slice.len -= chunk_len;

            // Update start cache for next call
            if self.slice.len > 0 {
                self.start_ptr = end_ptr;
                self.start_seg = end_seg;
                // If we crossed segments, update the segment end too
                if end_seg != start_seg {
                    unsafe {
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            self.slice.buf.as_ref().segment_ptr(end_seg).add(cap),
                        );
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.len == 0 {
            (0, Some(0))
        } else {
            let chunks = self.slice.len / self.chunk_size;
            (chunks, Some(chunks))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.checked_mul(self.chunk_size)?;
        if skip >= self.slice.len {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            return None;
        }

        self.slice.start += skip;
        self.slice.len -= skip;

        // Re-initialize start cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.slice.start);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for ChunksExact<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let result = SegmentedSlice {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.slice.end_ptr,
                end_seg: self.slice.end_seg,
                _marker: PhantomData,
            };

            self.slice.len -= chunk_len;

            // Update self.slice.end_ptr and end_seg for next calls.
            if self.slice.len > 0 {
                unsafe {
                    let buf = self.slice.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_start);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.slice.end_ptr = NonNull::new_unchecked(ptr);
                        self.slice.end_seg = seg;
                    }
                }
            } else {
                self.slice.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.slice.end_ptr = NonNull::dangling();
            return None;
        }

        let skip_len = n * self.chunk_size;
        self.slice.len -= skip_len;

        // Re-initialize end cache
        let (mut seg, mut off) =
            RawSegmentedVec::<T, A>::location(self.slice.start + self.slice.len);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for ChunksExact<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for ChunksExact<'_, T, A> {}

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
#[derive(Debug)]
pub struct ChunksExactMut<'a, T, A: Allocator + 'a = Global> {
    pub(crate) slice: SegmentedSliceMut<'a, T, A>,
    pub(crate) rem: SegmentedSliceMut<'a, T, A>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T, A: Allocator + 'a> ChunksExactMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem_len = len % chunk_size;
        let exact_len = len - rem_len;

        let rem = SegmentedSliceMut::new(slice.buf, slice.start + exact_len, slice.start + len);
        let exact_slice = SegmentedSliceMut::new(slice.buf, slice.start, slice.start + exact_len);

        if exact_len == 0 {
            return Self {
                slice: exact_slice,
                rem,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            slice: exact_slice,
            rem,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
        }
    }

    /// Returns the remainder of the original slice that is not covered by
    /// the iterator.
    #[inline]
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T, A> {
        self.rem
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for ChunksExactMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start;

            // Use cached start info
            let start_ptr = self.start_ptr;
            let start_seg = self.start_seg;

            // Identify chunk end pointer and segment.
            let (end_ptr, end_seg) = unsafe {
                let end_ptr = start_ptr.as_ptr().add(chunk_len);
                if end_ptr <= self.start_seg_end.as_ptr() {
                    (NonNull::new_unchecked(end_ptr), start_seg)
                } else {
                    // It crosses segment boundary.
                    let chunk_end_idx = chunk_start + chunk_len;
                    let buf = self.slice.buf.as_ref();
                    let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_end_idx);
                    let (seg, ptr) = if seg >= buf.segment_count() {
                        let last_seg = seg - 1;
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                        (last_seg, buf.segment_ptr(last_seg).add(cap))
                    } else {
                        (seg, buf.segment_ptr(seg).add(off))
                    };
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSliceMut {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            // Update iterator state
            self.slice.start += chunk_len;
            self.slice.len -= chunk_len;

            // Update start cache for next call
            if self.slice.len > 0 {
                self.start_ptr = end_ptr;
                self.start_seg = end_seg;
                // If we crossed segments, update the segment end too
                if end_seg != start_seg {
                    unsafe {
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            self.slice.buf.as_ref().segment_ptr(end_seg).add(cap),
                        );
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.len == 0 {
            (0, Some(0))
        } else {
            let n = self.slice.len / self.chunk_size;
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.checked_mul(self.chunk_size)?;
        if skip >= self.slice.len {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            return None;
        }

        self.slice.start += skip;
        self.slice.len -= skip;

        // Re-initialize start cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.slice.start);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for ChunksExactMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let result = SegmentedSliceMut {
                buf: self.slice.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.slice.end_ptr,
                end_seg: self.slice.end_seg,
                _marker: PhantomData,
            };

            self.slice.len -= chunk_len;

            // Update self.slice.end_ptr and end_seg for next calls.
            if self.slice.len > 0 {
                unsafe {
                    let buf = self.slice.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(chunk_start);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.slice.end_ptr = NonNull::new_unchecked(ptr);
                        self.slice.end_seg = seg;
                    }
                }
            } else {
                self.slice.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.slice.start += self.slice.len;
            self.slice.len = 0;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.slice.end_ptr = NonNull::dangling();
            return None;
        }

        let skip_len = n * self.chunk_size;
        self.slice.len -= skip_len;

        // Re-initialize end cache
        let (mut seg, mut off) =
            RawSegmentedVec::<T, A>::location(self.slice.start + self.slice.len);
        let buf = unsafe { self.slice.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for ChunksExactMut<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for ChunksExactMut<'_, T, A> {}

/// An iterator over chunks of a `SegmentedSlice`, starting from the end.
#[derive(Debug, Clone)]
pub struct RChunks<'a, T, A: Allocator + 'a = Global> {
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current range.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
    /// Cached pointer to the end (one-past-last) of the current range.
    pub(crate) end_ptr: NonNull<T>,
    /// Current segment index for end pointer.
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator + 'a> RChunks<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                buf: slice.buf,
                start: slice.start,
                end: slice.start,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
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
            let chunk_len = if self.end - self.start >= self.chunk_size {
                self.chunk_size
            } else {
                self.end - self.start
            };
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let chunk = SegmentedSlice {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: PhantomData,
            };

            self.end -= chunk_len;

            // Update self.end_ptr and end_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let buf = self.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.end);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.end_ptr = NonNull::new_unchecked(ptr);
                        self.end_seg = seg;
                    }
                }
            } else {
                self.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.start);
        let chunks = remaining.div_ceil(self.chunk_size);
        (chunks, Some(chunks))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        // Adjust for possible short first chunk (which is the last one in RChunks next() logic)
        // Wait, RChunks::next() takes from the end.
        // So n=0: takes last self.chunk_size.
        // If len=10, chunk_size=3:
        // total_chunks = 4.
        // n=0: chunk from 7 to 10.
        // n=1: chunk from 4 to 7.
        // n=2: chunk from 1 to 4.
        // n=3: chunk from 0 to 1.

        let chunk_end_abs =
            std::cmp::min(self.end, self.start + (total_chunks - n) * self.chunk_size);
        self.end = chunk_end_abs;

        // Re-initialize end cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.end);
        let buf = unsafe { self.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunks<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let total_len = self.end - self.start;
            let rem = total_len % self.chunk_size;
            let chunk_len = if rem == 0 { self.chunk_size } else { rem };

            let chunk_start = self.start;
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_start + chunk_len);
            let buf = unsafe { self.buf.as_ref() };
            let end_ptr = if end_seg >= buf.segment_count() {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T, A>::location(chunk_start + chunk_len - 1);
                unsafe { NonNull::new_unchecked(buf.segment_ptr(last_seg).add(last_offset + 1)) }
            } else {
                unsafe { NonNull::new_unchecked(buf.segment_ptr(end_seg).add(end_offset)) }
            };

            let chunk = SegmentedSlice {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            self.start += chunk_len;

            // Update self.start_ptr and start_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let _buf = self.buf.as_ref();
                    let available =
                        self.start_seg_end
                            .as_ptr()
                            .offset_from(self.start_ptr.as_ptr()) as usize;

                    if available > chunk_len {
                        // Same segment
                        self.start_ptr =
                            NonNull::new_unchecked(self.start_ptr.as_ptr().add(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
                        let ptr = buf.segment_ptr(seg).add(off);
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                        let seg_end = buf.segment_ptr(seg).add(cap);

                        self.start_ptr = NonNull::new_unchecked(ptr);
                        self.start_seg = seg;
                        self.start_seg_end = NonNull::new_unchecked(seg_end);
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
                // end cache also reset if we met
                self.end_ptr = NonNull::dangling();
            }

            Some(chunk)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        let total_len = self.end - self.start;
        let rem = total_len % self.chunk_size;
        let first_chunk_len = if rem == 0 { self.chunk_size } else { rem };

        let to_skip = if n > 0 {
            first_chunk_len + (n - 1) * self.chunk_size
        } else {
            0
        };

        if n > 0 {
            self.start += to_skip;
            // Re-initialize start cache
            let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
            let buf = unsafe { self.buf.as_ref() };
            let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
            self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };
        }

        self.next_back()
    }
}

impl<T, A: Allocator> ExactSizeIterator for RChunks<'_, T, A> {}
impl<T, A: Allocator> std::iter::FusedIterator for RChunks<'_, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks,
/// starting at the end of the slice.
///
/// This struct is created by the [`rchunks_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_mut`]: SegmentedSlice::rchunks_mut
#[derive(Debug)]
pub struct RChunksMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    start: usize,
    end: usize,
    chunk_size: usize,
    /// Cached pointer to the start of the current range.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
    /// Cached pointer to the end (one-past-last) of the current range.
    pub(crate) end_ptr: NonNull<T>,
    /// Current segment index for end pointer.
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> RChunksMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                buf: slice.buf,
                start: slice.start,
                end: slice.start,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(slice.start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            buf: slice.buf,
            start: slice.start,
            end: slice.start + slice.len,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
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
            let chunk_len = if self.end - self.start >= self.chunk_size {
                self.chunk_size
            } else {
                self.end - self.start
            };
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let chunk = SegmentedSliceMut {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: PhantomData,
            };

            self.end -= chunk_len;

            // Update self.end_ptr and end_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let buf = self.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.end);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.end_ptr = NonNull::new_unchecked(ptr);
                        self.end_seg = seg;
                    }
                }
            } else {
                self.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

            Some(chunk)
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
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        let chunk_end_abs =
            std::cmp::min(self.end, self.start + (total_chunks - n) * self.chunk_size);
        self.end = chunk_end_abs;

        // Re-initialize end cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.end);
        let buf = unsafe { self.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let total_len = self.end - self.start;
            let rem = total_len % self.chunk_size;
            let chunk_len = if rem == 0 { self.chunk_size } else { rem };

            let chunk_start = self.start;
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_start + chunk_len);
            let buf = unsafe { self.buf.as_ref() };
            let end_ptr = if end_seg >= buf.segment_count() {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T, A>::location(chunk_start + chunk_len - 1);
                unsafe { NonNull::new_unchecked(buf.segment_ptr(last_seg).add(last_offset + 1)) }
            } else {
                unsafe { NonNull::new_unchecked(buf.segment_ptr(end_seg).add(end_offset)) }
            };

            let chunk = SegmentedSliceMut {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            self.start += chunk_len;

            // Update self.start_ptr and start_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let _buf = self.buf.as_ref();
                    let available =
                        self.start_seg_end
                            .as_ptr()
                            .offset_from(self.start_ptr.as_ptr()) as usize;

                    if available > chunk_len {
                        // Same segment
                        self.start_ptr =
                            NonNull::new_unchecked(self.start_ptr.as_ptr().add(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
                        let ptr = buf.segment_ptr(seg).add(off);
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                        let seg_end = buf.segment_ptr(seg).add(cap);

                        self.start_ptr = NonNull::new_unchecked(ptr);
                        self.start_seg = seg;
                        self.start_seg_end = NonNull::new_unchecked(seg_end);
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
                // end cache also reset if we met
                self.end_ptr = NonNull::dangling();
            }

            Some(chunk)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        let total_len = self.end - self.start;
        let rem = total_len % self.chunk_size;
        let first_chunk_len = if rem == 0 { self.chunk_size } else { rem };

        let to_skip = if n > 0 {
            first_chunk_len + (n - 1) * self.chunk_size
        } else {
            0
        };

        if n > 0 {
            self.start += to_skip;
            // Re-initialize start cache
            let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
            let buf = unsafe { self.buf.as_ref() };
            let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
            self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };
        }

        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for RChunksMut<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for RChunksMut<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) chunks of exact
/// size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact`]: SegmentedSlice::rchunks_exact
pub struct RChunksExact<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    /// Start of the exact chunks (excludes remainder)
    pub(crate) start: usize,
    pub(crate) end: usize,
    /// Start of the full slice (includes remainder)
    pub(crate) full_start: usize,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current range.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
    /// Cached pointer to the end (one-past-last) of the current range.
    pub(crate) end_ptr: NonNull<T>,
    /// Current segment index for end pointer.
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator + 'a> RChunksExact<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        let exact_end = slice.start + len;

        if exact_start == exact_end {
            return Self {
                buf: slice.buf,
                start: exact_start,
                end: exact_end,
                full_start: slice.start,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(exact_start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            buf: slice.buf,
            start: exact_start,
            end: exact_end,
            full_start: slice.start,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
            _marker: PhantomData,
        }
    }

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
            start_ptr: self.start_ptr,
            start_seg: self.start_seg,
            start_seg_end: self.start_seg_end,
            end_ptr: self.end_ptr,
            end_seg: self.end_seg,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunksExact<'a, T, A> {
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let result = SegmentedSlice {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: PhantomData,
            };

            self.end -= chunk_len;

            // Update self.end_ptr and end_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let buf = self.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.end);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.end_ptr = NonNull::new_unchecked(ptr);
                        self.end_seg = seg;
                    }
                }
            } else {
                self.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

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
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        let chunk_end_abs = self.end - n * self.chunk_size;
        self.end = chunk_end_abs;

        // Re-initialize end cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.end);
        let buf = unsafe { self.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksExact<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.start;

            // The chunk starts at self.start_ptr/start_seg.
            // We need the end of THIS chunk.
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_start + chunk_len);
            let buf = unsafe { self.buf.as_ref() };
            let end_ptr = if end_seg >= buf.segment_count() {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T, A>::location(chunk_start + chunk_len - 1);
                unsafe { NonNull::new_unchecked(buf.segment_ptr(last_seg).add(last_offset + 1)) }
            } else {
                unsafe { NonNull::new_unchecked(buf.segment_ptr(end_seg).add(end_offset)) }
            };

            let result = SegmentedSlice {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            self.start += chunk_len;

            // Update self.start_ptr and start_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let _buf = self.buf.as_ref();
                    let available =
                        self.start_seg_end
                            .as_ptr()
                            .offset_from(self.start_ptr.as_ptr()) as usize;

                    if available > chunk_len {
                        // Same segment
                        self.start_ptr =
                            NonNull::new_unchecked(self.start_ptr.as_ptr().add(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
                        let ptr = buf.segment_ptr(seg).add(off);
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                        let seg_end = buf.segment_ptr(seg).add(cap);

                        self.start_ptr = NonNull::new_unchecked(ptr);
                        self.start_seg = seg;
                        self.start_seg_end = NonNull::new_unchecked(seg_end);
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
                // end cache also reset if we met
                self.end_ptr = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        if n > 0 {
            self.start += n * self.chunk_size;
            // Re-initialize start cache
            let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
            let buf = unsafe { self.buf.as_ref() };
            let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
            self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };
        }

        self.next_back()
    }
}

impl<'a, T, A: Allocator + 'a> ExactSizeIterator for RChunksExact<'a, T, A> {}
impl<'a, T, A: Allocator + 'a> FusedIterator for RChunksExact<'a, T, A> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks of
/// exact size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact_mut`]: SegmentedSlice::rchunks_exact_mut
pub struct RChunksExactMut<'a, T, A: Allocator + 'a = Global> {
    buf: NonNull<RawSegmentedVec<T, A>>,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) full_start: usize,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current range.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
    /// Cached pointer to the end (one-past-last) of the current range.
    pub(crate) end_ptr: NonNull<T>,
    /// Current segment index for end pointer.
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> RChunksExactMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        let exact_end = slice.start + len;

        if exact_start == exact_end {
            return Self {
                buf: slice.buf,
                start: exact_start,
                end: exact_end,
                full_start: slice.start,
                chunk_size,
                start_ptr: NonNull::dangling(),
                start_seg: 0,
                start_seg_end: NonNull::dangling(),
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let buf = unsafe { slice.buf.as_ref() };

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T, A>::location(exact_start);
        if start_seg >= buf.segment_count() {
            if start_seg > 0 {
                start_seg -= 1;
                start_offset = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
            }
        }
        let start_ptr = unsafe {
            let ptr = buf.segment_ptr(start_seg).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if start_seg < buf.segment_count() {
                let cap = RawSegmentedVec::<T, A>::segment_capacity(start_seg);
                NonNull::new_unchecked(buf.segment_ptr(start_seg).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            buf: slice.buf,
            start: exact_start,
            end: exact_end,
            full_start: slice.start,
            chunk_size,
            start_ptr,
            start_seg,
            start_seg_end,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T, A> {
        SegmentedSliceMut::new(self.buf, self.full_start, self.start)
    }
}

impl<'a, T, A: Allocator + 'a> Iterator for RChunksExactMut<'a, T, A> {
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let result = SegmentedSliceMut {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: PhantomData,
            };

            self.end -= chunk_len;

            // Update self.end_ptr and end_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let buf = self.buf.as_ref();
                    let seg_ptr = buf.segment_ptr(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.end);
                        let (seg, ptr) = if seg >= buf.segment_count() {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T, A>::segment_capacity(last_seg);
                            (last_seg, buf.segment_ptr(last_seg).add(cap))
                        } else {
                            (seg, buf.segment_ptr(seg).add(off))
                        };
                        self.end_ptr = NonNull::new_unchecked(ptr);
                        self.end_seg = seg;
                    }
                }
            } else {
                self.end_ptr = NonNull::dangling();
                // start cache also reset if we met
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
            }

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
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        let chunk_end_abs = self.end - n * self.chunk_size;
        self.end = chunk_end_abs;

        // Re-initialize end cache
        let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(self.end);
        let buf = unsafe { self.buf.as_ref() };
        if seg >= buf.segment_count() {
            if seg > 0 {
                seg -= 1;
                off = RawSegmentedVec::<T, A>::segment_capacity(seg);
            }
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };

        self.next()
    }
}

impl<'a, T, A: Allocator + 'a> DoubleEndedIterator for RChunksExactMut<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.start;

            // The chunk starts at self.start_ptr/start_seg.
            // We need the end of THIS chunk.
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(chunk_start + chunk_len);
            let buf = unsafe { self.buf.as_ref() };
            let end_ptr = if end_seg >= buf.segment_count() {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T, A>::location(chunk_start + chunk_len - 1);
                unsafe { NonNull::new_unchecked(buf.segment_ptr(last_seg).add(last_offset + 1)) }
            } else {
                unsafe { NonNull::new_unchecked(buf.segment_ptr(end_seg).add(end_offset)) }
            };

            let result = SegmentedSliceMut {
                buf: self.buf,
                start: chunk_start,
                len: chunk_len,
                end_ptr,
                end_seg,
                _marker: PhantomData,
            };

            self.start += chunk_len;

            // Update self.start_ptr and start_seg for next calls.
            if self.start < self.end {
                unsafe {
                    let _buf = self.buf.as_ref();
                    let available =
                        self.start_seg_end
                            .as_ptr()
                            .offset_from(self.start_ptr.as_ptr()) as usize;

                    if available > chunk_len {
                        // Same segment
                        self.start_ptr =
                            NonNull::new_unchecked(self.start_ptr.as_ptr().add(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
                        let ptr = buf.segment_ptr(seg).add(off);
                        let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                        let seg_end = buf.segment_ptr(seg).add(cap);

                        self.start_ptr = NonNull::new_unchecked(ptr);
                        self.start_seg = seg;
                        self.start_seg_end = NonNull::new_unchecked(seg_end);
                    }
                }
            } else {
                self.start_ptr = NonNull::dangling();
                self.start_seg_end = NonNull::dangling();
                // end cache also reset if we met
                self.end_ptr = NonNull::dangling();
            }

            Some(result)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let total_chunks = self.size_hint().0;
        if n >= total_chunks {
            self.start = self.end;
            self.start_ptr = NonNull::dangling();
            self.start_seg_end = NonNull::dangling();
            self.end_ptr = NonNull::dangling();
            return None;
        }

        if n > 0 {
            self.start += n * self.chunk_size;
            // Re-initialize start cache
            let (seg, off) = RawSegmentedVec::<T, A>::location(self.start);
            let buf = unsafe { self.buf.as_ref() };
            let cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(off)) };
            self.start_seg_end = unsafe { NonNull::new_unchecked(buf.segment_ptr(seg).add(cap)) };
        }

        self.next_back()
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

unsafe impl<T: Send, A: Allocator + Send> Send for SliceIterMut<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for SliceIterMut<'_, T, A> {}

/// An iterator over subslices separated by elements that match a predicate function.
#[derive(Debug, Clone)]
pub struct ChunkBy<'a, T, F, A: Allocator + 'a = Global> {
    slice: SegmentedSlice<'a, T, A>,
    pred: F,
}

impl<'a, T, F, A: Allocator + 'a> ChunkBy<'a, T, F, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T, A>, pred: F) -> Self {
        Self { slice, pred }
    }
}

impl<'a, T, F, A: Allocator + 'a> Iterator for ChunkBy<'a, T, F, A>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSlice<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let len = self.slice.len();
        // Find the index of the first element that does not satisfy the predicate with the next element.
        // We look for i such that !pred(slice[i], slice[i+1]).
        // The chunk includes slice[i], so the split point is i + 1.
        let split_point = self
            .slice
            .iter()
            .zip(self.slice.iter().skip(1))
            .position(|(prev, next)| !(self.pred)(prev, next))
            .map(|i| i + 1)
            .unwrap_or(len);

        let (head, tail) = self.slice.split_at(split_point);
        self.slice = tail;
        Some(head)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            (1, Some(self.slice.len()))
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, F, A: Allocator + 'a> DoubleEndedIterator for ChunkBy<'a, T, F, A>
where
    F: FnMut(&T, &T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        // Find the index of the last element that does not satisfy the predicate with the previous element.
        let split_point = self
            .slice
            .iter()
            .zip(self.slice.iter().skip(1))
            .rposition(|(prev, next)| !(self.pred)(prev, next))
            .map(|i| i + 1)
            .unwrap_or(0);

        let (head, tail) = self.slice.split_at(split_point);
        self.slice = head;
        Some(tail)
    }
}

impl<'a, T, F, A: Allocator + 'a> FusedIterator for ChunkBy<'a, T, F, A> where
    F: FnMut(&T, &T) -> bool
{
}

/// An iterator over mutable subslices separated by elements that match a predicate function.
pub struct ChunkByMut<'a, T, F, A: Allocator + 'a = Global> {
    slice: SegmentedSliceMut<'a, T, A>,
    pred: F,
}

impl<'a, T, F, A: Allocator + 'a> ChunkByMut<'a, T, F, A> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T, A>, pred: F) -> Self {
        Self { slice, pred }
    }
}

impl<'a, T, F, A: Allocator + 'a> Iterator for ChunkByMut<'a, T, F, A>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T, A>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let len = self.slice.len();
        // To find the split point, we need to inspect elements.
        // We can create a temporary immutable view of the remaining mutable slice.
        // This is safe because we only hold it for finding the index and then drop it
        // before performing the mutable split.
        let temp_view = SegmentedSlice {
            buf: self.slice.buf,
            start: self.slice.start,
            len: self.slice.len,
            end_ptr: self.slice.end_ptr,
            end_seg: self.slice.end_seg,
            _marker: PhantomData,
        };

        let split_point = temp_view
            .iter()
            .zip(temp_view.iter().skip(1))
            .position(|(prev, next)| !(self.pred)(prev, next))
            .map(|i| i + 1)
            .unwrap_or(len);

        // Now we can split safely
        // split_at_mut consumes self (or effectively reborrows if we implemented it that way),
        // but here we want to update self.slice.
        // SegmentedSliceMut methods usually return new slices.
        // We need `split_at_mut` behavior on `self.slice`.
        // Let's rely on `split_at_mut_checked` or similar if available, or just construct manually/use logic.
        // SegmentedSliceMut::split_at_mut returns (head, tail).

        // We need to temporarily take `self.slice` out or clone the metadata (it's Copy-ish but not Copy).
        // Actually SegmentedSliceMut is not Copy.
        // But we can duplicate the metadata since we have exclusive access to `self`.

        // Let's use a helper or manual split to avoid fighting ownership if `split_at_mut` consumes.
        // Looking at `SegmentedSliceMut`, `split_at_mut` likely consumes or reborrows.
        // We can just manually split:

        let head_len = split_point;

        // Construct head
        let head = SegmentedSliceMut::new(
            self.slice.buf,
            self.slice.start,
            self.slice.start + head_len,
        );

        // Update self.slice to tail
        self.slice = SegmentedSliceMut::new(
            self.slice.buf,
            self.slice.start + head_len,
            self.slice.start + len,
        ); // start + len is end

        Some(head)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            (1, Some(self.slice.len()))
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T, F, A: Allocator + 'a> DoubleEndedIterator for ChunkByMut<'a, T, F, A>
where
    F: FnMut(&T, &T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            return None;
        }

        let len = self.slice.len();
        // Find the index of the last element that does not satisfy the predicate with the previous element.
        // We look for i such that !pred(slice[i], slice[i+1]).
        // The chunk includes slice[i+1], so the split point is i+1.

        // Use temporary view to find index
        let temp_view = SegmentedSlice {
            buf: self.slice.buf,
            start: self.slice.start,
            len: self.slice.len,
            end_ptr: self.slice.end_ptr,
            end_seg: self.slice.end_seg,
            _marker: PhantomData,
        };

        let split_point = temp_view
            .iter()
            .zip(temp_view.iter().skip(1))
            .rposition(|(prev, next)| !(self.pred)(prev, next))
            .map(|i| i + 1)
            .unwrap_or(0);

        // Manually split mutably
        let head_len = split_point;
        // let tail_len = len - split_point;

        // Construct tail (to return)
        let tail = SegmentedSliceMut::new(
            self.slice.buf,
            self.slice.start + head_len, // start of tail
            self.slice.start + len,      // end of tail (original end)
        );

        // Update self.slice to head
        self.slice = SegmentedSliceMut::new(
            self.slice.buf,
            self.slice.start,
            self.slice.start + head_len,
        );

        Some(tail)
    }
}

impl<'a, T, F, A: Allocator + 'a> FusedIterator for ChunkByMut<'a, T, F, A> where
    F: FnMut(&T, &T) -> bool
{
}
