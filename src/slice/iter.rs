//! Iterator definitions for `SegmentedSlice`.
//!
//! This module contains iterator types for `SegmentedSlice`, including:
//! - Split iterators (`Split`, `SplitMut`, `SplitInclusive`, etc.)
//! - Mutable chunk iterators (`ChunksMut`, `ChunksExactMut`, etc.)

use std::cmp;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

use crate::raw_vec::RawSegmentedVec;
use crate::slice::{SegmentedSlice, SegmentedSliceMut};

impl<'a, T> IntoIterator for SegmentedSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for SegmentedSliceMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = SliceIterMut<'a, T>;

    #[inline]
    fn into_iter(mut self) -> Self::IntoIter {
        SliceIterMut::new(&mut self)
    }
}

/// An iterator over the elements of a `SegmentedSlice`.
///
/// This iterator is optimized for sequential access by tracking pointers
/// directly instead of computing segment locations on each iteration.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SliceIter<'a, T> {
    segments: NonNull<*mut T>,
    /// Current pointer for forward iteration
    ptr: NonNull<T>,
    /// End of current segment for forward iteration
    seg_end: NonNull<T>,
    /// Current segment index for forward iteration
    idx: usize,
    /// Current pointer for backward iteration (exclusive, points to one past the next element to yield)
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
        f.debug_tuple("Iter").field(&self.as_slice()).finish()
    }
}

unsafe impl<T: Sync> Sync for SliceIter<'_, T> {}
unsafe impl<T: Sync> Send for SliceIter<'_, T> {}

impl<'a, T> SliceIter<'a, T> {
    /// Creates a new `SliceIter` from a buffer and index range.
    #[inline]
    pub(crate) fn new(slice: &SegmentedSlice<'a, T>) -> Self {
        let len = slice.len;

        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                segments: slice.segments,
                ptr: NonNull::dangling(),
                seg_end: NonNull::dangling(),
                idx: slice.start,
                back_ptr: NonNull::dangling(),
                back_seg_start: NonNull::dangling(),
                back_seg: 0,
                remaining: len,
                _marker: PhantomData,
            };
        }

        let segments = slice.segments;

        // Compute forward iteration state
        let (start_seg, start_offset) = RawSegmentedVec::<T>::location(slice.start);
        let start_ptr = unsafe {
            NonNull::new_unchecked((*segments.as_ptr().add(start_seg)).add(start_offset))
        };
        let start_seg_cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
        let start_seg_end = unsafe {
            NonNull::new_unchecked((*segments.as_ptr().add(start_seg)).add(start_seg_cap))
        };

        let end_seg_start = unsafe { *segments.as_ptr().add(slice.end_seg) };

        // Compute backward iteration state
        let (back_ptr, back_seg_start, back_seg) = (
            slice.end_ptr,
            unsafe { NonNull::new_unchecked(end_seg_start) },
            slice.end_seg,
        );

        Self {
            segments: slice.segments,
            ptr: start_ptr,
            seg_end: start_seg_end,
            idx: slice.start,
            back_ptr,
            back_seg_start,
            back_seg,
            remaining: len,
            _marker: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'a, T> {
        if self.remaining == 0 || std::mem::size_of::<T>() == 0 {
            return SegmentedSlice {
                segments: self.segments,
                start: self.idx,
                len: self.remaining,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let end_ptr = self.back_ptr;
        let end_seg = self.back_seg;

        SegmentedSlice {
            segments: self.segments,
            start: self.idx,
            len: self.remaining,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for SliceIter<'a, T> {
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

        if self.ptr == self.seg_end {
            // Use idx instead of seg field
            let (new_seg, _) = RawSegmentedVec::<T>::location(self.idx);

            // SAFETY: SegmentedSlice guarantees these segments exist.
            let seg_ptr = unsafe { *self.segments.as_ptr().add(new_seg) };
            let cap = RawSegmentedVec::<T>::segment_capacity(new_seg);

            // SAFETY: SegmentedSlice guarantees these segments exist.
            self.ptr = unsafe { NonNull::new_unchecked(seg_ptr) };
            // SAFETY: SegmentedSlice guarantees these segments exist.
            self.seg_end = unsafe { NonNull::new_unchecked(seg_ptr.add(cap)) };
        }

        // SAFETY: remaining > 0 means ptr is valid
        let result = unsafe { self.ptr.as_ref() };
        // SAFETY: ptr is valid
        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(1)) };
        self.idx += 1;
        self.remaining -= 1;

        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.idx += self.remaining;
            self.remaining = 0;
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            self.remaining -= n + 1;
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means ptr is valid
        let dist_from_end =
            unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) } as usize;
        if n < dist_from_end {
            let result = unsafe { self.ptr.as_ptr().add(n) };
            self.ptr = unsafe { NonNull::new_unchecked(result.add(1)) };
            self.idx += n + 1;
            self.remaining -= n + 1;
            return Some(unsafe { &*result });
        } else {
            let target_idx = self.idx + n;
            let (next_seg, offset) = RawSegmentedVec::<T>::location(target_idx);
            let next_seg_ptr = unsafe { *self.segments.as_ptr().add(next_seg) };
            let next_seg_cap = RawSegmentedVec::<T>::segment_capacity(next_seg);

            let result = unsafe { next_seg_ptr.add(offset) };
            self.ptr = unsafe { NonNull::new_unchecked(result.add(1)) };
            self.seg_end = unsafe { NonNull::new_unchecked(next_seg_ptr.add(next_seg_cap)) };
            self.idx = target_idx + 1;
            self.remaining -= n + 1;
            return Some(unsafe { &*result });
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if self.remaining == 0 {
            return init;
        }

        if std::mem::size_of::<T>() == 0 {
            let mut acc = init;
            for _ in 0..self.remaining {
                acc = f(acc, unsafe {
                    &*std::ptr::NonNull::<T>::dangling().as_ptr()
                });
            }
            return acc;
        }

        let mut acc = init;
        // SAFETY: remaining > 0 means ptr is valid
        let dist_from_end =
            unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) } as usize;
        let first_chunk_len = usize::min(dist_from_end, self.remaining);
        let first_chunk = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), first_chunk_len) };
        acc = first_chunk.iter().fold(acc, &mut f);

        self.remaining -= first_chunk_len;
        if self.remaining == 0 {
            return acc;
        }

        let next_idx = self.idx + first_chunk_len;
        let (mut current_seg_idx, _) = RawSegmentedVec::<T>::location(next_idx);

        while self.remaining > 0 {
            // SAFETY: remaining > 0 means segments is valid
            let seg_ptr = unsafe { *self.segments.as_ptr().add(current_seg_idx) };
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(current_seg_idx);
            let chunk_len = usize::min(seg_cap, self.remaining);

            // SAFETY: chunk_len <= seg_cap means seg_ptr is valid
            let seg_slice = unsafe { std::slice::from_raw_parts(seg_ptr, chunk_len) };
            acc = seg_slice.iter().fold(acc, &mut f);

            self.remaining -= chunk_len;
            current_seg_idx += 1;
        }

        acc
    }

    // TODO: implement try_fold when it's stabilized

    // fn is_sorted_by<F>(self, mut compare: F) -> bool
    // where
    //     Self: Sized,
    //     F: FnMut(&Self::Item, &Self::Item) -> bool,
    // {
    //     self.as_slice().is_sorted_by(|a, b| compare(&a, &b))
    // }
}

impl<T> DoubleEndedIterator for SliceIter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            // SAFETY: ZSTs don't need valid memory, dangling pointer is fine
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // Check if we need to retreat to the previous segment
        if self.back_ptr == self.back_seg_start {
            let back_seg = self.back_seg - 1;

            // Safety: Valid iterators always have valid preceding segments
            let seg_ptr = unsafe { *self.segments.as_ptr().add(back_seg) };
            let cap = RawSegmentedVec::<T>::segment_capacity(back_seg);

            self.back_ptr = unsafe { NonNull::new_unchecked(seg_ptr.add(cap)) };
            self.back_seg_start = unsafe { NonNull::new_unchecked(seg_ptr) };
            self.back_seg = back_seg;
        }

        // Decrement back_ptr to point to the element to yield
        // SAFETY: remaining > 0 means we can decrement and read
        self.back_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().sub(1)) };
        let result = unsafe { self.back_ptr.as_ref() };

        self.remaining -= 1;
        Some(result)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.remaining = 0;
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            self.remaining -= n + 1;
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means back_ptr is valid (and > back_seg_start if in current seg)
        // Distance is from back_ptr (exclusive end) to start.
        let dist_to_start = unsafe {
            self.back_ptr
                .as_ptr()
                .offset_from(self.back_seg_start.as_ptr())
        } as usize;

        if n < dist_to_start {
            // Fast path: target is in current segment
            // We want to skip n elements.
            // Current state: back_ptr is one past next element to yield.
            // If n=0, we call next_back, which decrs by 1 and yields.
            // nth_back(n) skips n elements and yields (n+1)-th.
            // Effectively we want to move back_ptr by (n+1).
            let res_ptr = unsafe { self.back_ptr.as_ptr().sub(n + 1) };
            self.back_ptr = unsafe { NonNull::new_unchecked(res_ptr) };
            self.remaining -= n + 1;
            return Some(unsafe { &*res_ptr });
        } else {
            // Slow path: crossing segment boundary
            let target_idx = self.idx + self.remaining - 1 - n;
            let (prev_seg, offset) = RawSegmentedVec::<T>::location(target_idx);
            let prev_seg_ptr = unsafe { *self.segments.as_ptr().add(prev_seg) };
            // offset is index of the element to yield.
            // back_ptr should become exclusive pointer to NEXT element after result?
            // Or just exclusive pointer to result?
            // Wait, next_back decrs then yields.
            // So back_ptr should be result_ptr + 1.
            let result_ptr = unsafe { prev_seg_ptr.add(offset) };
            self.back_ptr = unsafe { NonNull::new_unchecked(result_ptr) };
            self.back_seg_start = unsafe { NonNull::new_unchecked(prev_seg_ptr) };
            self.back_seg = prev_seg;
            self.remaining -= n + 1;
            return Some(unsafe { &*result_ptr });
        }
    }

    fn rfold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        if self.remaining == 0 {
            return init;
        }

        if std::mem::size_of::<T>() == 0 {
            let mut acc = init;
            for _ in 0..self.remaining {
                acc = f(acc, unsafe {
                    &*std::ptr::NonNull::<T>::dangling().as_ptr()
                });
            }
            return acc;
        }

        let mut acc = init;
        // SAFETY: remaining > 0 means ptr is valid
        let dist_to_start = unsafe {
            self.back_ptr
                .as_ptr()
                .offset_from(self.back_seg_start.as_ptr())
        } as usize;
        let first_chunk_len = usize::min(dist_to_start, self.remaining);
        let first_chunk_base = unsafe { self.back_ptr.as_ptr().sub(first_chunk_len) };
        let first_chunk = unsafe { std::slice::from_raw_parts(first_chunk_base, first_chunk_len) };
        acc = first_chunk.iter().rfold(acc, &mut f);

        self.remaining -= first_chunk_len;
        if self.remaining == 0 {
            return acc;
        }

        let mut current_seg_idx = self.back_seg;

        while self.remaining > 0 {
            // SAFETY: remaining > 0 means segments is valid
            let seg_ptr = unsafe { *self.segments.as_ptr().add(current_seg_idx) };
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(current_seg_idx);
            let chunk_len = usize::min(seg_cap, self.remaining);

            // SAFETY: chunk_len <= seg_cap means seg_ptr is valid
            let seg_slice = unsafe { std::slice::from_raw_parts(seg_ptr, chunk_len) };
            acc = seg_slice.iter().rfold(acc, &mut f);

            self.remaining -= chunk_len;
            current_seg_idx = current_seg_idx.wrapping_sub(1);
        }

        acc
    }
}

impl<T> ExactSizeIterator for SliceIter<'_, T> {}
impl<T> std::iter::FusedIterator for SliceIter<'_, T> {}

/// A mutable iterator over the elements of a `SegmentedSlice`.
///
/// This iterator is optimized for sequential access by tracking pointers
/// directly instead of computing segment locations on each iteration.
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SliceIterMut<'a, T> {
    segments: NonNull<*mut T>,
    /// Current pointer for forward iteration
    ptr: NonNull<T>,
    /// End of current segment for forward iteration
    seg_end: NonNull<T>,
    /// Current segment index for forward iteration
    idx: usize,
    /// Current pointer for backward iteration (exclusive end)
    back_ptr: NonNull<T>,
    /// Start of current segment for backward iteration
    back_seg_start: NonNull<T>,
    /// Current segment index for backward iteration
    back_seg: usize,
    /// Remaining element count (used for size_hint and termination)
    remaining: usize,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<T: Send> Send for SliceIterMut<'_, T> {}
unsafe impl<T: Sync> Sync for SliceIterMut<'_, T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for SliceIterMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IterMut").field(&self.as_slice()).finish()
    }
}

impl<'a, T> SliceIterMut<'a, T> {
    /// Creates a new `SliceIterMut` from a buffer and index range.
    #[inline]
    pub(crate) fn new(slice: &mut SegmentedSliceMut<'_, T>) -> Self {
        let len = slice.len();

        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                segments: slice.segments,
                ptr: NonNull::dangling(),
                seg_end: NonNull::dangling(),
                idx: slice.start,
                back_ptr: NonNull::dangling(),
                back_seg_start: NonNull::dangling(),
                back_seg: 0,
                remaining: len,
                _marker: PhantomData,
            };
        }

        let segments = slice.segments;

        // Compute forward iteration state
        let (start_seg, start_offset) = RawSegmentedVec::<T>::location(slice.start);
        let start_ptr = unsafe {
            NonNull::new_unchecked((*segments.as_ptr().add(start_seg)).add(start_offset))
        };
        let start_seg_cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
        let start_seg_end = unsafe {
            NonNull::new_unchecked((*segments.as_ptr().add(start_seg)).add(start_seg_cap))
        };

        let end_seg_start = unsafe { *segments.as_ptr().add(slice.end_seg) };

        // Compute backward iteration state
        let (back_ptr, back_seg_start, back_seg) = (
            slice.end_ptr,
            unsafe { NonNull::new_unchecked(end_seg_start) },
            slice.end_seg,
        );

        Self {
            segments: slice.segments,
            ptr: start_ptr,
            seg_end: start_seg_end,
            idx: slice.start,
            back_ptr,
            back_seg_start,
            back_seg,
            remaining: len,
            _marker: PhantomData,
        }
    }

    #[must_use = "`self` will be dropped if the result is not used"]
    pub fn into_slice(self) -> SegmentedSliceMut<'a, T> {
        if self.remaining == 0 || std::mem::size_of::<T>() == 0 {
            return SegmentedSliceMut {
                segments: self.segments,
                start: self.idx,
                len: self.remaining,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let end_ptr = self.back_ptr;
        let end_seg = self.back_seg;

        SegmentedSliceMut {
            segments: self.segments,
            start: self.idx,
            len: self.remaining,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'a, T> {
        if self.remaining == 0 || std::mem::size_of::<T>() == 0 {
            return SegmentedSlice {
                segments: self.segments,
                start: self.idx,
                len: self.remaining,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }

        let end_ptr = self.back_ptr;
        let end_seg = self.back_seg;

        SegmentedSlice {
            segments: self.segments,
            start: self.idx,
            len: self.remaining,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for SliceIterMut<'a, T> {
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

        if self.ptr == self.seg_end {
            // Use idx instead of seg field
            let (new_seg, _) = RawSegmentedVec::<T>::location(self.idx);

            // SAFETY: SegmentedSlice guarantees these segments exist.
            let seg_ptr = unsafe { *self.segments.as_ptr().add(new_seg) };
            let cap = RawSegmentedVec::<T>::segment_capacity(new_seg);

            // SAFETY: SegmentedSlice guarantees these segments exist.
            self.ptr = unsafe { NonNull::new_unchecked(seg_ptr) };
            // SAFETY: SegmentedSlice guarantees these segments exist.
            self.seg_end = unsafe { NonNull::new_unchecked(seg_ptr.add(cap)) };
        }

        // SAFETY: remaining > 0 means ptr is valid
        let result = unsafe { &mut *self.ptr.as_ptr() };
        // SAFETY: ptr is valid
        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(1)) };
        self.idx += 1;
        self.remaining -= 1;

        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.idx += self.remaining;
            self.remaining = 0;
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            self.remaining -= n + 1;
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means ptr is valid
        let dist_from_end =
            unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) } as usize;
        if n < dist_from_end {
            let result = unsafe { self.ptr.as_ptr().add(n) };
            self.ptr = unsafe { NonNull::new_unchecked(result.add(1)) };
            self.idx += n + 1;
            self.remaining -= n + 1;
            return Some(unsafe { &mut *result });
        } else {
            let target_idx = self.idx + n;
            let (next_seg, offset) = RawSegmentedVec::<T>::location(target_idx);
            let next_seg_ptr = unsafe { *self.segments.as_ptr().add(next_seg) };
            let next_seg_cap = RawSegmentedVec::<T>::segment_capacity(next_seg);

            let result = unsafe { next_seg_ptr.add(offset) };
            self.ptr = unsafe { NonNull::new_unchecked(result.add(1)) };
            self.seg_end = unsafe { NonNull::new_unchecked(next_seg_ptr.add(next_seg_cap)) };
            self.idx = target_idx + 1;
            self.remaining -= n + 1;
            return Some(unsafe { &mut *result });
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        if self.remaining == 0 {
            return init;
        }

        if std::mem::size_of::<T>() == 0 {
            let mut acc = init;
            for _ in 0..self.remaining {
                acc = f(acc, unsafe {
                    &mut *std::ptr::NonNull::<T>::dangling().as_ptr()
                });
            }
            return acc;
        }

        let mut acc = init;
        // SAFETY: remaining > 0 means ptr is valid
        let dist_from_end =
            unsafe { self.seg_end.as_ptr().offset_from(self.ptr.as_ptr()) } as usize;
        let first_chunk_len = usize::min(dist_from_end, self.remaining);
        let first_chunk =
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), first_chunk_len) };
        acc = first_chunk.iter_mut().fold(acc, &mut f);

        self.remaining -= first_chunk_len;
        if self.remaining == 0 {
            return acc;
        }

        let next_idx = self.idx + first_chunk_len;
        let (mut current_seg_idx, _) = RawSegmentedVec::<T>::location(next_idx);

        while self.remaining > 0 {
            // SAFETY: remaining > 0 means segments is valid
            let seg_ptr = unsafe { *self.segments.as_ptr().add(current_seg_idx) };
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(current_seg_idx);
            let chunk_len = usize::min(seg_cap, self.remaining);

            // SAFETY: chunk_len <= seg_cap means seg_ptr is valid
            let seg_slice = unsafe { std::slice::from_raw_parts_mut(seg_ptr, chunk_len) };
            acc = seg_slice.iter_mut().fold(acc, &mut f);

            self.remaining -= chunk_len;
            current_seg_idx += 1;
        }

        acc
    }

    // TODO: implement try_fold when it's stabilized
}

impl<T> DoubleEndedIterator for SliceIterMut<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // ZST fast path
        if std::mem::size_of::<T>() == 0 {
            self.remaining -= 1;
            // SAFETY: ZSTs don't need valid memory, dangling pointer is fine
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // Check if we need to retreat to the previous segment
        if self.back_ptr == self.back_seg_start {
            let back_seg = self.back_seg - 1;

            // Safety: Valid iterators always have valid preceding segments
            let seg_ptr = unsafe { *self.segments.as_ptr().add(back_seg) };
            let cap = RawSegmentedVec::<T>::segment_capacity(back_seg);

            self.back_ptr = unsafe { NonNull::new_unchecked(seg_ptr.add(cap)) };
            self.back_seg_start = unsafe { NonNull::new_unchecked(seg_ptr) };
            self.back_seg = back_seg;
        }

        // SAFETY: back_ptr is not at start (due to possible retreat_segment)
        self.back_ptr = unsafe { NonNull::new_unchecked(self.back_ptr.as_ptr().sub(1)) };

        // SAFETY: back_ptr is valid (it points to the element we just stepped over)
        let result = unsafe { &mut *self.back_ptr.as_ptr() };

        self.remaining -= 1;
        Some(result)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.remaining {
            self.remaining = 0;
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            self.remaining -= n + 1;
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        // SAFETY: remaining > 0 means back_ptr is valid (and > back_seg_start if in current seg)
        // Distance is from back_ptr (exclusive end) to start.
        let dist_to_start = unsafe {
            self.back_ptr
                .as_ptr()
                .offset_from(self.back_seg_start.as_ptr())
        } as usize;

        if n < dist_to_start {
            // Fast path: target is in current segment
            // We want to skip n elements.
            // Current state: back_ptr is one past next element to yield.
            // If n=0, we call next_back, which decrs by 1 and yields.
            // nth_back(n) skips n elements and yields (n+1)-th.
            // Effectively we want to move back_ptr by (n+1).
            let res_ptr = unsafe { self.back_ptr.as_ptr().sub(n + 1) };
            self.back_ptr = unsafe { NonNull::new_unchecked(res_ptr) };
            self.remaining -= n + 1;
            return Some(unsafe { &mut *res_ptr });
        } else {
            // Slow path: crossing segment boundary
            let target_idx = self.idx + self.remaining - 1 - n;
            let (prev_seg, offset) = RawSegmentedVec::<T>::location(target_idx);
            let prev_seg_ptr = unsafe { *self.segments.as_ptr().add(prev_seg) };
            // offset is index of the element to yield.
            // back_ptr should become exclusive pointer to NEXT element after result?
            // Or just exclusive pointer to result?
            // Wait, next_back decrs then yields.
            // So back_ptr should be result_ptr + 1.
            let result_ptr = unsafe { prev_seg_ptr.add(offset) };
            self.back_ptr = unsafe { NonNull::new_unchecked(result_ptr) };
            self.back_seg_start = unsafe { NonNull::new_unchecked(prev_seg_ptr) };
            self.back_seg = prev_seg;
            self.remaining -= n + 1;
            return Some(unsafe { &mut *result_ptr });
        }
    }

    fn rfold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        if self.remaining == 0 {
            return init;
        }

        if std::mem::size_of::<T>() == 0 {
            let mut acc = init;
            for _ in 0..self.remaining {
                acc = f(acc, unsafe {
                    &mut *std::ptr::NonNull::<T>::dangling().as_ptr()
                });
            }
            return acc;
        }

        let mut acc = init;
        // SAFETY: remaining > 0 means ptr is valid
        let dist_to_start = unsafe {
            self.back_ptr
                .as_ptr()
                .offset_from(self.back_seg_start.as_ptr())
        } as usize;
        let first_chunk_len = usize::min(dist_to_start, self.remaining);
        let first_chunk_base = unsafe { self.back_ptr.as_ptr().sub(first_chunk_len) };
        let first_chunk =
            unsafe { std::slice::from_raw_parts_mut(first_chunk_base, first_chunk_len) };
        acc = first_chunk.iter_mut().rfold(acc, &mut f);

        self.remaining -= first_chunk_len;
        if self.remaining == 0 {
            return acc;
        }

        let mut current_seg_idx = self.back_seg;

        while self.remaining > 0 {
            // SAFETY: remaining > 0 means segments is valid
            let seg_ptr = unsafe { *self.segments.as_ptr().add(current_seg_idx) };
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(current_seg_idx);
            let chunk_len = usize::min(seg_cap, self.remaining);

            // SAFETY: chunk_len <= seg_cap means seg_ptr is valid
            let seg_slice = unsafe { std::slice::from_raw_parts_mut(seg_ptr, chunk_len) };
            acc = seg_slice.iter_mut().rfold(acc, &mut f);

            self.remaining -= chunk_len;
            current_seg_idx = current_seg_idx.wrapping_sub(1);
        }

        acc
    }
}

impl<T> ExactSizeIterator for SliceIterMut<'_, T> {}
impl<T> std::iter::FusedIterator for SliceIterMut<'_, T> {}
impl<T> Default for SliceIterMut<'_, T> {
    fn default() -> Self {
        Self {
            segments: NonNull::dangling(),
            ptr: NonNull::dangling(),
            seg_end: NonNull::dangling(),
            idx: 0,
            back_ptr: NonNull::dangling(),
            back_seg_start: NonNull::dangling(),
            back_seg: 0,
            remaining: 0,
            _marker: std::marker::PhantomData,
        }
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
pub struct Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    pub(crate) slice: SegmentedSlice<'a, T>,
    pred: P,
    finished: bool,
}

impl<'a, T, P: FnMut(&T) -> bool> Split<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, pred: P) -> Self {
        Self {
            slice,
            pred,
            finished: false,
        }
    }

    /// Returns the remainder of the original slice that has not yet been yielded.
    #[inline]
    pub fn as_slice(&self) -> &SegmentedSlice<'a, T> {
        &self.slice
    }
}

impl<T: std::fmt::Debug, P> std::fmt::Debug for Split<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Split")
            .field("slice", &self.slice)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> Clone for Split<'a, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        Self {
            slice: self.slice.clone(),
            pred: self.pred.clone(),
            finished: self.finished,
        }
    }
}

impl<'a, T, P> Iterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.slice.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let (left, right) =
                    // SAFETY: if v.iter().position returns Some(idx), that
                    // idx is definitely a valid index for v
                    unsafe { (self.slice.get_unchecked(..idx), self.slice.get_unchecked(idx + 1..)) };
                let ret = Some(left);
                self.slice = right;
                ret
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

impl<'a, T, P> DoubleEndedIterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.slice.iter().rposition(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let (left, right) =
                    // SAFETY: if v.iter().rposition returns Some(idx), then
                    // idx is definitely a valid index for v
                    unsafe { (self.slice.get_unchecked(..idx), self.slice.get_unchecked(idx + 1..)) };
                let ret = Some(right);
                self.slice = left;
                ret
            }
        }
    }
}

impl<'a, T, P> SplitIter for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSlice<'a, T>> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(self.slice)
        }
    }
}

impl<'a, T, P> FusedIterator for Split<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function. Unlike `Split`, it contains the matched part as a terminator
/// of the subslice.
///
/// This struct is created by the [`split_inclusive`] method on [`SegmentedSlice`].
///
/// [`split_inclusive`]: SegmentedSlice::split_inclusive
pub struct SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSlice<'a, T>,
    pred: P,
    finished: bool,
}

impl<'a, T, P: FnMut(&T) -> bool> SplitInclusive<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<T: std::fmt::Debug, P> std::fmt::Debug for SplitInclusive<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitInclusive")
            .field("slice", &self.slice)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> Clone for SplitInclusive<'a, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        Self {
            slice: self.slice.clone(),
            pred: self.pred.clone(),
            finished: self.finished,
        }
    }
}

impl<'a, T, P> Iterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let idx = self
            .slice
            .iter()
            .position(|x| (self.pred)(x))
            .map(|idx| idx + 1)
            .unwrap_or(self.slice.len());
        if idx == self.slice.len() {
            self.finished = true;
        }
        let ret = Some(self.slice.range(..idx));
        self.slice = self.slice.range(idx..);
        ret
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len()` one-element slices,
            // or a single empty slice.
            (1, Some(cmp::max(1, self.slice.len())))
        }
    }
}

impl<'a, T, P> DoubleEndedIterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // The last index of self.slice is already checked and found to match
        // by the last iteration, so we start searching a new match
        // one index to the left.
        let remainder = if self.slice.is_empty() {
            self.slice
        } else {
            self.slice.range(..(self.slice.len() - 1))
        };
        let idx = remainder
            .iter()
            .rposition(|x| (self.pred)(x))
            .map(|idx| idx + 1)
            .unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let ret = Some(self.slice.range(idx..));
        self.slice = self.slice.range(..idx);
        ret
    }
}

impl<'a, T, P> FusedIterator for SplitInclusive<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate function.
pub struct SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSliceMut<'a, T>,
    pred: P,
    finished: bool,
}

impl<'a, T, P: FnMut(&T) -> bool> SplitMut<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<T: std::fmt::Debug, P> std::fmt::Debug for SplitMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitMut")
            .field("slice", &self.slice)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> SplitIter for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSliceMut<'a, T>> {
        if self.finished {
            None
        } else {
            self.finished = true;
            // Capture fields before borrowing slice mutably
            let slice_buf = self.slice.segments;
            let slice_start = self.slice.start;

            Some(std::mem::replace(
                &mut self.slice,
                SegmentedSliceMut {
                    segments: slice_buf,
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

impl<'a, T, P> Iterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.slice.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let mut tmp = std::mem::take(&mut self.slice);
                // idx is the index of the element we are splitting on. We want to set self to the
                // region after idx, and return the subslice before and not including idx.
                // So first we split after idx
                let (head, tail) = tmp.split_at_mut(idx + 1);
                self.slice = tail;
                // Then return the subslice up to but not including the found element
                Some(head.range(..idx))
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

impl<'a, T, P> DoubleEndedIterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.slice.iter().rposition(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let mut tmp = std::mem::take(&mut self.slice);
                let (head, tail) = tmp.split_at_mut(idx);
                self.slice = head;
                Some(tail.range(1..))
            }
        }
    }
}

impl<'a, T, P> FusedIterator for SplitMut<'a, T, P> where P: FnMut(&T) -> bool {}

pub struct SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    slice: SegmentedSliceMut<'a, T>,
    pred: P,
    finished: bool,
}

impl<'a, T, P: FnMut(&T) -> bool> SplitInclusiveMut<'a, T, P> {
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, pred: P) -> Self {
        let finished = slice.is_empty();
        Self {
            slice,
            pred,
            finished,
        }
    }
}

impl<T: std::fmt::Debug, P> std::fmt::Debug for SplitInclusiveMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitInclusiveMut")
            .field("slice", &self.slice)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, T, P> Iterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.slice.iter().position(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(self.slice.len());
        if idx == self.slice.len() {
            self.finished = true;
        }
        let mut tmp = std::mem::take(&mut self.slice);
        let (head, tail) = tmp.split_at_mut(idx);
        self.slice = tail;
        Some(head.range(..idx))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len()` one-element slices,
            // or a single empty slice.
            (1, Some(cmp::max(1, self.slice.len())))
        }
    }
}

impl<'a, T, P> DoubleEndedIterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let idx_opt = if self.slice.is_empty() {
            None
        } else {
            // work around borrowck limitations
            let pred = &mut self.pred;

            // The last index of self.v is already checked and found to match
            // by the last iteration, so we start searching a new match
            // one index to the left.
            let remainder = self.slice.as_slice().range(..(self.slice.len() - 1));
            remainder.iter().rposition(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let mut tmp = std::mem::take(&mut self.slice);
        let (head, tail) = tmp.split_at_mut(idx);
        self.slice = head;
        Some(tail.range(1..))
    }
}

impl<'a, T, P> FusedIterator for SplitInclusiveMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [`SegmentedSlice`].
///
/// [`rsplit`]: SegmentedSlice::rsplit
#[derive(Debug, Clone)]
pub struct RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: Split<'a, T, P>,
}

impl<'a, T, P: FnMut(&T) -> bool> RSplit<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, pred: P) -> Self {
        Self {
            inner: Split::new(slice, pred),
        }
    }
}

impl<'a, T, P> Iterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> DoubleEndedIterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T, P> SplitIter for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSlice<'a, T>> {
        self.inner.finish()
    }
}

impl<'a, T, P> FusedIterator for RSplit<'a, T, P> where P: FnMut(&T) -> bool {}

#[derive(Debug)]
pub struct RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: SplitMut<'a, T, P>,
}

impl<'a, T, P> SplitIter for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<SegmentedSliceMut<'a, T>> {
        self.inner.finish()
    }
}

impl<'a, T, P: FnMut(&T) -> bool> RSplitMut<'a, T, P> {
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, pred: P) -> Self {
        Self {
            inner: SplitMut::new(slice, pred),
        }
    }
}

impl<'a, T, P> Iterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> DoubleEndedIterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, T, P> FusedIterator for RSplitMut<'a, T, P> where P: FnMut(&T) -> bool {}

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
pub struct SplitN<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<Split<'a, T, P>>,
}

impl<'a, T, P: FnMut(&T) -> bool> SplitN<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: Split::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, P> Iterator for SplitN<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> FusedIterator for SplitN<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits, starting from the end.
///
/// This struct is created by the [`rsplitn`] method on [`SegmentedSlice`].
///
/// [`rsplitn`]: SegmentedSlice::rsplitn
pub struct RSplitN<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<RSplit<'a, T, P>>,
}

impl<'a, T, P: FnMut(&T) -> bool> RSplitN<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: RSplit::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, P> Iterator for RSplitN<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> FusedIterator for RSplitN<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [`SegmentedSliceMut`].
///
/// [`splitn_mut`]: SegmentedSliceMut::splitn_mut
pub struct SplitNMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<SplitMut<'a, T, P>>,
}

impl<'a, T, P: FnMut(&T) -> bool> SplitNMut<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: SplitMut::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, P> Iterator for SplitNMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> FusedIterator for SplitNMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits, starting from the end.
///
/// This struct is created by the [`rsplitn_mut`] method on [`SegmentedSliceMut`].
///
/// [`rsplitn_mut`]: SegmentedSliceMut::rsplitn_mut
pub struct RSplitNMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<RSplitMut<'a, T, P>>,
}

impl<'a, T, P: FnMut(&T) -> bool> RSplitNMut<'a, T, P> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, n: usize, pred: P) -> Self {
        Self {
            inner: GenericSplitN {
                iter: RSplitMut::new(slice, pred),
                count: n,
            },
        }
    }
}

impl<'a, T, P> Iterator for RSplitNMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, P> FusedIterator for RSplitNMut<'a, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over overlapping windows of a `SegmentedSlice`.
#[derive(Clone, Debug)]
pub struct Windows<'a, T> {
    pub(crate) slice: SegmentedSlice<'a, T>,
    pub(crate) size: NonZeroUsize,
    // Cached end location for the current window.
    pub(crate) window_end_ptr: NonNull<T>,
    // Current segment index for window end pointer
    pub(crate) window_end_seg: usize,
    // End of current segment for window end pointer
    pub(crate) window_end_seg_end: NonNull<T>,
}

impl<'a, T: 'a> Windows<'a, T> {
    #[inline]
    pub(super) fn new(slice: SegmentedSlice<'a, T>, size: NonZeroUsize) -> Self {
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
        let (window_end_seg, window_end_offset) = RawSegmentedVec::<T>::location_for_end(end_index);
        let buf = slice.segments;

        let window_end_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(window_end_seg)).add(window_end_offset);
            NonNull::new_unchecked(ptr)
        };

        let window_end_seg_end = unsafe {
            let cap = RawSegmentedVec::<T>::segment_capacity(window_end_seg);
            NonNull::new_unchecked((*buf.as_ptr().add(window_end_seg)).add(cap))
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

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let size = self.size.get();
        if self.slice.len() < size {
            return None;
        }

        // Construct the window slice using cached end pointers.
        // We use struct initialization directly to bypass `SegmentedSlice::new` which recalculates.
        let window = SegmentedSlice {
            segments: self.slice.segments,
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
        // Logic: if we have more windows, we must advance window_end.
        // We only move to the next segment if we actually have elements there.
        if self.slice.len >= size {
            let buf = self.slice.segments;

            // Non-ZST logic
            if core::mem::size_of::<T>() > 0 {
                // Check if we reached the boundary of the current segment
                if self.window_end_ptr == self.window_end_seg_end {
                    // Move to next segment
                    self.window_end_seg += 1;
                    // Since len >= size, we know the next segment MUST be allocated
                    let ptr = unsafe { *buf.as_ptr().add(self.window_end_seg) };
                    // Set ptr to start of next segment
                    self.window_end_ptr = unsafe { NonNull::new_unchecked(ptr) };
                    let cap = RawSegmentedVec::<T>::segment_capacity(self.window_end_seg);
                    self.window_end_seg_end = unsafe { NonNull::new_unchecked(ptr.add(cap)) };
                }

                // Now advance pointer by 1
                self.window_end_ptr =
                    unsafe { NonNull::new_unchecked(self.window_end_ptr.as_ptr().add(1)) };
            } else {
                let end_index = self.slice.start + size;
                let (seg, _) = RawSegmentedVec::<T>::location(end_index);
                self.window_end_seg = seg;
            }
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
        let (mut seg, mut offset) = RawSegmentedVec::<T>::location(end_index);

        let buf = self.slice.segments;
        // If offset is 0, we're at segment boundary, backtrack to previous segment
        if offset == 0 && seg > 0 {
            seg -= 1;
            offset = RawSegmentedVec::<T>::segment_capacity(seg);
        }

        self.window_end_seg = seg;

        let ptr = unsafe { (*buf.as_ptr().add(seg)).add(offset) };
        self.window_end_ptr = unsafe { NonNull::new_unchecked(ptr) };

        self.window_end_seg_end = unsafe {
            let cap = RawSegmentedVec::<T>::segment_capacity(seg);
            NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap))
        };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }
}

impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
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

        let window = SegmentedSlice::new(self.slice.segments, start_index, end_index);

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

impl<T> ExactSizeIterator for Windows<'_, T> {}
impl<T> FusedIterator for Windows<'_, T> {}

/// An iterator over chunks of a `SegmentedSlice`.
#[derive(Debug, Clone)]
pub struct Chunks<'a, T> {
    pub(crate) slice: SegmentedSlice<'a, T>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T> Chunks<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, chunk_size: usize) -> Self {
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

        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        let buf = slice.segments;

        // If offset is 0, we're at segment boundary, backtrack to previous segment
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }

        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };

        let start_seg_end = unsafe {
            let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
            NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
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

impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = SegmentedSlice<'a, T>;

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
                    // Actually, let's just use self.slice.segments.ptr_at(chunk_end_idx) which is optimized.
                    let buf = self.slice.segments;
                    let (seg, off) = RawSegmentedVec::<T>::location_for_end(chunk_end_idx);
                    let ptr = (*buf.as_ptr().add(seg)).add(off);
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let chunk = SegmentedSlice {
                segments: self.slice.segments,
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
                        let cap = RawSegmentedVec::<T>::segment_capacity(chunk_end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            (*self.slice.segments.as_ptr().add(chunk_end_seg)).add(cap),
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
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
                segments: self.slice.segments,
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
                    let buf = self.slice.segments;
                    let seg_ptr = *buf.as_ptr().add(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location(chunk_start);
                        let (seg, ptr) = if off == 0 && seg > 0 {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T>::segment_capacity(last_seg);
                            (last_seg, (*buf.as_ptr().add(last_seg)).add(cap))
                        } else {
                            (seg, (*buf.as_ptr().add(seg)).add(off))
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(chunk_end_abs);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next_back()
    }
}

impl<T> ExactSizeIterator for Chunks<'_, T> {}
impl<T> std::iter::FusedIterator for Chunks<'_, T> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks.
///
/// This struct is created by the [`chunks_mut`] method on [`SegmentedSlice`].
///
/// [`chunks_mut`]: SegmentedSlice::chunks_mut
pub struct ChunksMut<'a, T> {
    pub(crate) slice: SegmentedSliceMut<'a, T>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T> ChunksMut<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, chunk_size: usize) -> Self {
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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
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

impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = SegmentedSliceMut<'a, T>;

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
                    let buf = self.slice.segments;
                    let (seg, off) = RawSegmentedVec::<T>::location_for_end(chunk_end_idx);
                    let ptr = (*buf.as_ptr().add(seg)).add(off);
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSliceMut {
                segments: self.slice.segments,
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
                        let cap = RawSegmentedVec::<T>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            (*self.slice.segments.as_ptr().add(end_seg)).add(cap),
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };

        self.next()
    }
}

impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
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
                segments: self.slice.segments,
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
                    let buf = self.slice.segments;
                    let seg_ptr = *buf.as_ptr().add(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location(chunk_start);
                        let (seg, ptr) = if off == 0 && seg > 0 {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T>::segment_capacity(last_seg);
                            (last_seg, (*buf.as_ptr().add(last_seg)).add(cap))
                        } else {
                            (seg, (*buf.as_ptr().add(seg)).add(off))
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start + self.slice.len);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next_back()
    }
}

impl<'a, T> ExactSizeIterator for ChunksMut<'a, T> {}
impl<'a, T> FusedIterator for ChunksMut<'a, T> {}
unsafe impl<T> Send for ChunksMut<'_, T> where T: Send {}
unsafe impl<T> Sync for ChunksMut<'_, T> where T: Sync {}

#[derive(Debug, Clone)]
pub struct ChunksExact<'a, T> {
    pub(crate) slice: SegmentedSlice<'a, T>,
    pub(crate) rem: SegmentedSlice<'a, T>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T> ChunksExact<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem_len = len % chunk_size;
        let exact_len = len - rem_len;

        let rem = slice.range(exact_len..len);
        let exact_slice = slice.range(0..exact_len);

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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
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
    pub fn remainder(&self) -> SegmentedSlice<'a, T> {
        self.rem
    }
}

impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = SegmentedSlice<'a, T>;

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
                    let buf = self.slice.segments;
                    let (seg, off) = RawSegmentedVec::<T>::location_for_end(chunk_end_idx);
                    let ptr = (*buf.as_ptr().add(seg)).add(off);
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSlice {
                segments: self.slice.segments,
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
                        let cap = RawSegmentedVec::<T>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            (*self.slice.segments.as_ptr().add(end_seg)).add(cap),
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let result = SegmentedSlice {
                segments: self.slice.segments,
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
                    let buf = self.slice.segments;
                    let seg_ptr = *buf.as_ptr().add(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location(chunk_start);
                        let (seg, ptr) = if off == 0 && seg > 0 {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T>::segment_capacity(last_seg);
                            (last_seg, (*buf.as_ptr().add(last_seg)).add(cap))
                        } else {
                            (seg, (*buf.as_ptr().add(seg)).add(off))
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start + self.slice.len);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next_back()
    }
}

impl<T> ExactSizeIterator for ChunksExact<'_, T> {}
impl<T> std::iter::FusedIterator for ChunksExact<'_, T> {}

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
pub struct ChunksExactMut<'a, T> {
    pub(crate) slice: SegmentedSliceMut<'a, T>,
    pub(crate) rem: SegmentedSliceMut<'a, T>,
    pub(crate) chunk_size: usize,
    /// Cached pointer to the start of the current chunk.
    pub(crate) start_ptr: NonNull<T>,
    /// Current segment index for start pointer.
    pub(crate) start_seg: usize,
    /// End of current segment for start pointer.
    pub(crate) start_seg_end: NonNull<T>,
}

impl<'a, T> ChunksExactMut<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem_len = len % chunk_size;
        let exact_len = len - rem_len;

        let rem =
            SegmentedSliceMut::new(slice.segments, slice.start + exact_len, slice.start + len);
        let exact_slice =
            SegmentedSliceMut::new(slice.segments, slice.start, slice.start + exact_len);

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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
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
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T> {
        self.rem
    }
}

impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = SegmentedSliceMut<'a, T>;

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
                    let buf = self.slice.segments;
                    let (seg, off) = RawSegmentedVec::<T>::location_for_end(chunk_end_idx);
                    let ptr = (*buf.as_ptr().add(seg)).add(off);
                    (NonNull::new_unchecked(ptr), seg)
                }
            };

            let result = SegmentedSliceMut {
                segments: self.slice.segments,
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
                        let cap = RawSegmentedVec::<T>::segment_capacity(end_seg);
                        self.start_seg_end = NonNull::new_unchecked(
                            (*self.slice.segments.as_ptr().add(end_seg)).add(cap),
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.start_seg = seg;
        self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
        self.start_seg_end = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for ChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len == 0 {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.slice.start + self.slice.len - chunk_len;

            // The chunk ends at self.slice.end_ptr/end_seg.
            let result = SegmentedSliceMut {
                segments: self.slice.segments,
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
                    let buf = self.slice.segments;
                    let seg_ptr = *buf.as_ptr().add(self.slice.end_seg);
                    let offset = self.slice.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.slice.end_ptr =
                            NonNull::new_unchecked(self.slice.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location(chunk_start);
                        let (seg, ptr) = if off == 0 && seg > 0 {
                            let last_seg = seg - 1;
                            let cap = RawSegmentedVec::<T>::segment_capacity(last_seg);
                            (last_seg, (*buf.as_ptr().add(last_seg)).add(cap))
                        } else {
                            (seg, (*buf.as_ptr().add(seg)).add(off))
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.slice.start + self.slice.len);
        let buf = self.slice.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.slice.end_seg = seg;
        self.slice.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next_back()
    }
}

impl<T> ExactSizeIterator for ChunksExactMut<'_, T> {}
impl<T> std::iter::FusedIterator for ChunksExactMut<'_, T> {}

/// An iterator over chunks of a `SegmentedSlice`, starting from the end.
#[derive(Debug, Clone)]
pub struct RChunks<'a, T> {
    pub(crate) segments: NonNull<*mut T>,
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

impl<'a, T> RChunks<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                segments: slice.segments,
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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            segments: slice.segments,
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

impl<'a, T> Iterator for RChunks<'a, T> {
    type Item = SegmentedSlice<'a, T>;

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
                segments: self.segments,
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
                    let buf = self.segments;
                    let seg_ptr = *buf.as_ptr().add(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location_for_end(self.end);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.end);
        let buf = self.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for RChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let total_len = self.end - self.start;
            let rem = total_len % self.chunk_size;
            let chunk_len = if rem == 0 { self.chunk_size } else { rem };

            let chunk_start = self.start;
            let (end_seg, end_offset) = RawSegmentedVec::<T>::location(chunk_start + chunk_len);
            let buf = self.segments;
            let end_ptr = if end_offset == 0 && end_seg > 0 {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T>::location(chunk_start + chunk_len - 1);
                unsafe {
                    NonNull::new_unchecked((*buf.as_ptr().add(last_seg)).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(end_seg)).add(end_offset)) }
            };

            let chunk = SegmentedSlice {
                segments: self.segments,
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
                    let _buf = *self.segments.as_ptr();
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
                        let (seg, off) = RawSegmentedVec::<T>::location(self.start);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
                        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
                        let seg_end = (*buf.as_ptr().add(seg)).add(cap);

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
            let (seg, off) = RawSegmentedVec::<T>::location(self.start);
            let buf = self.segments;
            let cap = RawSegmentedVec::<T>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
            self.start_seg_end =
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };
        }

        self.next_back()
    }
}

impl<T> ExactSizeIterator for RChunks<'_, T> {}
impl<T> std::iter::FusedIterator for RChunks<'_, T> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks,
/// starting at the end of the slice.
///
/// This struct is created by the [`rchunks_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_mut`]: SegmentedSlice::rchunks_mut
#[derive(Debug)]
pub struct RChunksMut<'a, T> {
    segments: NonNull<*mut T>,
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

impl<'a, T> RChunksMut<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");

        if slice.len == 0 {
            return Self {
                segments: slice.segments,
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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(slice.start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            segments: slice.segments,
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

impl<'a, T> Iterator for RChunksMut<'a, T> {
    type Item = SegmentedSliceMut<'a, T>;

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
                segments: self.segments,
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
                    let buf = self.segments;
                    let seg_ptr = *buf.as_ptr().add(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location_for_end(self.end);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.end);
        let buf = self.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a, T> DoubleEndedIterator for RChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let total_len = self.end - self.start;
            let rem = total_len % self.chunk_size;
            let chunk_len = if rem == 0 { self.chunk_size } else { rem };

            let chunk_start = self.start;
            let (end_seg, end_offset) = RawSegmentedVec::<T>::location(chunk_start + chunk_len);
            let buf = self.segments;
            let end_ptr = if end_offset == 0 && end_seg > 0 {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T>::location(chunk_start + chunk_len - 1);
                unsafe {
                    NonNull::new_unchecked((*buf.as_ptr().add(last_seg)).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(end_seg)).add(end_offset)) }
            };

            let chunk = SegmentedSliceMut {
                segments: self.segments,
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
                    let _buf = *self.segments.as_ptr();
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
                        let (seg, off) = RawSegmentedVec::<T>::location(self.start);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
                        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
                        let seg_end = (*buf.as_ptr().add(seg)).add(cap);

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
            let (seg, off) = RawSegmentedVec::<T>::location(self.start);
            let buf = self.segments;
            let cap = RawSegmentedVec::<T>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
            self.start_seg_end =
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };
        }

        self.next_back()
    }
}

impl<'a, T> ExactSizeIterator for RChunksMut<'a, T> {}
impl<'a, T> FusedIterator for RChunksMut<'a, T> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) chunks of exact
/// size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact`]: SegmentedSlice::rchunks_exact
pub struct RChunksExact<'a, T> {
    segments: NonNull<*mut T>,
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

impl<'a, T> RChunksExact<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        let exact_end = slice.start + len;

        if exact_start == exact_end {
            return Self {
                segments: slice.segments,
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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(exact_start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            segments: slice.segments,
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
    pub fn remainder(&self) -> SegmentedSlice<'a, T> {
        SegmentedSlice::new(self.segments, self.full_start, self.start)
    }
}

impl<'a, T> Clone for RChunksExact<'a, T> {
    fn clone(&self) -> Self {
        Self {
            segments: self.segments,
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

impl<'a, T> Iterator for RChunksExact<'a, T> {
    type Item = SegmentedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let result = SegmentedSlice {
                segments: self.segments,
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
                    let buf = self.segments;
                    let seg_ptr = *buf.as_ptr().add(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location_for_end(self.end);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.end);
        let buf = self.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next()
    }
}

impl<'a, T> DoubleEndedIterator for RChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.start;

            // The chunk starts at self.start_ptr/start_seg.
            // We need the end of THIS chunk.
            let (end_seg, end_offset) = RawSegmentedVec::<T>::location(chunk_start + chunk_len);
            let buf = self.segments;
            let end_ptr = if end_offset == 0 && end_seg > 0 {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T>::location(chunk_start + chunk_len - 1);
                unsafe {
                    NonNull::new_unchecked((*buf.as_ptr().add(last_seg)).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(end_seg)).add(end_offset)) }
            };

            let result = SegmentedSlice {
                segments: self.segments,
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
                    let _buf = *self.segments.as_ptr();
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
                        let (seg, off) = RawSegmentedVec::<T>::location(self.start);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
                        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
                        let seg_end = (*buf.as_ptr().add(seg)).add(cap);

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
            let (seg, off) = RawSegmentedVec::<T>::location(self.start);
            let buf = self.segments;
            let cap = RawSegmentedVec::<T>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
            self.start_seg_end =
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };
        }

        self.next_back()
    }
}

impl<'a, T> ExactSizeIterator for RChunksExact<'a, T> {}
impl<'a, T> FusedIterator for RChunksExact<'a, T> {}

/// An iterator over a `SegmentedSlice` in (non-overlapping) mutable chunks of
/// exact size, starting at the end of the slice.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [`SegmentedSlice`].
///
/// [`rchunks_exact_mut`]: SegmentedSlice::rchunks_exact_mut
pub struct RChunksExactMut<'a, T> {
    segments: NonNull<*mut T>,
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

impl<'a, T> RChunksExactMut<'a, T> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk_size must be non-zero");
        let len = slice.len;
        let rem = len % chunk_size;
        let exact_start = slice.start + rem;
        let exact_end = slice.start + len;

        if exact_start == exact_end {
            return Self {
                segments: slice.segments,
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

        let buf = slice.segments;

        // Start cache
        let (mut start_seg, mut start_offset) = RawSegmentedVec::<T>::location(exact_start);
        if start_offset == 0 && start_seg > 0 {
            start_seg -= 1;
            start_offset = RawSegmentedVec::<T>::segment_capacity(start_seg);
        }
        let start_ptr = unsafe {
            let ptr = (*buf.as_ptr().add(start_seg)).add(start_offset);
            NonNull::new_unchecked(ptr)
        };
        let start_seg_end = unsafe {
            if true {
                let cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                NonNull::new_unchecked((*buf.as_ptr().add(start_seg)).add(cap))
            } else {
                NonNull::dangling()
            }
        };

        Self {
            segments: slice.segments,
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
    pub fn into_remainder(self) -> SegmentedSliceMut<'a, T> {
        SegmentedSliceMut::new(self.segments, self.full_start, self.start)
    }
}

impl<'a, T> Iterator for RChunksExactMut<'a, T> {
    type Item = SegmentedSliceMut<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.end - chunk_len;

            // The chunk ends at self.end_ptr/end_seg.
            let result = SegmentedSliceMut {
                segments: self.segments,
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
                    let buf = self.segments;
                    let seg_ptr = *buf.as_ptr().add(self.end_seg);
                    let offset = self.end_ptr.as_ptr().offset_from(seg_ptr) as usize;

                    if offset >= chunk_len {
                        // Same segment
                        self.end_ptr = NonNull::new_unchecked(self.end_ptr.as_ptr().sub(chunk_len));
                    } else {
                        // Crosses segment(s)
                        let (seg, off) = RawSegmentedVec::<T>::location_for_end(self.end);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
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
        let (mut seg, mut off) = RawSegmentedVec::<T>::location(self.end);
        let buf = self.segments;
        if off == 0 && seg > 0 {
            seg -= 1;
            off = RawSegmentedVec::<T>::segment_capacity(seg);
        }
        self.end_seg = seg;
        self.end_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };

        self.next()
    }
}

impl<'a, T> DoubleEndedIterator for RChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_len = self.chunk_size;
            let chunk_start = self.start;

            // The chunk starts at self.start_ptr/start_seg.
            // We need the end of THIS chunk.
            let (end_seg, end_offset) = RawSegmentedVec::<T>::location(chunk_start + chunk_len);
            let buf = self.segments;
            let end_ptr = if end_offset == 0 && end_seg > 0 {
                let (last_seg, last_offset) =
                    RawSegmentedVec::<T>::location(chunk_start + chunk_len - 1);
                unsafe {
                    NonNull::new_unchecked((*buf.as_ptr().add(last_seg)).add(last_offset + 1))
                }
            } else {
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(end_seg)).add(end_offset)) }
            };

            let result = SegmentedSliceMut {
                segments: self.segments,
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
                    let _buf = *self.segments.as_ptr();
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
                        let (seg, off) = RawSegmentedVec::<T>::location(self.start);
                        let ptr = (*buf.as_ptr().add(seg)).add(off);
                        let cap = RawSegmentedVec::<T>::segment_capacity(seg);
                        let seg_end = (*buf.as_ptr().add(seg)).add(cap);

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
            let (seg, off) = RawSegmentedVec::<T>::location(self.start);
            let buf = self.segments;
            let cap = RawSegmentedVec::<T>::segment_capacity(seg);
            self.start_seg = seg;
            self.start_ptr = unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(off)) };
            self.start_seg_end =
                unsafe { NonNull::new_unchecked((*buf.as_ptr().add(seg)).add(cap)) };
        }

        self.next_back()
    }
}

impl<'a, T> ExactSizeIterator for RChunksExactMut<'a, T> {}
impl<'a, T> FusedIterator for RChunksExactMut<'a, T> {}

/// An iterator over subslices separated by elements that match a predicate function.
#[derive(Clone)]
pub struct ChunkBy<'a, T, F> {
    slice: SegmentedSlice<'a, T>,
    pred: F,
}

impl<T: std::fmt::Debug, F> std::fmt::Debug for ChunkBy<'_, T, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkBy")
            .field("slice", &self.slice)
            .finish()
    }
}

impl<'a, T, F> ChunkBy<'a, T, F> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSlice<'a, T>, pred: F) -> Self {
        Self { slice, pred }
    }
}

impl<'a, T, F> Iterator for ChunkBy<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSlice<'a, T>;

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

impl<'a, T, F> DoubleEndedIterator for ChunkBy<'a, T, F>
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

impl<'a, T, F> FusedIterator for ChunkBy<'a, T, F> where F: FnMut(&T, &T) -> bool {}

/// An iterator over mutable subslices separated by elements that match a predicate function.
pub struct ChunkByMut<'a, T, F> {
    slice: SegmentedSliceMut<'a, T>,
    pred: F,
}

impl<T: std::fmt::Debug, F> std::fmt::Debug for ChunkByMut<'_, T, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkByMut")
            .field("slice", &self.slice)
            .finish()
    }
}

impl<'a, T, F> ChunkByMut<'a, T, F> {
    #[inline]
    pub(crate) fn new(slice: SegmentedSliceMut<'a, T>, pred: F) -> Self {
        Self { slice, pred }
    }
}

impl<'a, T, F> Iterator for ChunkByMut<'a, T, F>
where
    F: FnMut(&T, &T) -> bool,
{
    type Item = SegmentedSliceMut<'a, T>;

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
            segments: self.slice.segments,
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
            self.slice.segments,
            self.slice.start,
            self.slice.start + head_len,
        );

        // Update self.slice to tail
        self.slice = SegmentedSliceMut::new(
            self.slice.segments,
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

impl<'a, T, F> DoubleEndedIterator for ChunkByMut<'a, T, F>
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
            segments: self.slice.segments,
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
            self.slice.segments,
            self.slice.start + head_len, // start of tail
            self.slice.start + len,      // end of tail (original end)
        );

        // Update self.slice to head
        self.slice = SegmentedSliceMut::new(
            self.slice.segments,
            self.slice.start,
            self.slice.start + head_len,
        );

        Some(tail)
    }
}

impl<'a, T, F> FusedIterator for ChunkByMut<'a, T, F> where F: FnMut(&T, &T) -> bool {}
