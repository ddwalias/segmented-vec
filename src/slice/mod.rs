//! Slice management and manipulation for SegmentedVec.
//!
//! This module provides `SegmentedSlice` and `SegmentedSliceMut`, which are
//! analogous to `&[T]` and `&mut [T]` but for segmented (non-contiguous) storage.
//!
//! # Key Differences from std::slice
//!
//! Unlike standard slices, `SegmentedSlice` does not represent contiguous memory.
//! This means:
//!
//! - **No `as_ptr()` or `as_mut_ptr()`**: The underlying memory is not contiguous,
//!   so there is no single pointer that can represent the entire slice.
//!
//! - **No `as_ptr_range()`**: Similarly, pointer ranges don't make sense for
//!   non-contiguous storage.
//!
//! - **Split operations return `SegmentedSlice`**: Methods like `split_at()` return
//!   `SegmentedSlice` instead of `&[T]`.
//!
//! - **No `from_raw_parts()`**: Since the data structure is more complex than a
//!   simple pointer+length, raw parts construction isn't supported.
//!
//! - **Chunk methods that return `&[T]` are not available**: Methods like `as_chunks()`
//!   that return contiguous array references cannot be implemented. Instead, use
//!   the segment-based iteration methods.
//!
//! Most other slice methods are available with the same signatures.

pub mod index;
pub mod iter;

use allocator_api2::alloc::{Allocator, Global};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, RangeBounds};
use std::ptr::NonNull;

use self::index::SliceIndex;
use crate::raw_vec::RawSegmentedVec;
use crate::SegmentedVec;

// Re-export iterator types
pub use iter::{
    ChunkBy, ChunkByMut, Chunks, ChunksExact, ChunksExactMut, ChunksMut, RChunks, RChunksExact,
    RChunksExactMut, RChunksMut, RSplit, RSplitMut, RSplitN, SliceIter, SliceIterMut, Split,
    SplitInclusive, SplitInclusiveMut, SplitN, Windows,
};

/// A view into a portion of a `SegmentedVec`.
///
/// This is analogous to `&[T]` / `&mut [T]` but for segmented (non-contiguous) storage.
/// It provides most of the same methods as a standard slice, with the key
/// difference that the underlying memory may span multiple non-contiguous segments.
///
/// # Mutability
///
/// - `SegmentedSlice` provides read-only access
/// - `SegmentedSliceMut` provides read-write access
/// Immutable segmented slice.
///
/// This is analogous to `&[T]` but for segmented (non-contiguous) storage.
pub struct SegmentedSlice<'a, T, A: Allocator + 'a = Global> {
    /// Raw pointer to the underlying SegmentedVec's buffer.
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    /// Starting logical index (inclusive)
    pub(crate) start: usize,
    /// Length of the slice
    pub(crate) len: usize,
    /// Cached pointer to end (exclusive)
    pub(crate) end_ptr: NonNull<T>,
    /// Cached segment index for end
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T, A: Allocator + 'a> Clone for SegmentedSlice<'a, T, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, A: Allocator + 'a> Copy for SegmentedSlice<'a, T, A> {}

/// Mutable segmented slice.
///
/// This is analogous to `&mut [T]` but for segmented (non-contiguous) storage.
#[derive(Debug)]
pub struct SegmentedSliceMut<'a, T, A: Allocator + 'a = Global> {
    /// Raw pointer to the underlying SegmentedVec's buffer.
    pub(crate) buf: NonNull<RawSegmentedVec<T, A>>,
    /// Starting logical index (inclusive)
    pub(crate) start: usize,
    /// Length of the slice
    pub(crate) len: usize,
    /// Cached pointer to end (exclusive)
    pub(crate) end_ptr: NonNull<T>,
    /// Cached segment index for end
    pub(crate) end_seg: usize,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, A: Allocator + 'a> SegmentedSliceMut<'a, T, A> {
    #[inline]
    pub(crate) fn new(buf: NonNull<RawSegmentedVec<T, A>>, start: usize, end: usize) -> Self {
        let len = end - start;
        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                buf,
                start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }
        let buf_ref = unsafe { buf.as_ref() };
        let last_idx = end - 1;
        let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(last_idx);
        let end_ptr = unsafe {
            let ptr = buf_ref.segment_ptr(end_seg).add(end_offset + 1);
            NonNull::new_unchecked(ptr)
        };
        Self {
            buf,
            start,
            len,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> From<SegmentedSliceMut<'a, T, A>> for SegmentedSlice<'a, T, A> {
    #[inline(always)]
    fn from(s: SegmentedSliceMut<'a, T, A>) -> Self {
        SegmentedSlice {
            buf: s.buf,
            start: s.start,
            len: s.len,
            end_ptr: s.end_ptr,
            end_seg: s.end_seg,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, A: Allocator + 'a> SegmentedSlice<'a, T, A> {
    #[inline]
    pub(crate) fn new(buf: NonNull<RawSegmentedVec<T, A>>, start: usize, end: usize) -> Self {
        let len = end - start;
        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                buf,
                start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            };
        }
        let buf_ref = unsafe { buf.as_ref() };
        let last_idx = end - 1;
        let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(last_idx);
        // SAFETY: We checked len != 0, so buffer must have capacity if elements are ZST-sized or we are accessing valid memory.
        // Actually, for ZSTs size_of is 0, handled above.
        // For non-ZST, capacity > 0 is implied if we are creating a valid slice of len > 0.
        // But location() handles large indices by clamping or returning locations.
        // We trust location() returns a valid segment index for a valid `end-1`.
        let end_ptr = unsafe {
            let ptr = buf_ref.segment_ptr(end_seg).add(end_offset + 1);
            NonNull::new_unchecked(ptr)
        };
        Self {
            buf,
            start,
            len,
            end_ptr,
            end_seg,
            _marker: PhantomData,
        }
    }
    /// Returns the number of elements in the slice.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has a length of 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the first element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let vec: SegmentedVec<i32> = (0..10).collect();
    /// assert_eq!(vec.as_slice().first(), Some(&0));
    ///
    /// let empty: SegmentedVec<i32> = SegmentedVec::new();
    /// assert_eq!(empty.as_slice().first(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        Some(unsafe { &*self.buf.as_ref().ptr_at(self.start) })
    }

    pub fn split_first(&self) -> Option<(&T, SegmentedSlice<'a, T, A>)> {
        if let Some(first) = self.first() {
            let tail = SegmentedSlice {
                buf: self.buf,
                start: self.start + 1,
                len: self.len() - 1,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: self._marker,
            };

            return Some((first, tail));
        }

        None
    }

    /// Returns a reference to the last and all the rest of the elements of the slice,
    /// or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let vec: SegmentedVec<i32> = (0..5).collect();
    /// let slice = vec.as_slice();
    ///
    /// if let Some((last, rest)) = slice.split_last() {
    ///     assert_eq!(*last, 4);
    ///     assert_eq!(rest.len(), 4);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn split_last(&self) -> Option<(&T, SegmentedSlice<'a, T, A>)> {
        if let Some(last) = self.last() {
            let head = SegmentedSlice {
                buf: self.buf,
                start: self.start,
                len: self.len() - 1,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: self._marker,
            };

            return Some((last, head));
        }

        None
    }

    /// Returns a reference to the last element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let vec: SegmentedVec<i32> = (0..10).collect();
    /// assert_eq!(vec.as_slice().last(), Some(&9));
    ///
    /// let empty: SegmentedVec<i32> = SegmentedVec::new();
    /// assert_eq!(empty.as_slice().last(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };

        if self.end_ptr.as_ptr() > segment_base {
            Some(unsafe { &*self.end_ptr.as_ptr().sub(1) })
        } else {
            // Cold path: write_ptr is at the start of the active segment,
            // so the last element is in the previous (fully populated) segment.
            let prev_segment_index = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_segment_index);

            unsafe {
                let prev_segment_base = self.buf.as_ref().segment_ptr(prev_segment_index);
                Some(&*prev_segment_base.add(prev_cap - 1))
            }
        }
    }

    /// Returns an array reference to the first `N` items in the slice.
    #[inline]
    pub fn first_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            return None;
        }

        let (seg_idx, offset) = RawSegmentedVec::<T, A>::location(self.start);
        let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
        let available = seg_cap - offset;

        if available < N {
            return None;
        }

        unsafe {
            let ptr = self.buf.as_ref().segment_ptr(seg_idx).add(offset);
            Some(&*(ptr as *const [T; N]))
        }
    }

    /// Returns an array reference to the first `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_first_chunk<const N: usize>(&self) -> Option<(&[T; N], SegmentedSlice<'a, T, A>)> {
        let chunk = self.first_chunk::<N>()?;
        let (_, tail) = self.split_at(N);
        Some((chunk, tail))
    }

    /// Returns an array reference to the last `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_last_chunk<const N: usize>(&self) -> Option<(SegmentedSlice<'a, T, A>, &[T; N])> {
        let chunk = self.last_chunk::<N>()?;
        let (init, _) = self.split_at(self.len() - N);
        Some((init, chunk))
    }

    /// Returns an array reference to the last `N` items in the slice.
    #[inline]
    pub fn last_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            return None;
        }

        if N == 0 {
            return Some(unsafe { &*(std::ptr::NonNull::<[T; N]>::dangling().as_ptr()) });
        }

        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };
        let current_len = unsafe { self.end_ptr.as_ptr().offset_from(segment_base) as usize };

        if current_len >= N {
            return Some(unsafe { &*(self.end_ptr.as_ptr().sub(N) as *const [T; N]) });
        }

        if current_len == 0 && self.end_seg > 0 {
            let prev_seg = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_seg);

            if prev_cap >= N {
                let prev_base = unsafe { self.buf.as_ref().segment_ptr(prev_seg) };
                return Some(unsafe { &*(prev_base.add(prev_cap - N) as *const [T; N]) });
            }
        }

        None
    }
    /// Returns a reference to an element or subslice depending on the type of index.
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<I::Output<'_>>
    where
        I: SliceIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a reference to an element or subslice without doing bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*.
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> I::Output<'_>
    where
        I: SliceIndex<Self>,
    {
        index.get_unchecked(self)
    }

    /// Returns an iterator over the slice.
    #[inline]
    pub fn iter(&self) -> SliceIter<'a, T, A> {
        SliceIter::new(self)
    }

    /// Returns an iterator over overlapping windows of length `size`.
    #[inline]
    #[track_caller]
    pub fn windows(&self, size: usize) -> Windows<'a, T, A> {
        assert!(size != 0, "window size must be non-zero");
        Windows::new(*self, unsafe {
            core::num::NonZeroUsize::new_unchecked(size)
        })
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        Chunks::new(*self, chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    #[inline]
    #[track_caller]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExact::new(*self, chunk_size)
    }

    /// Returns a sub-slice of the slice.
    #[inline]
    #[must_use]
    pub fn sub_slice<R: RangeBounds<usize>>(&self, range: R) -> SegmentedSlice<'a, T, A> {
        let len = self.len();
        let start = match range.start_bound() {
            core::ops::Bound::Included(&s) => s,
            core::ops::Bound::Excluded(&s) => s + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(&e) => e + 1,
            core::ops::Bound::Excluded(&e) => e,
            core::ops::Bound::Unbounded => len,
        };
        assert!(start <= end && end <= len);
        SegmentedSlice::new(self.buf, self.start + start, self.start + end)
    }

    #[inline]
    pub fn split_once<F>(
        &self,
        pred: F,
    ) -> Option<(SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>)>
    where
        F: FnMut(&T) -> bool,
    {
        let index = self.iter().position(pred)?;
        Some((self.sub_slice(..index), self.sub_slice(index + 1..)))
    }

    #[inline]
    pub fn rsplit_once<F>(
        &self,
        pred: F,
    ) -> Option<(SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>)>
    where
        F: FnMut(&T) -> bool,
    {
        let index = self.iter().rposition(pred)?;
        Some((self.sub_slice(..index), self.sub_slice(index + 1..)))
    }
}

impl<'a, T, A: Allocator> SegmentedSliceMut<'a, T, A> {
    /// Returns a mutable iterator over the slice.
    #[inline]
    pub fn iter_mut(&mut self) -> SliceIterMut<'a, T, A> {
        SliceIterMut::new(self)
    }

    /// Returns the number of elements in the slice.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has a length of 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the first element of the slice, or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        Some(unsafe { &*self.buf.as_ref().ptr_at(self.start) })
    }

    /// Returns a mutable reference to the first element of the slice, or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            return None;
        }

        let start = self.start;
        Some(unsafe { &mut *self.buf.as_ref().ptr_at(start) })
    }

    pub fn split_first(&'a self) -> Option<(&'a T, SegmentedSlice<'a, T, A>)> {
        if let Some(first) = self.first() {
            let tail = SegmentedSlice {
                buf: self.buf,
                start: self.start + 1,
                len: self.len() - 1,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: PhantomData,
            };

            return Some((first, tail));
        }

        None
    }

    /// Returns a mutable reference to the first and all the rest of the elements of the slice,
    /// or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn split_first_mut(&'a mut self) -> Option<(&'a mut T, SegmentedSliceMut<'a, T, A>)> {
        let first_ptr = if let Some(first) = self.first_mut() {
            first as *mut T
        } else {
            return None;
        };

        unsafe {
            let tail = SegmentedSliceMut {
                buf: std::ptr::read(&self.buf),
                start: self.start + 1,
                len: self.len() - 1,
                end_ptr: self.end_ptr,
                end_seg: self.end_seg,
                _marker: self._marker,
            };

            Some((&mut *first_ptr, tail))
        }
    }

    /// Returns a reference to the last and all the rest of the elements of the slice,
    /// or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn split_last(&'a self) -> Option<(&'a T, SegmentedSlice<'a, T, A>)> {
        if self.is_empty() {
            return None;
        }

        // Use end_ptr to get last element efficiently
        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };

        // Determine the pointer to the last element.
        // This acts as the pointer to the returned element AND the exclusive end_ptr for the rest slice.
        let (last_ptr, last_seg) = if self.end_ptr.as_ptr() > segment_base {
            // Fast path: just decrement end_ptr
            (
                unsafe { NonNull::new_unchecked(self.end_ptr.as_ptr().sub(1)) },
                self.end_seg,
            )
        } else {
            // Cold path: step back to previous segment
            let prev_seg = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_seg);
            (
                unsafe {
                    NonNull::new_unchecked(
                        self.buf.as_ref().segment_ptr(prev_seg).add(prev_cap - 1),
                    )
                },
                prev_seg,
            )
        };

        let last = unsafe { &*last_ptr.as_ptr() };

        let rest = SegmentedSlice {
            buf: self.buf,
            start: self.start,
            len: self.len() - 1,
            end_ptr: last_ptr,
            end_seg: last_seg,
            _marker: PhantomData,
        };

        Some((last, rest))
    }

    /// Returns a mutable reference to the last and all the rest of the elements of the slice,
    /// or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn split_last_mut(&'a mut self) -> Option<(&'a mut T, SegmentedSliceMut<'a, T, A>)> {
        if self.is_empty() {
            return None;
        }

        // Use end_ptr to get last element efficiently
        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };

        // Determine the pointer to the last element.
        // This acts as the pointer to the returned element AND the exclusive end_ptr for the rest slice.
        let (last_ptr, last_seg) = if self.end_ptr.as_ptr() > segment_base {
            // Fast path: just decrement end_ptr
            (
                unsafe { NonNull::new_unchecked(self.end_ptr.as_ptr().sub(1)) },
                self.end_seg,
            )
        } else {
            // Cold path: step back to previous segment
            let prev_seg = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_seg);
            (
                unsafe {
                    NonNull::new_unchecked(
                        self.buf.as_ref().segment_ptr(prev_seg).add(prev_cap - 1),
                    )
                },
                prev_seg,
            )
        };

        let last = unsafe { &mut *last_ptr.as_ptr() };

        let rest = SegmentedSliceMut {
            buf: self.buf,
            start: self.start,
            len: self.len - 1,
            end_ptr: last_ptr,
            end_seg: last_seg,
            _marker: self._marker,
        };

        Some((last, rest))
    }

    /// Returns a reference to the last element of the slice, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let vec: SegmentedVec<i32> = (0..10).collect();
    /// assert_eq!(vec.as_slice().last(), Some(&9));
    ///
    /// let empty: SegmentedVec<i32> = SegmentedVec::new();
    /// assert_eq!(empty.as_slice().last(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };

        if self.end_ptr.as_ptr() > segment_base {
            Some(unsafe { &*self.end_ptr.as_ptr().sub(1) })
        } else {
            // Cold path: write_ptr is at the start of the active segment,
            // so the last element is in the previous (fully populated) segment.
            let prev_segment_index = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_segment_index);

            unsafe {
                let prev_segment_base = self.buf.as_ref().segment_ptr(prev_segment_index);
                Some(&*prev_segment_base.add(prev_cap - 1))
            }
        }
    }

    /// Returns a mutable reference to the last element of the slice, or `None` if it is empty.
    #[inline]
    #[must_use]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            return None;
        }

        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };

        if self.end_ptr.as_ptr() > segment_base {
            Some(unsafe { &mut *self.end_ptr.as_ptr().sub(1) })
        } else {
            // Cold path: write_ptr is at the start of the active segment,
            // so the last element is in the previous (fully populated) segment.
            let prev_segment_index = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_segment_index);

            unsafe {
                let prev_segment_base = self.buf.as_ref().segment_ptr(prev_segment_index);
                Some(&mut *prev_segment_base.add(prev_cap - 1))
            }
        }
    }

    /// Returns an array reference to the first `N` items in the slice.
    #[inline]
    pub fn first_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            return None;
        }

        let (seg_idx, offset) = RawSegmentedVec::<T, A>::location(self.start);
        let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
        let available = seg_cap - offset;

        if available < N {
            return None;
        }

        unsafe {
            let ptr = self.buf.as_ref().segment_ptr(seg_idx).add(offset);
            Some(&*(ptr as *const [T; N]))
        }
    }

    /// Returns an array reference to the first `N` items in the slice.
    #[inline]
    pub fn first_chunk_mut<const N: usize>(&mut self) -> Option<&mut [T; N]> {
        if self.len() < N {
            return None;
        }

        let (seg_idx, offset) = RawSegmentedVec::<T, A>::location(self.start);
        let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
        let available = seg_cap - offset;

        if available < N {
            return None;
        }

        unsafe {
            let ptr = self.buf.as_ref().segment_ptr(seg_idx).add(offset);
            Some(&mut *(ptr as *mut [T; N]))
        }
    }

    /// Returns an array reference to the first `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_first_chunk<const N: usize>(&self) -> Option<(&[T; N], SegmentedSlice<'a, T, A>)> {
        let chunk = self.first_chunk::<N>()?;
        let (_, tail) = self.split_at(N);
        Some((chunk, tail))
    }

    /// Returns an array reference to the first `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_first_chunk_mut<const N: usize>(
        &'a mut self,
    ) -> Option<(&'a mut [T; N], SegmentedSliceMut<'a, T, A>)> {
        let chunk_ptr = {
            let chunk_ref = self.first_chunk_mut::<N>()?;
            chunk_ref as *mut [T; N]
        };

        let tail = SegmentedSliceMut {
            buf: unsafe { std::ptr::read(&self.buf) },
            start: self.start + N,
            len: self.len() - N,
            end_ptr: self.end_ptr,
            end_seg: self.end_seg,
            _marker: self._marker,
        };

        Some((unsafe { &mut *chunk_ptr }, tail))
    }

    /// Returns an array reference to the last `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_last_chunk<const N: usize>(&self) -> Option<(SegmentedSlice<'a, T, A>, &[T; N])> {
        let chunk = self.last_chunk::<N>()?;
        let (init, _) = self.split_at(self.len() - N);
        Some((init, chunk))
    }

    /// Returns an array reference to the last `N` items in the slice and the remaining slice.
    #[inline]
    pub fn split_last_chunk_mut<const N: usize>(
        &mut self,
    ) -> Option<(SegmentedSliceMut<'a, T, A>, &mut [T; N])> {
        if self.len() < N {
            return None;
        }

        // We split the slice first to avoid double mutable borrow
        let (init, right) = self.split_at_mut(self.len() - N);

        // Check if the right part (the chunk) is contiguous
        // Since right.len == N, it is contiguous iff start and end fall in same segment
        let (start_seg, _) = RawSegmentedVec::<T, A>::location(right.start);
        let (end_seg, _) = RawSegmentedVec::<T, A>::location(right.start + N - 1);

        if start_seg == end_seg {
            // Safe to cast to array ref.
            // We get the pointer from the buffer manually to form the reference.
            let ptr = unsafe { right.buf.as_ref().ptr_at(right.start) };
            Some((init, unsafe { &mut *(ptr as *mut [T; N]) }))
        } else {
            None
        }
    }

    /// Returns an array reference to the last `N` items in the slice.
    #[inline]
    pub fn last_chunk_mut<const N: usize>(&mut self) -> Option<&mut [T; N]> {
        self.split_last_chunk_mut::<N>().map(|(_, chunk)| chunk)
    }

    /// Divides one mutable slice into two at an index.
    #[inline]
    #[track_caller]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (SegmentedSliceMut<'a, T, A>, SegmentedSliceMut<'a, T, A>) {
        assert!(mid <= self.len, "split point out of bounds");
        let len = self.len;
        let right_start = self.start + mid;

        let left = SegmentedSliceMut::new(self.buf, self.start, right_start);
        let right = SegmentedSliceMut::new(self.buf, right_start, self.start + len);

        (left, right)
    }

    /// Returns an array reference to the last `N` items in the slice.
    #[inline]
    pub fn last_chunk<const N: usize>(&self) -> Option<&[T; N]> {
        if self.len() < N {
            return None;
        }

        if N == 0 {
            return Some(unsafe { &*(std::ptr::NonNull::<[T; N]>::dangling().as_ptr()) });
        }

        let segment_base = unsafe { self.buf.as_ref().segment_ptr(self.end_seg) };
        let current_len = unsafe { self.end_ptr.as_ptr().offset_from(segment_base) as usize };

        if current_len >= N {
            return Some(unsafe { &*(self.end_ptr.as_ptr().sub(N) as *const [T; N]) });
        }

        if current_len == 0 && self.end_seg > 0 {
            let prev_seg = self.end_seg - 1;
            let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev_seg);

            if prev_cap >= N {
                let prev_base = unsafe { self.buf.as_ref().segment_ptr(prev_seg) };
                return Some(unsafe { &*(prev_base.add(prev_cap - N) as *const [T; N]) });
            }
        }

        None
    }

    /// Returns a reference to an element or subslice depending on the type of index.
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<I::Output<'_>>
    where
        I: SliceIndex<Self>,
    {
        index.get(self)
    }

    pub fn get_mut<I>(&mut self, index: I) -> Option<I::OutputMut<'_>>
    where
        I: SliceIndex<Self>,
    {
        index.get_mut(self)
    }

    /// Returns a reference to an element or subslice without doing bounds checking.
    #[inline]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> I::Output<'_>
    where
        I: SliceIndex<Self>,
    {
        unsafe { index.get_unchecked(self) }
    }

    /// Returns a reference to an element or subslice without doing bounds checking.
    #[inline]
    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> I::OutputMut<'_>
    where
        I: SliceIndex<Self>,
    {
        unsafe { index.get_unchecked_mut(self) }
    }

    /// Swaps two elements in the slice without doing bounds checking.
    #[inline]
    pub unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        let start = self.start;
        let pa = self.buf.as_ref().ptr_at(start + a);
        let pb = self.buf.as_ref().ptr_at(start + b);
        std::ptr::swap(pa, pb);
    }

    /// Reverses the order of elements in the slice, in place.
    #[inline]
    pub fn reverse(&mut self) {
        if core::mem::size_of::<T>() == 0 {
            return;
        }

        let len = self.len();
        if len <= 1 {
            return;
        }

        let mut remaining_swaps = len / 2;

        unsafe {
            // Determine initial right-side pointer and segment.
            // Use end_ptr to get last element efficiently.
            // Determine initial right-side pointer and segment.
            // Use end_ptr to get last element efficiently.
            let segment_base = self.buf.as_ref().segment_ptr(self.end_seg);
            let (mut r_ptr, mut r_seg, mut r_avail) = if self.end_ptr.as_ptr() > segment_base {
                // r_avail is exactly the number of elements in the current end segment
                (
                    self.end_ptr.as_ptr().sub(1),
                    self.end_seg,
                    self.end_ptr.as_ptr().offset_from(segment_base) as usize,
                )
            } else {
                // Cold path: write_ptr is at the start of the active segment,
                // so the last element is in the previous (fully populated) segment.
                let prev = self.end_seg - 1;
                let prev_cap = RawSegmentedVec::<T, A>::segment_capacity(prev);
                (
                    self.buf.as_ref().segment_ptr(prev).add(prev_cap - 1),
                    prev,
                    prev_cap,
                )
            };

            // Determine initial left-side pointer and segment.
            let (mut l_seg, l_off) = RawSegmentedVec::<T, A>::location(self.start);
            let mut l_ptr = self.buf.as_ref().segment_ptr(l_seg).add(l_off);

            // Calculate available items in current segments
            let mut l_avail = RawSegmentedVec::<T, A>::segment_capacity(l_seg) - l_off;

            while remaining_swaps > 0 {
                // Optimization: if pointers have converged to the same segment,
                // we can reverse the remaining middle part using slice::reverse.
                if l_seg == r_seg {
                    // The range to reverse is from l_ptr to r_ptr inclusive.
                    // Since l_ptr points to the first element to swap and r_ptr to the last,
                    // the length is offset + 1.
                    let count = r_ptr.offset_from(l_ptr) as usize + 1;
                    let slice = core::slice::from_raw_parts_mut(l_ptr, count);
                    slice.reverse();
                    break;
                }

                let count = core::cmp::min(remaining_swaps, core::cmp::min(l_avail, r_avail));

                // Swap `count` elements.
                // We use a helper function that takes slices to allow LLVM to use `noalias` optimization.
                // Safety:
                // 1. `l_ptr`...`l_ptr + count` is valid (checked by l_avail)
                // 2. `r_ptr - count + 1`...`r_ptr + 1` is valid (checked by r_avail)
                // 3. Regions are disjoint (different segments or convergence check handled)
                let left = core::slice::from_raw_parts_mut(l_ptr, count);
                let right = core::slice::from_raw_parts_mut(r_ptr.sub(count - 1), count);

                swap_reversed(left, right);

                remaining_swaps -= count;
                if remaining_swaps == 0 {
                    break;
                }

                l_avail -= count;
                r_avail -= count;

                // Advance pointers.
                l_ptr = l_ptr.add(count);
                r_ptr = r_ptr.sub(count);

                // If we exhausted the left segment, move to next
                if l_avail == 0 {
                    l_seg += 1;
                    let cap = RawSegmentedVec::<T, A>::segment_capacity(l_seg);
                    l_ptr = self.buf.as_ref().segment_ptr(l_seg);
                    l_avail = cap;
                }

                // If we exhausted the right segment, move to previous
                if r_avail == 0 {
                    r_seg -= 1;
                    let cap = RawSegmentedVec::<T, A>::segment_capacity(r_seg);
                    r_ptr = self.buf.as_ref().segment_ptr(r_seg).add(cap - 1);
                    r_avail = cap;
                }
            }
        }

        /// Swaps elements of `a` with elements of `b` in reverse order.
        /// `a[i]` is swapped with `b[n - 1 - i]`.
        #[inline]
        unsafe fn swap_reversed<T>(a: &mut [T], b: &mut [T]) {
            let n = a.len();
            debug_assert!(b.len() == n);

            let mut a_ptr = a.as_mut_ptr();
            let mut b_ptr = b.as_mut_ptr().add(n - 1);

            let a_end_ptr = a_ptr.add(n);
            while a_ptr < a_end_ptr {
                std::ptr::swap(a_ptr, b_ptr);
                a_ptr = a_ptr.add(1);
                b_ptr = b_ptr.sub(1);
            }
        }
    }

    /// Returns an iterator over the slice.
    #[inline]
    pub fn iter(&self) -> SliceIter<'a, T, A> {
        SliceIter::new(&self.as_slice())
    }

    /// Returns an immutable slice view.
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'a, T, A> {
        SegmentedSlice {
            buf: self.buf,
            start: self.start,
            len: self.len,
            end_ptr: self.end_ptr,
            end_seg: self.end_seg,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over overlapping windows of length `size`.
    #[inline]
    #[track_caller]
    pub fn windows(&self, size: usize) -> Windows<'a, T, A> {
        let size = core::num::NonZeroUsize::new(size).expect("window size must be non-zero");
        Windows::new(self.as_slice(), size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        Chunks::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    #[inline]
    #[track_caller]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExact::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements at a time, starting at the end.
    #[inline]
    #[track_caller]
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        RChunks::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end.
    #[inline]
    #[track_caller]
    pub fn rchunks_exact(&self, chunk_size: usize) -> iter::RChunksExact<'a, T, A> {
        iter::RChunksExact::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the beginning.
    #[inline]
    #[track_caller]
    pub const fn chunk_by<F>(&self, _pred: F) -> ChunkBy<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        todo!()
    }

    /// Divides one slice into two at an index.
    #[inline]
    #[track_caller]
    pub fn split_at(&self, mid: usize) -> (SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>) {
        assert!(mid <= self.len(), "mid > len");

        let left = SegmentedSlice::new(self.buf, self.start, self.start + mid);
        let right = SegmentedSlice::new(self.buf, self.start + mid, self.start + self.len);

        (left, right)
    }

    #[inline]
    #[must_use]
    #[track_caller]
    pub const unsafe fn split_at_unchecked(
        &self,
        _mid: usize,
    ) -> (SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>) {
        todo!()
    }

    /// Divides one slice into two at an index, returning `None` if the slice is too short.
    #[inline]
    pub fn split_at_checked(
        &self,
        mid: usize,
    ) -> Option<(SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>)> {
        if mid <= self.len() {
            Some(self.split_at(mid))
        } else {
            None
        }
    }

    /// Returns an iterator over subslices separated by elements that match `pred`.
    #[inline]
    pub fn split<P>(&self, pred: P) -> iter::Split<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::Split::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`.
    #[inline]
    pub fn split_inclusive<P>(&self, pred: P) -> iter::SplitInclusive<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitInclusive::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, starting from the end.
    #[inline]
    pub fn rsplit<P>(&self, pred: P) -> iter::RSplit<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplit::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, limited to at most `n` splits.
    #[inline]
    pub fn splitn<P>(&self, n: usize, pred: P) -> iter::SplitN<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitN::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            n,
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, limited to at most `n` splits, starting from the end.
    #[inline]
    pub fn rsplitn<P>(&self, n: usize, pred: P) -> iter::RSplitN<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplitN::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            n,
            pred,
        )
    }

    /// Returns `true` if the slice contains an element with the given value.
    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.iter().any(|elem| elem == x)
    }

    /// Returns `true` if `needle` is a prefix of the slice.
    #[inline]
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let n = needle.len();
        self.len() >= n && self.iter().take(n).eq(needle.iter())
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    #[inline]
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let n = needle.len();
        self.len() >= n && self.iter().skip(self.len() - n).eq(needle.iter())
    }

    #[must_use = "returns the subslice without modifying the original"]
    pub fn strip_prefix(&self, prefix: &[T]) -> Option<SegmentedSlice<'a, T, A>>
    where
        T: PartialEq,
    {
        let n = prefix.len();
        if n <= self.len() {
            let (head, tail) = self.split_at(n);
            if head == prefix {
                return Some(tail);
            }
        }
        None
    }

    /// Returns a reference to the element at the given index, or panics if the index is out of bounds.
    #[inline]
    pub fn index<I>(&self, index: I) -> I::Output<'_>
    where
        I: SliceIndex<Self>,
    {
        index.index(self)
    }
}

impl<'a, T, A: Allocator> SegmentedSliceMut<'a, T, A> {
    /// Returns a mutable sub-slice of the slice.
    #[inline]
    #[must_use]

    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> Self {
        let len = self.len();
        let start = match range.start_bound() {
            core::ops::Bound::Included(&s) => s,
            core::ops::Bound::Excluded(&s) => s + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(&e) => e + 1,
            core::ops::Bound::Excluded(&e) => e,
            core::ops::Bound::Unbounded => len,
        };
        assert!(start <= end && end <= len);
        Self::new(self.buf, self.start + start, self.start + end)
    }

    /// Returns a mutable reference to the element at the given index, or panics if the index is out of bounds.
    #[inline]
    pub fn index_mut<I>(&mut self, index: I) -> I::OutputMut<'_>
    where
        I: SliceIndex<Self>,
    {
        index.index_mut(self)
    }
}

impl<'a, T, A: Allocator> SegmentedSlice<'a, T, A> {
    /// Divides one slice into two at an index.
    #[inline]
    #[track_caller]
    pub fn split_at(&self, mid: usize) -> (SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>) {
        assert!(mid <= self.len(), "mid > len");

        let left = SegmentedSlice::new(self.buf, self.start, self.start + mid);
        let right = SegmentedSlice::new(self.buf, self.start + mid, self.start + self.len);

        (left, right)
    }

    /// Divides one slice into two at an index, returning `None` if the slice is too short.
    #[inline]
    pub fn split_at_checked(
        &self,
        mid: usize,
    ) -> Option<(SegmentedSlice<'a, T, A>, SegmentedSlice<'a, T, A>)> {
        if mid <= self.len() {
            Some(self.split_at(mid))
        } else {
            None
        }
    }

    /// Returns `true` if the slice contains an element with the given value.
    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.iter().any(|elem| elem == x)
    }

    /// Returns `true` if `needle` is a prefix of the slice.
    #[inline]
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let n = needle.len();
        self.len() >= n && self.iter().take(n).eq(needle.iter())
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    #[inline]
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let n = needle.len();
        self.len() >= n && self.iter().skip(self.len() - n).eq(needle.iter())
    }

    /// Binary searches this slice for a given element.
    #[inline]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this slice with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let mut size = self.len();
        if size == 0 {
            return Err(0);
        }

        let mut base = 0usize;

        while size > 1 {
            let half = size / 2;
            let mid = base + half;

            let cmp = f(unsafe { self.get_unchecked(mid) });

            base = if cmp == Ordering::Greater { base } else { mid };
            size -= half;
        }

        let cmp = f(unsafe { self.get_unchecked(base) });
        if cmp == Ordering::Equal {
            Ok(base)
        } else {
            Err(base + (cmp == Ordering::Less) as usize)
        }
    }

    /// Binary searches this slice with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Returns an iterator over `chunk_size` elements at a time, starting at the end.
    #[inline]
    #[track_caller]
    pub fn rchunks(&self, chunk_size: usize) -> iter::RChunks<'a, T, A> {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        iter::RChunks::new(*self, chunk_size)
    }

    /// Compares two slices lexicographically.
    pub fn cmp_with(&self, other: &SegmentedSlice<'_, T, A>) -> Ordering
    where
        T: Ord,
    {
        self.iter().cmp(other.iter())
    }

    /// Copies elements into a `Vec`.
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Returns an iterator over subslices separated by elements that match `pred`.
    #[inline]
    pub fn split<P>(&self, pred: P) -> iter::Split<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::Split::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, starting from the end.
    #[inline]
    pub fn rsplit<P>(&self, pred: P) -> iter::RSplit<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplit::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`.
    #[inline]
    pub fn split_inclusive<P>(&self, pred: P) -> iter::SplitInclusive<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitInclusive::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, limited to at most `n` splits.
    #[inline]
    pub fn splitn<P>(&self, n: usize, pred: P) -> iter::SplitN<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitN::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            n,
            pred,
        )
    }

    /// Returns an iterator over subslices separated by elements that match `pred`, limited to at most `n` splits, starting from the end.
    #[inline]
    pub fn rsplitn<P>(&self, n: usize, pred: P) -> iter::RSplitN<'a, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplitN::new(
            SegmentedSlice::new(self.buf, self.start, self.start + self.len),
            n,
            pred,
        )
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end.
    #[inline]
    #[track_caller]
    pub fn rchunks_exact(&self, chunk_size: usize) -> iter::RChunksExact<'a, T, A> {
        iter::RChunksExact::new(*self, chunk_size)
    }
}

impl<'a, T, A: Allocator> SegmentedSliceMut<'a, T, A> {
    /// Binary searches this slice for a given element.
    #[inline]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this slice with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let mut size = self.len();
        if size == 0 {
            return Err(0);
        }

        let mut base = 0usize;

        while size > 1 {
            let half = size / 2;
            let mid = base + half;

            // Use get_unchecked (SegmentedSliceMut should have it too)
            let cmp = f(unsafe { self.get_unchecked(mid) });

            base = if cmp == Ordering::Greater { base } else { mid };
            size -= half;
        }

        let cmp = f(unsafe { self.get_unchecked(base) });
        if cmp == Ordering::Equal {
            Ok(base)
        } else {
            Err(base + (cmp == Ordering::Less) as usize)
        }
    }

    /// Binary searches this slice with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Compares two slices lexicographically.
    pub fn cmp_with(&self, other: &SegmentedSlice<'_, T, A>) -> Ordering
    where
        T: Ord,
    {
        self.iter().cmp(other.iter())
    }

    /// Copies elements into a `Vec`.
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }
}

impl<'a, T, A: Allocator> SegmentedSliceMut<'a, T, A> {
    /// Fills `self` with elements by cloning `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for i in 0..self.len() {
            unsafe {
                let start = self.start;
                let ptr = self.buf.as_ref().ptr_at(start + i);
                *ptr = value.clone();
            }
        }
    }

    /// Fills `self` with elements returned by calling a closure repeatedly.
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        for i in 0..self.len() {
            unsafe {
                let start = self.start;
                let ptr = self.buf.as_ref().ptr_at(start + i);
                *ptr = f();
            }
        }
    }

    /// Rotates the slice in-place such that the first `mid` elements move
    /// to the end of the slice.
    #[track_caller]
    pub fn rotate_left(&mut self, mid: usize) {
        let len = self.len();
        assert!(mid <= len, "mid > len");
        if mid == 0 || mid == len {
            return;
        }

        // Use the "juggling algorithm" which is efficient for non-contiguous storage
        let gcd = gcd(mid, len);
        for i in 0..gcd {
            let temp = unsafe { std::ptr::read(self.buf.as_ref().ptr_at(self.start + i)) };
            let mut j = i;
            loop {
                let k = if j + mid >= len {
                    j + mid - len
                } else {
                    j + mid
                };
                if k == i {
                    break;
                }
                unsafe {
                    let start = self.start;
                    let src = self.buf.as_ref().ptr_at(start + k);
                    let dst = self.buf.as_ref().ptr_at(start + j);
                    std::ptr::copy_nonoverlapping(src, dst, 1);
                }
                j = k;
            }
            unsafe {
                let start = self.start;
                std::ptr::write(self.buf.as_ref().ptr_at(start + j), temp);
            }
        }
    }

    /// Rotates the slice in-place such that the last `k` elements move
    /// to the front of the slice.
    #[track_caller]
    pub fn rotate_right(&mut self, k: usize) {
        let len = self.len();
        assert!(k <= len, "k > len");
        if k == 0 || k == len {
            return;
        }
        self.rotate_left(len - k);
    }

    /// Swaps two elements in the slice.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    #[inline]
    #[track_caller]
    pub fn swap(&mut self, a: usize, b: usize) {
        let len = self.len();
        if a >= len || b >= len {
            panic!("swap index out of bounds");
        }
        unsafe {
            let ptr_a = self.buf.as_ref().ptr_at(self.start + a);
            let ptr_b = self.buf.as_ref().ptr_at(self.start + b);
            std::ptr::swap(ptr_a, ptr_b);
        }
    }

    /// Divides one mutable slice into two at an index, returning `None` if the slice is too short.
    #[inline]
    pub fn split_at_mut_checked(
        &mut self,
        mid: usize,
    ) -> Option<(SegmentedSliceMut<'_, T, A>, SegmentedSliceMut<'_, T, A>)> {
        if mid > self.len() {
            None
        } else {
            // SAFETY: mid <= len checked above
            Some(self.split_at_mut(mid))
        }
    }

    /// Returns an iterator over mutable subslices separated by elements that match `pred`.
    #[inline]
    pub fn split_mut<P>(&mut self, pred: P) -> iter::SplitMut<'_, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitMut::new(self.slice_mut(..), pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that match `pred`, starting from the end.
    #[inline]
    pub fn rsplit_mut<P>(&mut self, pred: P) -> iter::RSplitMut<'_, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplitMut::new(self.slice_mut(..), pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that match `pred`.
    /// The matched element is contained in the end of the previous subslice.
    #[inline]
    pub fn split_inclusive_mut<P>(&mut self, pred: P) -> iter::SplitInclusiveMut<'_, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitInclusiveMut::new(self.slice_mut(..), pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that match `pred`, limited to at most `n` splits.
    #[inline]
    pub fn splitn_mut<P>(&mut self, n: usize, pred: P) -> iter::SplitNMut<'_, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::SplitNMut::new(self.slice_mut(..), n, pred)
    }

    /// Returns an iterator over mutable subslices separated by elements that match `pred`, limited to at most `n` splits, starting from the end.
    #[inline]
    pub fn rsplitn_mut<P>(&mut self, n: usize, pred: P) -> iter::RSplitNMut<'_, T, A, P>
    where
        P: FnMut(&T) -> bool,
    {
        iter::RSplitNMut::new(self.slice_mut(..), n, pred)
    }

    /// Sorts the slice.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(n * log(n)) worst-case.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is an adaptive, iterative merge sort inspired by
    /// [timsort](https://en.wikipedia.org/wiki/Timsort).
    /// It is designed to be very fast in cases where the slice is nearly sorted, or consists of
    /// two or more sorted sequences concatenated one after another.
    ///
    /// Also, it allocates temporary storage half the size of `self`, but for short slices a
    /// non-allocating version is used.
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort_by(|a, b| a.cmp(b))
    }

    /// Sorts the slice with a comparator function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(n * log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements in the slice. If
    /// the ordering is not total, the order of the elements is unspecified. An order is a
    /// total order if it is (for all `a`, `b`, and `c`):
    ///
    /// * total and antisymmetric: exactly one of `a < b`, `a == b`, or `a > b` is true, and
    /// * transitive: `a < b` and `b < c` implies `a < c`.  The same must hold for both `==` and `>`.
    ///
    /// For example, while `f64` doesn't implement `Ord` because `NaN != NaN`, we can use
    /// `partial_cmp` as our sort function when we know the slice doesn't contain a `NaN`.
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut v: SegmentedVec<f64> = SegmentedVec::new();
    /// v.push(5.0);
    /// v.push(4.0);
    /// v.push(1.0);
    /// v.push(3.0);
    /// v.push(2.0);
    /// v.push(5.5);
    ///
    /// v.as_mut_slice().sort_by(|a, b| a.partial_cmp(b).unwrap());
    /// assert!(v.as_slice().windows(2).all(|w| w[0] <= w[1]));
    /// ```
    #[inline]
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        merge_sort(self, &mut compare);
    }

    /// Sorts the slice with a key extraction function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(m * n * log(n))
    /// worst-case, where the key function is O(m).
    ///
    /// For expensive key functions (e.g. functions that are not O(1)), it is substantially faster
    /// to move the data into a `Vec`, sort it there, and move it back.
    #[inline]
    pub fn sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_by(|a, b| f(a).cmp(&f(b)))
    }

    /// Sorts the slice, but might not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place
    /// (i.e., does not allocate), and O(n * log(n)) worst-case.
    ///
    /// # Current implementation
    ///
    /// The current algorithm is based on [heapsort](https://en.wikipedia.org/wiki/Heapsort).
    #[inline]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable_by(|a, b| a.cmp(b))
    }

    /// Sorts the slice with a comparator function, but might not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place
    /// (i.e., does not allocate), and O(n * log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements in the slice. If
    /// the ordering is not total, the order of the elements is unspecified. An order is a
    /// total order if it is (for all `a`, `b`, and `c`):
    ///
    /// * total and antisymmetric: exactly one of `a < b`, `a == b`, or `a > b` is true, and
    /// * transitive: `a < b` and `b < c` implies `a < c`.  The same must hold for both `==` and `>`.
    ///
    /// For example, while `f64` doesn't implement `Ord` because `NaN != NaN`, we can use
    /// `partial_cmp` as our sort function when we know the slice doesn't contain a `NaN`.
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        heapsort(self, &mut compare);
    }

    /// Sorts the slice with a key extraction function, but might not preserve the order of equal
    /// elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place
    /// (i.e., does not allocate), and O(m * n * log(n)) worst-case, where the key function is
    /// O(m).
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_unstable_by(|a, b| f(a).cmp(&f(b)))
    }

    /// Copies elements from another slice into `self`.
    #[track_caller]
    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        assert_eq!(
            self.len(),
            src.len(),
            "source slice length ({}) does not match destination slice length ({})",
            src.len(),
            self.len()
        );

        for (i, &val) in src.iter().enumerate() {
            unsafe {
                let start = self.start;
                let ptr = self.buf.as_ref().ptr_at(start + i);
                *ptr = val;
            }
        }
    }

    /// Copies elements from another slice into `self`.
    #[track_caller]
    pub fn clone_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        assert_eq!(
            self.len(),
            src.len(),
            "source slice length ({}) does not match destination slice length ({})",
            src.len(),
            self.len()
        );

        for (i, val) in src.iter().enumerate() {
            unsafe {
                let start = self.start;
                let ptr = self.buf.as_ref().ptr_at(start + i);
                *ptr = val.clone();
            }
        }
    }
}

impl<T, A: Allocator> Index<usize> for SegmentedSlice<'_, T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, A: Allocator> Index<usize> for SegmentedSliceMut<'_, T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, A: Allocator> IndexMut<usize> for SegmentedSliceMut<'_, T, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

/// Compute GCD using Euclidean algorithm
#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Merge sort for stable sorting
fn merge_sort<T, A: Allocator, F>(slice: &mut SegmentedSliceMut<'_, T, A>, compare: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let len = slice.len();
    if len <= 1 {
        return;
    }

    // For small slices, use insertion sort
    if len <= 32 {
        insertion_sort(slice, compare);
        return;
    }

    // Create temporary storage for the merge
    let mut temp: Vec<T> = Vec::with_capacity(len);

    // Sort left and right halves using insertion sort on smaller chunks
    // For simplicity, we use a bottom-up merge sort approach
    let mut width = 1;
    while width < len {
        let mut i = 0;
        while i < len {
            let left = i;
            let mid = std::cmp::min(i + width, len);
            let right = std::cmp::min(i + 2 * width, len);

            // Merge [left, mid) and [mid, right)
            merge_inplace(slice, left, mid, right, compare, &mut temp);
            i += 2 * width;
        }
        width *= 2;
    }
}

/// Merge two sorted runs in-place using temporary storage
fn merge_inplace<T, A: Allocator, F>(
    slice: &mut SegmentedSliceMut<'_, T, A>,
    left: usize,
    mid: usize,
    right: usize,
    compare: &mut F,
    temp: &mut Vec<T>,
) where
    F: FnMut(&T, &T) -> Ordering,
{
    if mid >= right || left >= mid {
        return;
    }

    // Copy left half to temp
    temp.clear();
    let start = slice.start;
    for i in left..mid {
        temp.push(unsafe { std::ptr::read(slice.buf.as_ref().ptr_at(start + i)) });
    }

    let mut i = 0; // Index into temp (left half)
    let mut j = mid; // Index into right half (in-place)
    let mut k = left; // Index into output

    while i < temp.len() && j < right {
        let start = slice.start;
        if compare(&temp[i], unsafe { &*slice.buf.as_ref().ptr_at(start + j) }) != Ordering::Greater
        {
            unsafe {
                let start = slice.start;
                std::ptr::write(
                    slice.buf.as_ref().ptr_at(start + k),
                    std::ptr::read(&temp[i]),
                );
            }
            i += 1;
        } else {
            unsafe {
                let start = slice.start;
                let val = std::ptr::read(slice.buf.as_ref().ptr_at(start + j));
                std::ptr::write(slice.buf.as_ref().ptr_at(start + k), val);
            }
            j += 1;
        }
        k += 1;
    }

    // Copy remaining elements from temp
    while i < temp.len() {
        unsafe {
            let start = slice.start;
            std::ptr::write(
                slice.buf.as_ref().ptr_at(start + k),
                std::ptr::read(&temp[i]),
            );
        }
        i += 1;
        k += 1;
    }

    // Prevent temp from dropping the moved values
    // Safety: We've moved all values out of temp via ptr::read, so we clear
    // the length to prevent double-free on drop
    unsafe { temp.set_len(0) };
}

/// Insertion sort for small slices
fn insertion_sort<T, A: Allocator, F>(slice: &mut SegmentedSliceMut<'_, T, A>, compare: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let len = slice.len();
    for i in 1..len {
        let mut j = i;
        while j > 0 {
            let start = slice.start;
            let cmp = compare(
                unsafe { &*slice.buf.as_ref().ptr_at(start + j - 1) },
                unsafe { &*slice.buf.as_ref().ptr_at(start + j) },
            );
            if cmp == Ordering::Greater {
                slice.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
}

/// Heapsort for unstable sorting
fn heapsort<T, A: Allocator, F>(slice: &mut SegmentedSliceMut<'_, T, A>, compare: &mut F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let len = slice.len();
    if len <= 1 {
        return;
    }

    // Build max heap
    for i in (0..len / 2).rev() {
        sift_down(slice, i, len, compare);
    }

    // Extract elements from heap
    for i in (1..len).rev() {
        slice.swap(0, i);
        sift_down(slice, 0, i, compare);
    }
}

/// Sift down for heapsort
fn sift_down<T, A: Allocator, F>(
    slice: &mut SegmentedSliceMut<'_, T, A>,
    mut root: usize,
    end: usize,
    compare: &mut F,
) where
    F: FnMut(&T, &T) -> Ordering,
{
    loop {
        let mut max = root;
        let left = 2 * root + 1;
        let right = 2 * root + 2;

        let start = slice.start;
        if left < end {
            let cmp = compare(
                unsafe { &*slice.buf.as_ref().ptr_at(start + max) },
                unsafe { &*slice.buf.as_ref().ptr_at(start + left) },
            );
            if cmp == Ordering::Less {
                max = left;
            }
        }

        if right < end {
            let cmp = compare(
                unsafe { &*slice.buf.as_ref().ptr_at(start + max) },
                unsafe { &*slice.buf.as_ref().ptr_at(start + right) },
            );
            if cmp == Ordering::Less {
                max = right;
            }
        }

        if max == root {
            return;
        }

        slice.swap(root, max);
        root = max;
    }
}

impl<T, A: Allocator> SegmentedVec<T, A> {
    /// Returns a slice of the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let vec: SegmentedVec<i32> = (0..10).collect();
    /// let slice = vec.as_slice();
    ///
    /// assert_eq!(slice.len(), 10);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'_, T, A> {
        SegmentedSlice {
            buf: NonNull::from(&self.buf),
            start: 0,
            len: self.len,
            end_ptr: self.write_ptr,
            end_seg: self.active_segment_index,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable slice of the entire vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = (0..10).collect();
    /// vec.as_mut_slice().fill(42);
    ///
    /// assert_eq!(vec[0], 42);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> SegmentedSliceMut<'_, T, A> {
        let len = self.len();
        SegmentedSliceMut::new(NonNull::from(&self.buf), 0, len)
    }

    /// Returns a slice over the given range.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[inline]
    #[track_caller]
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> SegmentedSlice<'_, T, A> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.len(),
        };

        assert!(start <= end, "slice start > end");
        assert!(end <= self.len(), "slice end out of bounds");

        let len = end - start;
        if len == 0 {
            SegmentedSlice {
                buf: NonNull::from(&self.buf),
                start,
                len: 0,
                end_ptr: NonNull::dangling(),
                end_seg: 0,
                _marker: PhantomData,
            }
        } else {
            // Compute start optimization fields
            let (start_seg, start_offset) = RawSegmentedVec::<T, A>::location(start);
            let _start_ptr = unsafe { self.buf.segment_ptr(start_seg).add(start_offset) };

            // Compute end optimization fields
            let (mut end_seg, end_offset) = RawSegmentedVec::<T, A>::location(end);
            let end_ptr = if end_seg >= self.buf.segment_count() {
                let (last_seg, last_offset) = RawSegmentedVec::<T, A>::location(end - 1);
                end_seg = last_seg;
                unsafe { self.buf.segment_ptr(last_seg).add(last_offset + 1) }
            } else {
                unsafe { self.buf.segment_ptr(end_seg).add(end_offset) }
            };

            SegmentedSlice {
                buf: NonNull::from(&self.buf),
                start,
                len,
                end_ptr: unsafe { NonNull::new_unchecked(end_ptr) },
                end_seg,
                _marker: PhantomData,
            }
        }
    }

    /// Returns a mutable slice over the given range.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[inline]
    #[track_caller]
    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> SegmentedSliceMut<'_, T, A> {
        let len = self.len();
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => len,
        };

        assert!(start <= end, "slice start > end");
        assert!(end <= len, "slice end out of bounds");

        SegmentedSliceMut::new(NonNull::from(&self.buf), start, end)
    }
}

impl<T: std::fmt::Debug, A: Allocator> std::fmt::Debug for SegmentedSlice<'_, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq, A1: Allocator, A2: Allocator> PartialEq<SegmentedSlice<'_, T, A2>>
    for SegmentedSlice<'_, T, A1>
{
    fn eq(&self, other: &SegmentedSlice<'_, T, A2>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: PartialEq, A: Allocator> PartialEq<[T]> for SegmentedSlice<'_, T, A> {
    fn eq(&self, other: &[T]) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: PartialEq, A: Allocator> PartialEq<&[T]> for SegmentedSlice<'_, T, A> {
    fn eq(&self, other: &&[T]) -> bool {
        self.eq(*other)
    }
}

impl<T: PartialEq, A: Allocator> PartialEq<Vec<T>> for SegmentedSlice<'_, T, A> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: Eq, A: Allocator> Eq for SegmentedSlice<'_, T, A> {}

impl<T: PartialOrd, A1: Allocator, A2: Allocator> PartialOrd<SegmentedSlice<'_, T, A2>>
    for SegmentedSlice<'_, T, A1>
{
    fn partial_cmp(&self, other: &SegmentedSlice<'_, T, A2>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, A: Allocator> Ord for SegmentedSlice<'_, T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

// ============================================================================
// Hash implementation
// ============================================================================

impl<T: std::hash::Hash, A: Allocator> std::hash::Hash for SegmentedSlice<'_, T, A> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for elem in self.iter() {
            elem.hash(state);
        }
    }
}

unsafe impl<T: Sync, A: Allocator + Sync> Sync for SegmentedSlice<'_, T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Send for SegmentedSlice<'_, T, A> {}
