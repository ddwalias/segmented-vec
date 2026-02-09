//! A segmented growable vector with stable pointers.
//!
//! Unlike `Vec`, pushing new elements never invalidates pointers to existing elements.
//! This is achieved by storing elements in segments of exponentially growing sizes.
//!
//! # Example
//!
//! ```
//! use segmented_vec::SegmentedVec;
//!
//! let mut vec: SegmentedVec<i32> = SegmentedVec::new();
//! vec.push(1);
//! vec.push(2);
//!
//! // Get a pointer to the first element
//! let ptr = &vec[0] as *const i32;
//!
//! // Push more elements - the pointer remains valid!
//! for i in 3..100 {
//!     vec.push(i);
//! }
//!
//! // The pointer is still valid
//! assert_eq!(unsafe { *ptr }, 1);
//! ```

mod drain;
mod into_iter;
mod peek_mut;
mod raw_vec;
pub mod slice;

mod spec_extend;
mod spec_from_iter;

use allocator_api2::alloc::{Allocator, Global};
pub use drain::Drain;
pub use into_iter::IntoIter;
pub use peek_mut::PeekMut;
pub use slice::index::SliceIndex;
pub use slice::{
    Chunks, ChunksExact, RChunks, SegmentedSlice, SegmentedSliceMut, SliceIter, SliceIterMut,
    Windows,
};
pub use slice::{SliceIter as Iter, SliceIterMut as IterMut};

use raw_vec::RawSegmentedVec;
use spec_from_iter::SpecFromIter;
use std::alloc::Layout;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

/// The error type for `try_reserve` operations.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum TryReserveErrorKind {
    /// The capacity computation overflowed.
    CapacityOverflow,
    /// Memory allocation failed.
    AllocError { layout: Layout },
}

impl std::fmt::Display for TryReserveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            TryReserveErrorKind::CapacityOverflow => {
                write!(f, "memory allocation failed due to capacity overflow")
            }
            TryReserveErrorKind::AllocError { layout } => {
                write!(f, "memory allocation of {} bytes failed", layout.size())
            }
        }
    }
}

impl std::error::Error for TryReserveError {}

impl TryReserveError {
    pub(crate) fn capacity_overflow() -> Self {
        Self {
            kind: TryReserveErrorKind::CapacityOverflow,
        }
    }

    pub(crate) fn alloc_error(layout: Layout) -> Self {
        Self {
            kind: TryReserveErrorKind::AllocError { layout },
        }
    }
}

/// A segmented vector with stable pointers.
///
/// `SegmentedVec` stores elements in a series of exponentially growing segments.
/// Unlike `Vec`, pushing new elements never invalidates pointers to existing elements
/// because segments are never moved or reallocated.
///
/// # Memory Layout
///
/// Elements are stored in segments of sizes: MIN_CAP, 2*MIN_CAP, 4*MIN_CAP, ...
/// where MIN_CAP depends on the element size:
/// - 8 for 1-byte elements
/// - 4 for elements ≤ 1 KiB
/// - 1 for larger elements
///
/// With 64 segments, this can hold more than 2^64 elements.
pub struct SegmentedVec<T, A: Allocator = Global> {
    /// Low-level segment allocation management
    pub(crate) buf: RawSegmentedVec<T, A>,
    /// Number of initialized elements
    pub(crate) len: usize,
    /// Cached pointer to the next write position (for fast push)
    pub(crate) write_ptr: NonNull<T>,
    /// Pointer to the end of the current segment
    pub(crate) segment_end: NonNull<T>,
    /// Index of the current active segment
    pub(crate) active_segment_index: usize,
    /// Marker for drop check
    _marker: PhantomData<T>,
}

impl<T> SegmentedVec<T> {
    /// Constructs a new, empty `SegmentedVec<T>`.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// use segmented_vec::SegmentedVec;
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// ```
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self::new_in(Global)
    }

    /// Constructs a new, empty `SegmentedVec<T>` with at least the specified capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating additional segments.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }

    /// Constructs a new, empty `SegmentedVec<T>` with at least the specified capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacity exceeds `isize::MAX` _bytes_,
    /// or if the allocator reports allocation failure.
    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, Global)
    }

    /// Creates a `SegmentedVec<T>` by calling a function with each index.
    #[inline]
    pub fn from_fn<F>(length: usize, f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        // TODO: implement TrustedLen optimized version
        (0..length).map(f).collect()
    }
}

// Core implementation
impl<T, A: Allocator> SegmentedVec<T, A> {
    /// Creates a new `SegmentedVec` with at least the specified capacity.
    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let buf = RawSegmentedVec::with_capacity_in(capacity, alloc);

        // Improve initialization optimization by computing fields before struct creation.
        // This avoids writing nulls and then immediately overwriting them.
        let (write_ptr, segment_end) = if capacity > 0 && buf.segment_count() > 0 {
            unsafe {
                let ptr: *mut T = buf.segment_ptr(0);
                (
                    NonNull::new_unchecked(ptr),
                    NonNull::new_unchecked(ptr.add(RawSegmentedVec::<T, A>::segment_capacity(0))),
                )
            }
        } else {
            (NonNull::dangling(), NonNull::dangling())
        };

        Self {
            buf,
            len: 0,
            write_ptr,
            segment_end,
            active_segment_index: 0,
            _marker: PhantomData,
        }
    }

    /// Updates the cached write pointer based on the current length.
    #[inline]
    fn update_write_ptr_for_len(&mut self, len: usize) {
        if self.buf.segment_count() == 0 {
            self.write_ptr = NonNull::dangling();
            self.segment_end = NonNull::dangling();
            self.active_segment_index = usize::MAX;
            return;
        }

        if len == 0 {
            // Point to start of first segment
            unsafe {
                self.active_segment_index = 0;
                let ptr = self.buf.segment_ptr(0);
                self.write_ptr = NonNull::new_unchecked(ptr);
                let cap = RawSegmentedVec::<T, A>::segment_capacity(0);
                self.segment_end = NonNull::new_unchecked(ptr.add(cap));
            }
        } else {
            // Find segment and offset for current length
            let (mut seg_idx, mut offset) = RawSegmentedVec::<T, A>::location(len);

            // If shrinking reduced segments such that location(len) points to a deallocated segment,
            // we must step back to the end of the previous segment.
            if seg_idx >= self.buf.segment_count() {
                debug_assert_eq!(seg_idx, self.buf.segment_count());
                debug_assert_eq!(offset, 0);
                if seg_idx > 0 {
                    seg_idx -= 1;
                    offset = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
                }
            }

            let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);

            unsafe {
                self.active_segment_index = seg_idx;
                let seg_ptr = self.buf.segment_ptr(seg_idx);
                self.write_ptr = NonNull::new_unchecked(seg_ptr.add(offset));
                self.segment_end = NonNull::new_unchecked(seg_ptr.add(seg_cap));
            }
        }
    }

    /// Appends an element to the back of the collection.
    #[inline]
    pub fn push(&mut self, value: T) {
        let _ = self.push_mut(value);
    }

    /// Appends an element and returns a mutable reference to it.
    #[inline]
    #[must_use = "if you don't need a reference to the value, use `SegmentedVec::push` instead"]
    pub fn push_mut(&mut self, value: T) -> &mut T {
        if std::mem::size_of::<T>() == 0 {
            self.len += 1;
            // For ZST, any aligned non-null pointer is valid.
            // We use dangling() which is non-null and aligned.
            return unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() };
        }

        #[cold]
        #[inline(never)]
        fn push_slow<T, A: Allocator>(this: &mut SegmentedVec<T, A>, value: T) -> &mut T {
            unsafe {
                // Advance to the next segment (wrapping from usize::MAX to 0 for the first one)
                this.active_segment_index = this.active_segment_index.wrapping_add(1);

                if this.active_segment_index >= this.buf.segment_count() {
                    let (ptr, cap) = this.buf.grow_one();
                    this.write_ptr = NonNull::new_unchecked(ptr);
                    this.segment_end = NonNull::new_unchecked(ptr.add(cap));
                } else {
                    let ptr = this.buf.segment_ptr(this.active_segment_index);
                    let cap = RawSegmentedVec::<T, A>::segment_capacity(this.active_segment_index);
                    this.write_ptr = NonNull::new_unchecked(ptr);
                    this.segment_end = NonNull::new_unchecked(ptr.add(cap));
                }

                let ptr = this.write_ptr.as_ptr();
                std::ptr::write(ptr, value);
                this.write_ptr = NonNull::new_unchecked(ptr.add(1));
                this.len += 1;
                &mut *ptr
            }
        }

        // Fast path: we have space in the current segment
        if self.write_ptr < self.segment_end {
            unsafe {
                let ptr = self.write_ptr.as_ptr();
                std::ptr::write(ptr, value);
                self.write_ptr = NonNull::new_unchecked(ptr.add(1));
                self.len += 1;
                return &mut *ptr;
            }
        }

        // Slow path: need to allocate or move to next segment
        push_slow(self, value)
    }

    /// Creates a new empty `SegmentedVec` with the given allocator.
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self {
            buf: RawSegmentedVec::new_in(alloc),
            len: 0,
            write_ptr: NonNull::dangling(),
            segment_end: NonNull::dangling(),
            active_segment_index: usize::MAX,
            _marker: PhantomData,
        }
    }

    /// Try to create a new `SegmentedVec` with at least the specified capacity.
    #[inline]
    pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        let buf = RawSegmentedVec::try_with_capacity_in(capacity, alloc)?;

        // Improve initialization optimization by computing fields before struct creation.
        // This avoids writing nulls and then immediately overwriting them.
        let (write_ptr, segment_end) = if capacity > 0 && buf.segment_count() > 0 {
            unsafe {
                let ptr: *mut T = buf.segment_ptr(0);
                (
                    NonNull::new_unchecked(ptr),
                    NonNull::new_unchecked(ptr.add(RawSegmentedVec::<T, A>::segment_capacity(0))),
                )
            }
        } else {
            (NonNull::dangling(), NonNull::dangling())
        };

        Ok(Self {
            buf,
            len: 0,
            write_ptr,
            segment_end,
            active_segment_index: 0,
            _marker: PhantomData,
        })
    }

    /// Returns the total number of elements the vector can hold without allocating.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        // Eagerly advance if at boundary and next segment available
        // Optimization: check if reserve allocates the *exact* next segment we need.
        let next_idx = self.active_segment_index.wrapping_add(1);
        let old_seg_count = self.buf.segment_count();

        // This may return the info for the FIRST allocated segment.
        let new_seg_info = self.buf.reserve(self.len, additional);

        if self.write_ptr == self.segment_end && next_idx < self.buf.segment_count() {
            unsafe {
                self.active_segment_index = next_idx;
                // If the segment we need (next_idx) was just allocated (it WAS old_seg_count),
                // then new_seg_info contains exactly its pointer and capacity.
                let (ptr, cap) = if next_idx == old_seg_count {
                    new_seg_info.expect("Failed to reserve space")
                } else {
                    // It was already allocated (next_idx < old_seg_count), or
                    // we allocated multiple segments and next_idx > old_seg_count (impossible if we go 1 by 1),
                    // or other cases. Fallback to lookup.
                    (
                        self.buf.segment_ptr(next_idx),
                        RawSegmentedVec::<T, A>::segment_capacity(next_idx),
                    )
                };

                self.write_ptr = NonNull::new_unchecked(ptr);
                self.segment_end = NonNull::new_unchecked(ptr.add(cap));
            }
        }
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let next_idx = self.active_segment_index.wrapping_add(1);
        let old_seg_count = self.buf.segment_count();

        let new_seg_info = self.buf.try_reserve(self.len, additional)?;

        // Eagerly advance if at boundary and next segment available
        if self.write_ptr == self.segment_end && next_idx < self.buf.segment_count() {
            unsafe {
                self.active_segment_index = next_idx;
                let (ptr, cap) = if next_idx == old_seg_count {
                    new_seg_info.expect("Failed to reserve space")
                } else {
                    (
                        self.buf.segment_ptr(next_idx),
                        RawSegmentedVec::<T, A>::segment_capacity(next_idx),
                    )
                };

                self.write_ptr = NonNull::new_unchecked(ptr);
                self.segment_end = NonNull::new_unchecked(ptr.add(cap));
            }
        }
        Ok(())
    }

    /// Shrinks the capacity as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.len {
            // Optimization: calculate target segments directly
            let target_segments = if self.len == 0 {
                0
            } else {
                // If the active segment is allocated but empty (write_ptr at start),
                // we can drop it.
                // We use pointer arithmetic to check if write_ptr is at the start of
                // the segment, avoiding the expensive memory load of segment_ptr().
                // Logic: segment_end = start + capacity
                // If write_ptr == start, then write_ptr + capacity == segment_end.
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
                if unsafe { self.write_ptr.as_ptr().add(seg_cap) } == self.segment_end.as_ptr() {
                    self.active_segment_index
                } else {
                    self.active_segment_index.wrapping_add(1)
                }
            };

            unsafe { self.buf.shrink_to_fit_segments(target_segments) };

            // Only update pointers if we dropped the active segment
            // or if we are empty (to reset to initial state)
            if self.len == 0 || self.active_segment_index >= self.buf.segment_count() {
                self.update_write_ptr_for_len(self.len);
            }
        }
    }

    /// Shrinks the capacity with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        if self.capacity() > min_capacity {
            if min_capacity <= self.len {
                // Optimization: if min_capacity <= len, we are effectively just shrinking to fit.
                // We can reuse the optimized logic from shrink_to_fit to avoid expensive
                // capacity calculations.
                let target_segments = if self.len == 0 {
                    0
                } else {
                    let seg_cap =
                        RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
                    if unsafe { self.write_ptr.as_ptr().add(seg_cap) } == self.segment_end.as_ptr()
                    {
                        self.active_segment_index
                    } else {
                        self.active_segment_index + 1
                    }
                };
                unsafe { self.buf.shrink_to_fit_segments(target_segments) };
            } else {
                // Revert to capacity-based shrink if user asks for specific capacity > len
                self.buf.shrink_to_fit(min_capacity);
            }

            // Only update pointers if we dropped the active segment
            // or if we are empty (to reset to initial state)
            if self.len == 0 || self.active_segment_index >= self.buf.segment_count() {
                self.update_write_ptr_for_len(self.len);
            }
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }

        if std::mem::size_of::<T>() == 0 {
            self.len = len;
            return;
        }

        // Drop elements from len to self.len using chunk-based iteration
        // This is more efficient than element-by-element dropping.
        unsafe {
            if std::mem::needs_drop::<T>() {
                // Find the starting segment and offset for `len`
                let (mut seg_idx, mut offset) = RawSegmentedVec::<T, A>::location(len);
                let mut remaining = self.len - len;

                while remaining > 0 {
                    let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
                    let seg_ptr = self.buf.segment_ptr(seg_idx);

                    // Number of elements to drop in this segment
                    let count = std::cmp::min(remaining, seg_cap - offset);

                    // Drop the slice
                    let slice_ptr = seg_ptr.add(offset);
                    std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(slice_ptr, count));

                    remaining -= count;
                    seg_idx += 1;
                    offset = 0; // After the first segment, we always start at offset 0
                }
            }
        }

        self.len = len;
        self.update_write_ptr_for_len(len);
    }

    /// Returns a reference to the allocator.
    #[inline]
    pub fn allocator(&self) -> &A {
        self.buf.allocator()
    }

    /// Sets the length of the vector without running destructors.
    ///
    /// This is an internal method used by iterators and other internal code.
    /// It updates the cached write pointer as well.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity()`.
    /// - The elements at `old_len..new_len` must be initialized if growing.
    /// - The elements at `new_len..old_len` must be properly dropped if shrinking
    ///   (this method does NOT drop them).
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.buf.capacity());
        self.len = new_len;
        self.update_write_ptr_for_len(new_len);
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }

        unsafe {
            let hole = self.buf.ptr_at(index);
            let value = std::ptr::read(hole);

            if index != len - 1 {
                // Pop the last element (uses optimized fast/slow path)
                let last_value = self.pop().unwrap_unchecked();
                // Write it to the hole
                std::ptr::write(hole, last_value);
            } else {
                // The hole IS the last element, just update len and pointers
                // Use the same logic as pop's fast path
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
                let segment_start = self.segment_end.as_ptr().sub(seg_cap);

                self.len -= 1;
                if self.write_ptr.as_ptr() > segment_start {
                    self.write_ptr = NonNull::new_unchecked(self.write_ptr.as_ptr().sub(1));
                } else {
                    // At segment boundary - recalculate pointers
                    self.update_write_ptr_for_len(self.len);
                }
            }

            value
        }
    }

    /// Inserts an element at position `index`.
    #[track_caller]
    pub fn insert(&mut self, index: usize, element: T) {
        let _ = self.insert_mut(index, element);
    }

    /// Inserts an element and returns a mutable reference to it.
    #[inline]
    #[track_caller]
    #[must_use = "if you don't need a reference to the value, use `SegmentedVec::insert` instead"]
    pub fn insert_mut(&mut self, index: usize, element: T) -> &mut T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        let len = self.len();
        if index > len {
            assert_failed(index, len);
        }

        if std::mem::size_of::<T>() == 0 {
            self.len += 1;
            return unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() };
        }

        // Fast path: inserting at the end is just a push
        if index == len {
            return self.push_mut(element);
        }

        // Ensure we have space by pushing a placeholder (will be overwritten)
        // This properly updates len, write_ptr, etc.
        unsafe {
            // Reserve space
            if len == self.buf.capacity() {
                self.reserve(1);
            }
            self.len = len + 1;
            self.update_write_ptr_for_len(self.len);

            // Ripple shift: work forwards from insertion point to end
            let (insert_seg, insert_offset) = RawSegmentedVec::<T, A>::location(index);
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(len);

            if insert_seg == end_seg {
                // Simple case: insertion and end are in the same segment
                let seg_ptr = self.buf.segment_ptr(insert_seg);
                let count = end_offset - insert_offset;
                if count > 0 {
                    std::ptr::copy(
                        seg_ptr.add(insert_offset),
                        seg_ptr.add(insert_offset + 1),
                        count,
                    );
                }
                std::ptr::write(seg_ptr.add(insert_offset), element);
            } else {
                // Multi-segment case: need to ripple through segments (forward)
                let mut carry: Option<T> = Some(element);

                for seg_idx in insert_seg..=end_seg {
                    let seg_ptr = self.buf.segment_ptr(seg_idx);
                    let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);

                    // Determine range to shift in this segment
                    let start_off = if seg_idx == insert_seg {
                        insert_offset
                    } else {
                        0
                    };
                    let end_off = if seg_idx == end_seg {
                        end_offset
                    } else {
                        seg_cap
                    };

                    // Save the last element before it gets overwritten (only if there are more segments)
                    let next_carry = if seg_idx < end_seg {
                        Some(std::ptr::read(seg_ptr.add(seg_cap - 1)))
                    } else {
                        None
                    };

                    // Shift elements right: [start_off..end_off] -> [start_off+1..end_off+1]
                    let count = end_off - start_off;
                    if count > 0 {
                        std::ptr::copy(seg_ptr.add(start_off), seg_ptr.add(start_off + 1), count);
                    }

                    // Write carry to the start position
                    if let Some(c) = carry.take() {
                        std::ptr::write(seg_ptr.add(start_off), c);
                    }

                    // Update carry for next segment
                    carry = next_carry;
                }
            }

            let ptr = self.buf.ptr_at(index);
            &mut *ptr
        }
    }

    /// Removes and returns the element at position `index`.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        match self.try_remove(index) {
            Some(elem) => elem,
            None => assert_failed(index, self.len()),
        }
    }

    /// Tries to remove and return the element at position `index`.
    pub fn try_remove(&mut self, index: usize) -> Option<T> {
        let len = self.len();
        if index >= len {
            return None;
        }

        // Fast path: removing the last element is just a pop
        if index == len - 1 {
            return self.pop();
        }

        if std::mem::size_of::<T>() == 0 {
            self.len -= 1;
            return Some(unsafe { std::mem::zeroed() });
        }

        unsafe {
            let ptr = self.buf.ptr_at(index);
            let value = std::ptr::read(ptr);

            // Ripple shift: work forwards from removal point to end
            let (remove_seg, remove_offset) = RawSegmentedVec::<T, A>::location(index);
            let (end_seg, end_offset) = RawSegmentedVec::<T, A>::location(len - 1);

            if remove_seg == end_seg {
                // Simple case: removal and last element are in the same segment
                let seg_ptr = self.buf.segment_ptr(remove_seg);
                let count = end_offset - remove_offset;
                if count > 0 {
                    std::ptr::copy(
                        seg_ptr.add(remove_offset + 1),
                        seg_ptr.add(remove_offset),
                        count,
                    );
                }
            } else {
                // Multi-segment case: need to ripple through segments (forward)
                for seg_idx in remove_seg..=end_seg {
                    let seg_ptr = self.buf.segment_ptr(seg_idx);
                    let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);

                    // Determine range to shift in this segment
                    let start_off = if seg_idx == remove_seg {
                        remove_offset
                    } else {
                        0
                    };
                    let end_off = if seg_idx == end_seg {
                        end_offset
                    } else {
                        seg_cap - 1
                    };

                    // Shift elements left: [start_off+1..end_off+1] -> [start_off..end_off]
                    let count = end_off - start_off;
                    if count > 0 {
                        std::ptr::copy(seg_ptr.add(start_off + 1), seg_ptr.add(start_off), count);
                    }

                    // If not the last segment, copy first element of next segment to last position
                    if seg_idx < end_seg {
                        let next_seg_ptr = self.buf.segment_ptr(seg_idx + 1);
                        let carry = std::ptr::read(next_seg_ptr);
                        std::ptr::write(seg_ptr.add(seg_cap - 1), carry);
                    }
                }
            }

            self.len = len - 1;
            self.update_write_ptr_for_len(self.len);
            Some(value)
        }
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    /// Retains only the elements specified by the predicate, with mutable access.
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let original_len = self.len();
        if original_len == 0 {
            return;
        }

        unsafe {
            // Segment-based cursors
            let mut read_seg = 0usize;
            let mut read_off = 0usize;
            let mut read_seg_ptr = self.buf.segment_ptr(0);
            let mut read_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(0);

            let mut write_seg = 0usize;
            let mut write_off = 0usize;
            let mut write_seg_ptr = read_seg_ptr;
            let mut write_seg_cap = read_seg_cap;

            let mut retained = 0usize;
            let max_seg = self.active_segment_index;

            for _ in 0..original_len {
                let read_ptr = read_seg_ptr.add(read_off);

                if f(&mut *read_ptr) {
                    // Keep this element
                    if read_seg != write_seg || read_off != write_off {
                        let write_ptr = write_seg_ptr.add(write_off);
                        std::ptr::copy_nonoverlapping(read_ptr, write_ptr, 1);
                    }
                    retained += 1;

                    // Advance write cursor
                    write_off += 1;
                    if write_off >= write_seg_cap {
                        write_seg += 1;
                        write_off = 0;
                        if write_seg <= max_seg {
                            write_seg_ptr = self.buf.segment_ptr(write_seg);
                            write_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(write_seg);
                        }
                    }
                } else {
                    // Drop this element
                    std::ptr::drop_in_place(read_ptr);
                }

                // Advance read cursor
                read_off += 1;
                if read_off >= read_seg_cap {
                    read_seg += 1;
                    read_off = 0;
                    if read_seg <= max_seg {
                        read_seg_ptr = self.buf.segment_ptr(read_seg);
                        read_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(read_seg);
                    }
                }
            }

            self.len = retained;
            self.update_write_ptr_for_len(retained);
        }
    }

    /// Removes consecutive repeated elements using a key function.
    #[inline]
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    /// Removes consecutive repeated elements using an equality function.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return;
        }

        unsafe {
            let max_seg = self.active_segment_index;

            // Read cursor starts at index 1
            let (mut read_seg, mut read_off) = RawSegmentedVec::<T, A>::location(1);
            let mut read_seg_ptr = self.buf.segment_ptr(read_seg);
            let mut read_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(read_seg);

            // Write cursor starts at index 1 (first element is always kept)
            let mut write_seg = read_seg;
            let mut write_off = read_off;
            let mut write_seg_ptr = read_seg_ptr;
            let mut write_seg_cap = read_seg_cap;

            // Prev pointer starts at index 0
            let mut prev_off = 0usize;
            let mut prev_seg_ptr = self.buf.segment_ptr(0);

            let mut retained = 1usize;

            for _ in 1..len {
                let curr_ptr = read_seg_ptr.add(read_off);
                let prev_ptr = prev_seg_ptr.add(prev_off);

                if same_bucket(&mut *curr_ptr, &mut *prev_ptr) {
                    // Duplicate - drop it
                    std::ptr::drop_in_place(curr_ptr);
                } else {
                    // Keep this element
                    if read_seg != write_seg || read_off != write_off {
                        let write_ptr = write_seg_ptr.add(write_off);
                        std::ptr::copy_nonoverlapping(curr_ptr, write_ptr, 1);
                    }

                    // Update prev to point to the newly written element
                    prev_off = write_off;
                    prev_seg_ptr = write_seg_ptr;

                    retained += 1;

                    // Advance write cursor
                    write_off += 1;
                    if write_off >= write_seg_cap {
                        write_seg += 1;
                        write_off = 0;
                        if write_seg <= max_seg {
                            write_seg_ptr = self.buf.segment_ptr(write_seg);
                            write_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(write_seg);
                        }
                    }
                }

                // Advance read cursor
                read_off += 1;
                if read_off >= read_seg_cap {
                    read_seg += 1;
                    read_off = 0;
                    if read_seg <= max_seg {
                        read_seg_ptr = self.buf.segment_ptr(read_seg);
                        read_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(read_seg);
                    }
                }
            }

            self.len = retained;
            self.update_write_ptr_for_len(retained);
        }
    }

    /// Appends an element if there is sufficient capacity.
    #[inline]
    pub fn push_within_capacity(&mut self, value: T) -> Result<&mut T, T> {
        if std::mem::size_of::<T>() == 0 {
            self.len += 1;
            return Ok(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        if self.len == self.buf.capacity() {
            return Err(value);
        }

        // Use push which handles the cached pointers correctly
        Ok(self.push_mut(value))
    }

    /// Removes the last element and returns it, or `None` if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            self.len -= 1;
            return Some(unsafe { std::mem::zeroed() });
        }

        #[cold]
        #[inline(never)]
        fn pop_slow<T, A: Allocator>(this: &mut SegmentedVec<T, A>) -> T {
            // We're at the start of the current segment.
            // The element to pop is in the previous segment.
            debug_assert!(
                this.active_segment_index > 0,
                "pop_slow with segment 0 should be impossible"
            );

            this.active_segment_index -= 1;
            unsafe {
                let ptr = this.buf.segment_ptr(this.active_segment_index);
                let cap = RawSegmentedVec::<T, A>::segment_capacity(this.active_segment_index);
                this.segment_end = NonNull::new_unchecked(ptr.add(cap));
                this.write_ptr = NonNull::new_unchecked(this.segment_end.as_ptr().sub(1));
                this.len -= 1;
                std::ptr::read(this.write_ptr.as_ptr())
            }
        }

        // Compute segment start from segment_end and capacity
        let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
        let segment_start = unsafe { self.segment_end.as_ptr().sub(seg_cap) };

        // Fast path: we're not at the segment boundary
        if self.write_ptr.as_ptr() > segment_start {
            unsafe {
                self.write_ptr = NonNull::new_unchecked(self.write_ptr.as_ptr().sub(1));
                self.len -= 1;
                Some(std::ptr::read(self.write_ptr.as_ptr()))
            }
        } else {
            // Slow path: segment boundary crossing
            Some(pop_slow(self))
        }
    }

    /// Removes the last element if the predicate returns true.
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) {
            self.pop()
        } else {
            None
        }
    }

    /// Returns a wrapper that allows peeking at and potentially popping
    /// the last element.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<PeekMut<'_, T, A>> {
        PeekMut::new(self)
    }

    /// Moves all elements from `other` into `self`.
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        let other_len = other.len();
        if other_len == 0 {
            return;
        }

        self.reserve(other_len);

        unsafe {
            // Chunk-based moving: iterate through other's segments
            let mut remaining = other_len;
            let mut src_seg = 0usize;
            let mut src_off = 0usize;

            while remaining > 0 {
                let src_seg_ptr = other.buf.segment_ptr(src_seg);
                let src_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(src_seg);
                let available_in_src = src_seg_cap - src_off;
                let to_copy_from_src = remaining.min(available_in_src);

                // Copy in chunks to destination
                let mut copied_from_src = 0;
                while copied_from_src < to_copy_from_src {
                    let dst_seg_cap =
                        RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
                    let dst_seg_ptr = self.buf.segment_ptr(self.active_segment_index);
                    let (_, dst_off) = RawSegmentedVec::<T, A>::location(self.len);

                    let available_in_dst = dst_seg_cap - dst_off;
                    let to_copy = (to_copy_from_src - copied_from_src).min(available_in_dst);

                    std::ptr::copy_nonoverlapping(
                        src_seg_ptr.add(src_off + copied_from_src),
                        dst_seg_ptr.add(dst_off),
                        to_copy,
                    );

                    self.len += to_copy;
                    copied_from_src += to_copy;

                    // Update write_ptr and potentially move to next segment
                    if dst_off + to_copy >= dst_seg_cap && self.len < self.buf.capacity() {
                        self.active_segment_index += 1;
                        self.segment_end = NonNull::new_unchecked(
                            self.buf
                                .segment_ptr(self.active_segment_index)
                                .add(RawSegmentedVec::<T, A>::segment_capacity(
                                    self.active_segment_index,
                                )),
                        );
                    }
                }

                remaining -= to_copy_from_src;
                src_seg += 1;
                src_off = 0;
            }

            // Update write_ptr to final position
            self.update_write_ptr_for_len(self.len);
        }

        // Clear other without dropping elements (we moved them)
        other.len = 0;
        other.update_write_ptr_for_len(0);
    }

    /// Removes the subslice indicated by the given range from the vector,
    /// returning a double-ended iterator over the removed subslice.
    ///
    /// If the iterator is dropped before being fully consumed,
    /// it drops the remaining removed elements.
    ///
    /// The returned iterator keeps a mutable borrow on the vector to optimize
    /// its implementation.
    ///
    /// # Panics
    ///
    /// Panics if the range has `start_bound > end_bound`, or, if the range is
    /// bounded on either end and past the length of the vector.
    ///
    /// # Leaking
    ///
    /// If the returned iterator goes out of scope without being dropped (due to
    /// [`mem::forget`], for example), the vector may have lost and leaked
    /// elements arbitrarily, including elements outside the range.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut v: SegmentedVec<i32> = SegmentedVec::new();
    /// v.extend([1, 2, 3]);
    /// let u: Vec<_> = v.drain(1..).collect();
    /// assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1]);
    /// assert_eq!(u, vec![2, 3]);
    ///
    /// // A full range clears the vector, like `clear()` does
    /// v.drain(..);
    /// assert!(v.is_empty());
    /// ```
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, A>
    where
        R: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound;

        let len = self.len();

        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n.checked_add(1).expect("range start overflow"),
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("range end overflow"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        assert!(start <= end, "drain range start ({start}) > end ({end})");
        assert!(end <= len, "drain range end ({end}) > len ({len})");

        Drain {
            vec: std::ptr::NonNull::from(self),
            range_start: start,
            range_end: end,
            index: start,
            original_len: len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Clears the vector, removing all values.
    #[inline]
    pub fn clear(&mut self) {
        let len = self.len;
        if len == 0 {
            return;
        }

        // Chunk-based dropping: iterate through segments
        unsafe {
            let mut remaining = len;
            let mut seg = 0usize;
            let mut off = 0usize;

            while remaining > 0 {
                let seg_ptr = self.buf.segment_ptr(seg);
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                let available = seg_cap - off;
                let to_drop = remaining.min(available);

                // Drop all elements in this chunk at once
                let slice = std::slice::from_raw_parts_mut(seg_ptr.add(off), to_drop);
                std::ptr::drop_in_place(slice);

                remaining -= to_drop;
                seg += 1;
                off = 0;
            }
        }

        self.len = 0;
        self.update_write_ptr_for_len(0);
    }

    /// Returns the number of elements in the vector.
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to an element or subslice depending on the type of index.
    ///
    /// - If given a position, returns a reference to the element at that position or `None` if out of bounds.
    /// - If given a range, returns a `SegmentedSlice` or `None` if out of bounds.
    #[inline]
    pub fn get<I>(&self, index: I) -> Option<I::Output<'_>>
    where
        I: SliceIndex<Self>,
    {
        index.get(self)
    }

    /// Returns a mutable reference to an element or subslice depending on the type of index.
    #[inline]
    pub fn get_mut<I>(&mut self, index: I) -> Option<I::OutputMut<'_>>
    where
        I: SliceIndex<Self>,
    {
        index.get_mut(self)
    }

    /// Returns a reference to the first element, or `None` if empty.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        unsafe { Some(&*self.buf.segment_ptr(0)) }
    }

    /// Returns a mutable reference to the first element, or `None` if empty.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Returns a reference to the last element, or `None` if empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.get(self.len - 1)
        }
    }

    /// Returns a mutable reference to the last element, or `None` if empty.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            self.get_mut(self.len - 1)
        }
    }

    /// Returns an iterator over the vector.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(&self.as_slice())
    }

    /// Returns a mutable iterator over the vector.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(&mut self.as_mut_slice())
    }

    // =========================================================================
    // Convenience methods that delegate to slice operations
    // =========================================================================

    /// Swaps two elements in the vector.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    #[inline]
    #[track_caller]
    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }

    /// Reverses the order of elements in the vector, in place.
    #[inline]
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse();
    }

    /// Rotates the vector in-place such that the first `mid` elements
    /// move to the end.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    #[inline]
    #[track_caller]
    pub fn rotate_left(&mut self, mid: usize) {
        self.as_mut_slice().rotate_left(mid);
    }

    /// Rotates the vector in-place such that the last `k` elements
    /// move to the front.
    ///
    /// # Panics
    ///
    /// Panics if `k > len`.
    #[inline]
    #[track_caller]
    pub fn rotate_right(&mut self, k: usize) {
        self.as_mut_slice().rotate_right(k);
    }

    /// Fills the vector with elements by cloning `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.as_mut_slice().fill(value);
    }

    /// Fills the vector with elements returned by calling a closure repeatedly.
    #[inline]
    pub fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        self.as_mut_slice().fill_with(f);
    }

    /// Returns `true` if the vector contains an element with the given value.
    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    /// Returns `true` if `needle` is a prefix of the vector.
    #[inline]
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().starts_with(needle)
    }

    /// Returns `true` if `needle` is a suffix of the vector.
    #[inline]
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().ends_with(needle)
    }

    /// Binary searches this vector for a given element.
    #[inline]
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.as_slice().binary_search(x)
    }

    /// Binary searches this vector with a comparator function.
    #[inline]
    pub fn binary_search_by<F>(&self, f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> std::cmp::Ordering,
    {
        self.as_slice().binary_search_by(f)
    }

    /// Binary searches this vector with a key extraction function.
    #[inline]
    pub fn binary_search_by_key<B, F>(&self, b: &B, f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.as_slice().binary_search_by_key(b, f)
    }

    /// Sorts the vector in ascending order.
    ///
    /// This sort is stable (i.e., does not reorder equal elements).
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().sort();
    }

    /// Sorts the vector with a comparator function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements).
    #[inline]
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        self.as_mut_slice().sort_by(compare);
    }

    /// Sorts the vector with a key extraction function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements).
    #[inline]
    pub fn sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.as_mut_slice().sort_by_key(f);
    }

    /// Sorts the vector in ascending order **without** preserving the initial order of equal elements.
    ///
    /// This sort is unstable but typically faster than stable sort.
    #[inline]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().sort_unstable();
    }

    /// Sorts the vector with a comparator function, **without** preserving the initial order of equal elements.
    #[inline]
    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        self.as_mut_slice().sort_unstable_by(compare);
    }

    /// Sorts the vector with a key extraction function, **without** preserving the initial order of equal elements.
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.as_mut_slice().sort_unstable_by_key(f);
    }

    /// Returns an iterator over `chunk_size` elements at a time.
    ///
    /// The chunks do not overlap. If `chunk_size` does not divide
    /// the length, the last chunk will be shorter.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    #[inline]
    #[track_caller]
    pub fn chunks(&self, chunk_size: usize) -> slice::Chunks<'_, T> {
        self.as_slice().chunks(chunk_size)
    }

    /// Returns an iterator over overlapping windows of length `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    #[inline]
    #[track_caller]
    pub fn windows(&self, size: usize) -> slice::Windows<'_, T> {
        self.as_slice().windows(size)
    }

    /// Copies elements into a `Vec`.
    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    /// Checks if the vector is sorted in ascending order.
    #[inline]
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.iter().is_sorted()
    }

    /// Checks if the vector is sorted according to the given comparator.
    #[inline]
    pub fn is_sorted_by<F>(&self, mut compare: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        self.iter().is_sorted_by(|a, b| compare(*a, *b))
    }

    /// Checks if the vector is sorted according to the given key extraction function.
    #[inline]
    pub fn is_sorted_by_key<K, F>(&self, f: F) -> bool
    where
        F: FnMut(&T) -> K,
        K: PartialOrd,
    {
        self.iter().is_sorted_by_key(f)
    }

    /// Returns a reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn unchecked_at(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        &*self.buf.ptr_at(index)
    }

    /// Returns a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn unchecked_at_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        &mut *self.buf.ptr_at(index)
    }

    /// Reads and returns an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    /// The caller must ensure the element is not read again without being reinitialized.
    #[inline]
    pub unsafe fn unchecked_read(&self, index: usize) -> T {
        debug_assert!(index < self.len);
        std::ptr::read(self.buf.ptr_at(index))
    }

    /// Sets the length of the vector without running destructors.
    ///
    /// This is an internal method used by iterators and other internal code.
    /// It updates the cached write pointer as well.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity()`.
    /// - The elements at `old_len..new_len` must be initialized if growing.
    /// - The elements at `new_len..old_len` must be properly dropped if shrinking
    ///   (this method does NOT drop them).
    #[inline]
    pub(crate) unsafe fn set_len_internal(&mut self, new_len: usize) {
        self.set_len(new_len)
    }

    /// Splits the collection into two at the given index.
    #[inline]
    #[must_use = "use `.truncate()` if you don't need the other half"]
    #[track_caller]
    pub fn split_off(&mut self, at: usize) -> Self
    where
        A: Clone,
    {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(at: usize, len: usize) -> ! {
            panic!("`at` split index (is {at}) should be <= len (is {len})");
        }

        if at > self.len() {
            assert_failed(at, self.len());
        }

        let other_len = self.len - at;
        if other_len == 0 {
            return Self::new_in(self.allocator().clone());
        }

        let mut other = Self::with_capacity_in(other_len, self.allocator().clone());

        // Chunk-based moving from self[at..] to other
        unsafe {
            let mut remaining = other_len;
            let (mut src_seg, mut src_off) = RawSegmentedVec::<T, A>::location(at);
            let mut dst_seg = 0usize;
            let mut dst_off = 0usize;

            while remaining > 0 {
                let src_seg_ptr = self.buf.segment_ptr(src_seg);
                let src_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(src_seg);
                let available_in_src = src_seg_cap - src_off;
                let to_copy_from_src = remaining.min(available_in_src);

                // Copy chunks to destination
                let mut copied_from_src = 0;
                while copied_from_src < to_copy_from_src {
                    let dst_seg_ptr = other.buf.segment_ptr(dst_seg);
                    let dst_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(dst_seg);
                    let available_in_dst = dst_seg_cap - dst_off;
                    let to_copy = (to_copy_from_src - copied_from_src).min(available_in_dst);

                    std::ptr::copy_nonoverlapping(
                        src_seg_ptr.add(src_off + copied_from_src),
                        dst_seg_ptr.add(dst_off),
                        to_copy,
                    );

                    copied_from_src += to_copy;
                    dst_off += to_copy;

                    if dst_off >= dst_seg_cap {
                        dst_seg += 1;
                        dst_off = 0;
                    }
                }

                remaining -= to_copy_from_src;
                src_seg += 1;
                src_off = 0;
            }

            other.len = other_len;
            other.update_write_ptr_for_len(other_len);
        }

        self.len = at;
        self.update_write_ptr_for_len(at);
        other
    }

    /// Resizes the vector using a closure to generate new values.
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len();
        if new_len > len {
            let to_add = new_len - len;
            self.reserve(to_add);

            // Chunk-based filling: write directly to segments
            unsafe {
                let mut remaining = to_add;
                let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(len);

                while remaining > 0 {
                    let seg_ptr = self.buf.segment_ptr(seg);
                    let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                    let available = seg_cap - off;
                    let to_fill = remaining.min(available);

                    // Fill this chunk
                    for i in 0..to_fill {
                        std::ptr::write(seg_ptr.add(off + i), f());
                    }

                    remaining -= to_fill;
                    seg += 1;
                    off = 0;
                }

                self.len = new_len;
                self.update_write_ptr_for_len(new_len);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Converts a `SegmentedVec<T>` into a `SegmentedVec<U>` with the same layout,
    /// reusing the existing segment allocations.
    ///
    /// This is useful when you want to reuse allocated memory for a different type
    /// that has the same size and alignment.
    ///
    /// # Panics
    ///
    /// Panics if `T` and `U` have different sizes or alignments.
    pub fn recycle<U>(mut self) -> SegmentedVec<U, A>
    where
        U: Sized,
    {
        // Compile-time checks for layout compatibility
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<U>(),
            "Cannot recycle: size_of::<T>() != size_of::<U>()"
        );
        assert!(
            std::mem::align_of::<T>() == std::mem::align_of::<U>(),
            "Cannot recycle: align_of::<T>() != align_of::<U>()"
        );

        // Clear the vector (drops all elements but keeps allocations)
        self.clear();

        // Safety: We've verified T and U have the same layout.
        // The segments are empty after clear(), so we can safely reinterpret them.
        // We use transmute_copy and forget to move ownership without running Drop.
        unsafe {
            let recycled = SegmentedVec {
                buf: std::ptr::read(
                    &self.buf as *const RawSegmentedVec<T, A> as *const RawSegmentedVec<U, A>,
                ),
                len: 0,
                write_ptr: NonNull::dangling(),
                segment_end: NonNull::dangling(),
                active_segment_index: 0,
                _marker: PhantomData,
            };
            std::mem::forget(self);
            recycled
        }
    }

    /// Default extend implementation for iterators without known size.
    /// This is the fallback used by SpecExtend.
    pub(crate) fn extend_desugared<I: Iterator<Item = T>>(&mut self, mut iter: I) {
        let (lower, _) = iter.size_hint();
        self.reserve(lower);

        while let Some(item) = iter.next() {
            if self.write_ptr == self.segment_end {
                if self.len == self.buf.capacity() {
                    self.buf.grow_one();
                }

                if self.len == 0 {
                    self.update_write_ptr_for_len(0);
                } else {
                    self.active_segment_index += 1;
                    // SAFETY: We verified capacity exists, so next segment is allocated.
                    unsafe {
                        let ptr = self.buf.segment_ptr(self.active_segment_index);
                        self.write_ptr = NonNull::new_unchecked(ptr);
                        let cap =
                            RawSegmentedVec::<T, A>::segment_capacity(self.active_segment_index);
                        self.segment_end = NonNull::new_unchecked(ptr.add(cap));
                    }
                }
            }

            // SAFETY: We verified write_ptr < segment_end (or made space)
            unsafe {
                let mut ptr = self.write_ptr.as_ptr();
                let end = self.segment_end.as_ptr();

                // Write the first item verified by the outer loop
                std::ptr::write(ptr, item);
                ptr = ptr.add(1);

                while ptr < end {
                    if let Some(item) = iter.next() {
                        std::ptr::write(ptr, item);
                        ptr = ptr.add(1);
                    } else {
                        // Iterator exhausted
                        let count = ptr.offset_from(self.write_ptr.as_ptr()) as usize;
                        self.len += count;
                        self.write_ptr = NonNull::new_unchecked(ptr);
                        return;
                    }
                }

                // Segment full
                let count = ptr.offset_from(self.write_ptr.as_ptr()) as usize;
                self.len += count;
                self.write_ptr = NonNull::new_unchecked(ptr);
            }
        }
    }
}

impl<T: Clone, A: Allocator> SegmentedVec<T, A> {
    /// Resizes the vector to contain `new_len` elements.
    pub fn resize(&mut self, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            let to_add = new_len - len;
            self.reserve(to_add);

            // Chunk-based filling: write directly to segments
            unsafe {
                let mut remaining = to_add;
                let (mut seg, mut off) = RawSegmentedVec::<T, A>::location(len);

                while remaining > 0 {
                    let seg_ptr = self.buf.segment_ptr(seg);
                    let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                    let available = seg_cap - off;
                    let to_fill = remaining.min(available);

                    // Fill this chunk (clone for all but the last element)
                    if remaining > to_fill || to_fill > 1 {
                        // Not the last chunk or chunk has multiple elements
                        for i in 0..(to_fill - if remaining == to_fill { 1 } else { 0 }) {
                            std::ptr::write(seg_ptr.add(off + i), value.clone());
                        }
                        // Write last element without cloning if this is the final chunk
                        if remaining == to_fill {
                            std::ptr::write(seg_ptr.add(off + to_fill - 1), value);
                            break;
                        }
                    } else {
                        // Single element in final chunk - use value directly
                        std::ptr::write(seg_ptr.add(off), value);
                        break;
                    }

                    remaining -= to_fill;
                    seg += 1;
                    off = 0;
                }

                self.len = new_len;
                self.update_write_ptr_for_len(new_len);
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Extends the vector by cloning elements from a slice.
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }

    /// Extends the vector by cloning elements from within itself.
    pub fn extend_from_within<R>(&mut self, src: R)
    where
        R: std::ops::RangeBounds<usize>,
    {
        let start = match src.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match src.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => self.len(),
        };

        assert!(start <= end && end <= self.len(), "range out of bounds");

        let count = end - start;
        self.reserve(count);

        // Clone elements from the range and push them
        for i in start..end {
            unsafe {
                let src_ptr = self.buf.ptr_at(i);
                let value = (*src_ptr).clone();
                self.push(value);
            }
        }
    }
}

// Note: `into_flattened` is not implemented for SegmentedVec<[T; N], A> because
// segmented storage doesn't have contiguous memory layout that can be reinterpreted.
// Users should iterate and flatten manually if needed.

impl<T: PartialEq, A: Allocator> SegmentedVec<T, A> {
    /// Removes consecutive repeated elements.
    #[inline]
    pub fn dedup(&mut self) {
        self.dedup_by(|a, b| a == b)
    }
}

impl<T: Clone, A: Allocator + Clone> Clone for SegmentedVec<T, A> {
    fn clone(&self) -> Self {
        let len = self.len();
        let mut new_vec = Self::with_capacity_in(len, self.allocator().clone());

        if len == 0 {
            return new_vec;
        }

        // Chunk-based cloning: iterate through source segments
        unsafe {
            let mut remaining = len;
            let mut src_seg = 0usize;
            let mut src_off = 0usize;
            let mut dst_seg = 0usize;
            let mut dst_off = 0usize;

            while remaining > 0 {
                let src_seg_ptr = self.buf.segment_ptr(src_seg);
                let src_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(src_seg);
                let available_in_src = src_seg_cap - src_off;
                let to_clone_from_src = remaining.min(available_in_src);

                // Clone in chunks to destination
                let mut cloned = 0;
                while cloned < to_clone_from_src {
                    let dst_seg_ptr = new_vec.buf.segment_ptr(dst_seg);
                    let dst_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(dst_seg);
                    let available_in_dst = dst_seg_cap - dst_off;
                    let to_clone = (to_clone_from_src - cloned).min(available_in_dst);

                    for i in 0..to_clone {
                        let src_ptr = src_seg_ptr.add(src_off + cloned + i);
                        let dst_ptr = dst_seg_ptr.add(dst_off + i);
                        std::ptr::write(dst_ptr, (*src_ptr).clone());
                    }

                    cloned += to_clone;
                    dst_off += to_clone;

                    if dst_off >= dst_seg_cap {
                        dst_seg += 1;
                        dst_off = 0;
                    }
                }

                remaining -= to_clone_from_src;
                src_seg += 1;
                src_off = 0;
            }

            new_vec.len = len;
            new_vec.update_write_ptr_for_len(len);
        }

        new_vec
    }

    fn clone_from(&mut self, source: &Self) {
        let len = source.len();
        self.clear();
        self.reserve(len);

        if len == 0 {
            return;
        }

        // Chunk-based cloning: iterate through source segments
        unsafe {
            let mut remaining = len;
            let mut src_seg = 0usize;
            let mut src_off = 0usize;
            let mut dst_seg = 0usize;
            let mut dst_off = 0usize;

            while remaining > 0 {
                let src_seg_ptr = source.buf.segment_ptr(src_seg);
                let src_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(src_seg);
                let available_in_src = src_seg_cap - src_off;
                let to_clone_from_src = remaining.min(available_in_src);

                // Clone in chunks to destination
                let mut cloned = 0;
                while cloned < to_clone_from_src {
                    let dst_seg_ptr = self.buf.segment_ptr(dst_seg);
                    let dst_seg_cap = RawSegmentedVec::<T, A>::segment_capacity(dst_seg);
                    let available_in_dst = dst_seg_cap - dst_off;
                    let to_clone = (to_clone_from_src - cloned).min(available_in_dst);

                    for i in 0..to_clone {
                        let src_ptr = src_seg_ptr.add(src_off + cloned + i);
                        let dst_ptr = dst_seg_ptr.add(dst_off + i);
                        std::ptr::write(dst_ptr, (*src_ptr).clone());
                    }

                    cloned += to_clone;
                    dst_off += to_clone;

                    if dst_off >= dst_seg_cap {
                        dst_seg += 1;
                        dst_off = 0;
                    }
                }

                remaining -= to_clone_from_src;
                src_seg += 1;
                src_off = 0;
            }

            self.len = len;
            self.update_write_ptr_for_len(len);
        }
    }
}

impl<T: std::hash::Hash, A: Allocator> std::hash::Hash for SegmentedVec<T, A> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}
impl<T, A: Allocator> Index<usize> for SegmentedVec<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, A: Allocator> IndexMut<usize> for SegmentedVec<T, A> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T> FromIterator<T> for SegmentedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        <Self as SpecFromIter<T, I::IntoIter>>::from_iter(iter.into_iter())
    }
}

// Note: IntoIterator for SegmentedVec<T, A> requires A = Global since IntoIter
// doesn't have an allocator type parameter. This is a limitation of the current
// implementation to keep IntoIter simpler.
impl<T> IntoIterator for SegmentedVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the vector (from start to end). The vector cannot be used after calling
    /// this.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a SegmentedVec<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut SegmentedVec<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, A: Allocator> Extend<T> for SegmentedVec<T, A> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        // Use size_hint check for runtime dispatch to optimized path
        // when the iterator provides an exact count (lower == upper)
        crate::spec_extend::extend_with_size_hint_check(self, iter.into_iter())
    }
}

impl<'a, T: Copy + 'a, A: Allocator> Extend<&'a T> for SegmentedVec<T, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied())
    }
}

impl<T: PartialOrd, A1: Allocator, A2: Allocator> PartialOrd<SegmentedVec<T, A2>>
    for SegmentedVec<T, A1>
{
    #[inline]
    fn partial_cmp(&self, other: &SegmentedVec<T, A2>) -> Option<Ordering> {
        self.as_slice().partial_cmp(&other.as_slice())
    }
}

impl<T: Eq, A: Allocator> Eq for SegmentedVec<T, A> {}

impl<T: Ord, A: Allocator> Ord for SegmentedVec<T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<T, A: Allocator> Drop for SegmentedVec<T, A> {
    fn drop(&mut self) {
        if !std::mem::needs_drop::<T>() {
            return;
        }

        let len = self.len;
        if len == 0 {
            return;
        }

        // Chunk-based dropping: iterate through segments
        unsafe {
            let mut remaining = len;
            let mut seg = 0usize;
            let mut off = 0usize;

            while remaining > 0 {
                let seg_ptr = self.buf.segment_ptr(seg);
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg);
                let available = seg_cap - off;
                let to_drop = remaining.min(available);

                // Drop all elements in this chunk at once
                let slice = std::slice::from_raw_parts_mut(seg_ptr.add(off), to_drop);
                std::ptr::drop_in_place(slice);

                remaining -= to_drop;
                seg += 1;
                off = 0;
            }
        }
        // RawSegmentedVec handles segment deallocation in its own Drop
    }
}

impl<T> Default for SegmentedVec<T> {
    /// Creates an empty `SegmentedVec<T>`.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: std::fmt::Debug, A: Allocator> std::fmt::Debug for SegmentedVec<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.as_slice(), f)
    }
}

impl<T, A: Allocator> AsRef<SegmentedVec<T, A>> for SegmentedVec<T, A> {
    fn as_ref(&self) -> &SegmentedVec<T, A> {
        self
    }
}

impl<T, A: Allocator> AsMut<SegmentedVec<T, A>> for SegmentedVec<T, A> {
    fn as_mut(&mut self) -> &mut SegmentedVec<T, A> {
        self
    }
}

// Note: AsRef<[T]> and AsMut<[T]> are not implemented for SegmentedVec
// because the elements are not stored in contiguous memory.

impl<T: Clone> From<&[T]> for SegmentedVec<T> {
    /// Allocates a `SegmentedVec<T>` and fills it by cloning `s`'s items.
    fn from(s: &[T]) -> Self {
        let len = s.len();
        if len == 0 {
            return Self::new();
        }

        let mut vec = Self::with_capacity(len);

        // Chunk-based cloning
        unsafe {
            let mut src_idx = 0usize;
            let mut dst_seg = 0usize;
            let mut dst_off = 0usize;

            while src_idx < len {
                let dst_seg_ptr = vec.buf.segment_ptr(dst_seg);
                let dst_seg_cap = RawSegmentedVec::<T, Global>::segment_capacity(dst_seg);
                let available_in_dst = dst_seg_cap - dst_off;
                let remaining = len - src_idx;
                let to_clone = remaining.min(available_in_dst);

                for i in 0..to_clone {
                    std::ptr::write(dst_seg_ptr.add(dst_off + i), s[src_idx + i].clone());
                }

                src_idx += to_clone;
                dst_off += to_clone;

                if dst_off >= dst_seg_cap {
                    dst_seg += 1;
                    dst_off = 0;
                }
            }

            vec.len = len;
            vec.update_write_ptr_for_len(len);
        }

        vec
    }
}

impl<T: Clone> From<&mut [T]> for SegmentedVec<T> {
    /// Allocates a `SegmentedVec<T>` and fills it by cloning `s`'s items.
    fn from(s: &mut [T]) -> Self {
        Self::from(&*s)
    }
}

impl<T: Clone, const N: usize> From<&[T; N]> for SegmentedVec<T> {
    /// Allocates a `SegmentedVec<T>` and fills it by cloning `s`'s items.
    fn from(s: &[T; N]) -> Self {
        Self::from(s.as_slice())
    }
}

impl<T: Clone, const N: usize> From<&mut [T; N]> for SegmentedVec<T> {
    /// Allocates a `SegmentedVec<T>` and fills it by cloning `s`'s items.
    fn from(s: &mut [T; N]) -> Self {
        Self::from(&*s)
    }
}

impl<T, const N: usize> From<[T; N]> for SegmentedVec<T> {
    /// Allocates a `SegmentedVec<T>` and moves `s`'s items into it.
    fn from(s: [T; N]) -> Self {
        if N == 0 {
            return Self::new();
        }

        let mut vec = Self::with_capacity(N);

        // Chunk-based moving
        unsafe {
            // Use MaybeUninit to safely read from the array without dropping
            let arr = std::mem::ManuallyDrop::new(s);
            let arr_ptr = arr.as_ptr();

            let mut src_idx = 0usize;
            let mut dst_seg = 0usize;
            let mut dst_off = 0usize;

            while src_idx < N {
                let dst_seg_ptr = vec.buf.segment_ptr(dst_seg);
                let dst_seg_cap = RawSegmentedVec::<T, Global>::segment_capacity(dst_seg);
                let available_in_dst = dst_seg_cap - dst_off;
                let remaining = N - src_idx;
                let to_move = remaining.min(available_in_dst);

                std::ptr::copy_nonoverlapping(
                    arr_ptr.add(src_idx),
                    dst_seg_ptr.add(dst_off),
                    to_move,
                );

                src_idx += to_move;
                dst_off += to_move;

                if dst_off >= dst_seg_cap {
                    dst_seg += 1;
                    dst_off = 0;
                }
            }

            vec.len = N;
            vec.update_write_ptr_for_len(N);
        }

        vec
    }
}

impl<T: PartialEq, A1: Allocator, A2: Allocator> PartialEq<SegmentedVec<T, A2>>
    for SegmentedVec<T, A1>
{
    #[inline]
    fn eq(&self, other: &SegmentedVec<T, A2>) -> bool {
        self.as_slice().eq(&other.as_slice())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let vec: SegmentedVec<i32> = SegmentedVec::new();
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_push_pop() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_get() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(10);
        vec.push(20);
        vec.push(30);
        assert_eq!(vec.get(0), Some(&10));
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(2), Some(&30));
        assert_eq!(vec.get(3), None);
    }

    #[test]
    fn test_index() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(10);
        vec.push(20);
        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        vec[0] = 100;
        assert_eq!(vec[0], 100);
    }

    #[test]
    fn test_stable_pointers() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(1);
        let ptr = &vec[0] as *const i32;

        for i in 2..1000 {
            vec.push(i);
        }

        assert_eq!(unsafe { *ptr }, 1);
    }

    #[test]
    fn test_iter() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        for i in 0..100 {
            vec.push(i);
        }

        let collected: Vec<i32> = vec.iter().copied().collect();
        let expected: Vec<i32> = (0..100).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_iter_mut() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        for i in 0..10 {
            vec.push(i);
        }

        for item in vec.iter_mut() {
            *item *= 2;
        }

        let collected: Vec<i32> = vec.iter().copied().collect();
        let expected: Vec<i32> = (0..10).map(|x| x * 2).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_into_iter() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        for i in 0..10 {
            vec.push(i);
        }

        let collected: Vec<i32> = vec.into_iter().collect();
        let expected: Vec<i32> = (0..10).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_from_iter() {
        let vec: SegmentedVec<i32> = (0..10).collect();
        assert_eq!(vec.len(), 10);
        for i in 0..10 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_extend() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..5);
        vec.extend(5..10);
        assert_eq!(vec.len(), 10);
        for i in 0..10 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_clear() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        vec.clear();
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_truncate() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        vec.truncate(5);
        assert_eq!(vec.len(), 5);
        for i in 0..5 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_clone() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        let vec2 = vec.clone();
        assert_eq!(vec, vec2);
    }

    #[test]
    fn test_first_last() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        assert_eq!(vec.first(), None);
        assert_eq!(vec.last(), None);

        vec.push(1);
        assert_eq!(vec.first(), Some(&1));
        assert_eq!(vec.last(), Some(&1));

        vec.push(2);
        vec.push(3);
        assert_eq!(vec.first(), Some(&1));
        assert_eq!(vec.last(), Some(&3));
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.reserve(1000);
        assert!(vec.capacity() >= 1000);
        vec.push(1);
        vec.shrink_to_fit();
        assert!(vec.capacity() < 1000);
    }

    #[test]
    fn test_drop_elements() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let drop_count = Rc::new(RefCell::new(0));

        struct DropCounter {
            count: Rc<RefCell<i32>>,
        }

        impl Drop for DropCounter {
            fn drop(&mut self) {
                *self.count.borrow_mut() += 1;
            }
        }

        {
            let mut vec: SegmentedVec<DropCounter> = SegmentedVec::new();
            for _ in 0..10 {
                vec.push(DropCounter {
                    count: drop_count.clone(),
                });
            }
        }

        assert_eq!(*drop_count.borrow(), 10);
    }

    // Note: test_sort and test_drain removed - sort() and drain() methods
    // are not implemented in the core SegmentedVec. Users should use
    // the sorting algorithms in the sort module directly, or implement
    // drain via iteration.

    #[test]
    fn test_with_capacity() {
        let vec: SegmentedVec<i32> = SegmentedVec::with_capacity(100);
        assert!(vec.capacity() >= 100);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity_zero() {
        let vec: SegmentedVec<i32> = SegmentedVec::with_capacity(0);
        assert_eq!(vec.capacity(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_with_capacity_small() {
        for cap in 1..20 {
            let vec: SegmentedVec<i32> = SegmentedVec::with_capacity(cap);
            assert!(vec.capacity() >= cap);
            assert!(vec.is_empty());
        }
    }

    #[test]
    fn test_with_capacity_zst() {
        #[derive(Debug, PartialEq, Clone, Copy)]
        struct Zst;
        let vec: SegmentedVec<Zst> = SegmentedVec::with_capacity(100);
        assert!(vec.capacity() >= 100);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_push_after_with_capacity() {
        let mut vec = SegmentedVec::with_capacity(10);
        for i in 0..10 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 10);
        for i in 0..10 {
            assert_eq!(vec[i], i);
        }
    }

    #[test]
    fn test_insert_remove() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..5);
        vec.insert(2, 100);
        assert_eq!(
            vec.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 100, 2, 3, 4]
        );

        let removed = vec.remove(2);
        assert_eq!(removed, 100);
        assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_swap_remove() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..5);

        let removed = vec.swap_remove(1);
        assert_eq!(removed, 1);
        assert_eq!(vec.len(), 4);
        // Last element (4) should be at index 1 now
        assert_eq!(vec[1], 4);
    }
}
