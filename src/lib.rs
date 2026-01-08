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
mod iter;
mod raw_vec;
mod slice;
mod sort;

use allocator_api2::alloc::{Allocator, Global};
pub use drain::Drain;
pub use into_iter::IntoIter;
pub use iter::{Iter, IterMut};
pub use slice::{
    Chunks, ChunksExact, RChunks, SegmentedSlice, SegmentedSliceMut, SliceIter, SliceIterMut,
    Windows,
};

use raw_vec::RawSegmentedVec;
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
#[repr(C)]
pub struct SegmentedVec<T, A: Allocator = Global> {
    /// Low-level segment allocation management
    pub(crate) buf: RawSegmentedVec<T, A>,
    /// Number of initialized elements
    len: usize,
    /// Cached pointer to the next write position (for fast push)
    write_ptr: *mut T,
    /// Pointer to the end of the current segment
    segment_end: *mut T,
    /// Index of the current active segment
    active_segment_index: usize,
    /// Marker for drop check
    _marker: PhantomData<T>,
}

// Core implementation
impl<T> SegmentedVec<T> {
    /// Creates a new empty `SegmentedVec`.
    ///
    /// Does not allocate until elements are pushed.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    /// let vec: SegmentedVec<i32> = SegmentedVec::new();
    /// assert!(vec.is_empty());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: RawSegmentedVec::new(),
            len: 0,
            write_ptr: std::ptr::null_mut(),
            segment_end: std::ptr::null_mut(),
            active_segment_index: usize::MAX,
            _marker: PhantomData,
        }
    }

    /// Creates a new `SegmentedVec` with at least the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    /// let vec: SegmentedVec<i32> = SegmentedVec::with_capacity(100);
    /// assert!(vec.capacity() >= 100);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
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

    /// Returns the current capacity of the vector.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.len(), 2);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        // Fast path: we have space in the current segment
        if self.write_ptr < self.segment_end {
            unsafe {
                std::ptr::write(self.write_ptr, value);
                self.write_ptr = self.write_ptr.add(1);
            }
            self.len += 1;
            return;
        }

        // Slow path: need to grow or move to next segment
        self.push_slow(value);
    }

    #[cold]
    #[inline(never)]
    fn push_slow(&mut self, value: T) {
        // For ZSTs, we only need one segment ever
        if std::mem::size_of::<T>() == 0 {
            if self.buf.segment_count() == 0 {
                self.buf.grow_one();
                self.active_segment_index = 0;
            }
            // For ZST, write to dangling pointer (no-op for memory, but consumes value)
            unsafe {
                std::ptr::write(std::ptr::NonNull::dangling().as_ptr(), value);
            }
            self.len += 1;
            return;
        }

        self.active_segment_index = self.active_segment_index.wrapping_add(1);

        if self.active_segment_index >= self.buf.segment_count() {
            self.buf.grow_one();
        }

        let idx = self.active_segment_index;
        let base = unsafe { self.buf.segment_ptr(idx) };
        let segment_size = RawSegmentedVec::<T>::segment_capacity(idx);

        unsafe {
            std::ptr::write(base, value);
            self.write_ptr = base.add(1);
            self.segment_end = base.add(segment_size);
        }
        self.len += 1;
    }

    /// Removes the last element from the vector and returns it, or `None` if empty.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.push(1);
    /// vec.push(2);
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(vec.pop(), Some(1));
    /// assert_eq!(vec.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        // For ZSTs, just decrement len and return a new instance
        if std::mem::size_of::<T>() == 0 {
            self.len -= 1;
            // Read from dangling pointer - no-op for ZST, creates value from nothing
            return Some(unsafe { std::ptr::read(std::ptr::NonNull::dangling().as_ptr()) });
        }

        let segment_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };
        if self.write_ptr > segment_base {
            self.len -= 1;
            unsafe {
                self.write_ptr = self.write_ptr.sub(1);
                Some(std::ptr::read(self.write_ptr))
            }
        } else {
            self.pop_slow_path()
        }
    }

    #[cold]
    #[inline(never)]
    fn pop_slow_path(&mut self) -> Option<T> {
        self.active_segment_index -= 1;
        let idx = self.active_segment_index;

        let base = unsafe { self.buf.segment_ptr(idx) };
        let capacity = RawSegmentedVec::<T>::segment_capacity(idx);

        self.segment_end = unsafe { base.add(capacity) };
        self.write_ptr = unsafe { self.segment_end.sub(1) };
        self.len -= 1;
        Some(unsafe { std::ptr::read(self.write_ptr) })
    }

    /// Returns a reference to the element at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(unsafe { self.unchecked_at(index) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(unsafe { self.unchecked_at_mut(index) })
        } else {
            None
        }
    }

    /// Returns a reference to the first element, or `None` if empty.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(unsafe { &*self.buf.segment_ptr(0) })
        }
    }

    /// Returns a mutable reference to the first element, or `None` if empty.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            Some(unsafe { &mut *self.buf.segment_ptr(0) })
        }
    }

    /// Returns a reference to the last element, or `None` if empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            return Some(unsafe { &*std::ptr::NonNull::dangling().as_ptr() });
        }

        let segment_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };

        if self.write_ptr > segment_base {
            Some(unsafe { &*self.write_ptr.sub(1) })
        } else {
            // Cold path: write_ptr is at the start of the active segment,
            // so the last element is in the previous (fully populated) segment.
            let prev_segment_index = self.active_segment_index - 1;
            let prev_cap = RawSegmentedVec::<T>::segment_capacity(prev_segment_index);

            unsafe {
                let prev_segment_base = self.buf.segment_ptr(prev_segment_index);
                Some(&*prev_segment_base.add(prev_cap - 1))
            }
        }
    }

    /// Returns a mutable reference to the last element, or `None` if empty.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }

        if std::mem::size_of::<T>() == 0 {
            return Some(unsafe { &mut *std::ptr::NonNull::dangling().as_ptr() });
        }

        let segment_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };

        if self.write_ptr > segment_base {
            Some(unsafe { &mut *self.write_ptr.sub(1) })
        } else {
            // Cold path: write_ptr is at the start of the active segment,
            // so the last element is in the previous (fully populated) segment.
            let prev_segment_index = self.active_segment_index - 1;
            let prev_cap = RawSegmentedVec::<T>::segment_capacity(prev_segment_index);

            unsafe {
                let prev_segment_base = self.buf.segment_ptr(prev_segment_index);
                Some(&mut *prev_segment_base.add(prev_cap - 1))
            }
        }
    }

    /// Returns `true` if the slice contains an element with the given value.
    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        if self.len == 0 {
            return false;
        }

        // Check active segment first (largest, benefits most from memchr)
        let active_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };
        let active_len = unsafe { self.write_ptr.offset_from(active_base) as usize };
        if active_len > 0 {
            let slice = unsafe { std::slice::from_raw_parts(active_base, active_len) };
            if slice.contains(x) {
                return true;
            }
        }

        // Check previous segments in reverse order (all fully populated)
        let mut segment_idx = self.active_segment_index;
        while segment_idx > 0 {
            segment_idx -= 1;
            let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
            let base = unsafe { self.buf.segment_ptr(segment_idx) };
            let slice = unsafe { std::slice::from_raw_parts(base, segment_cap) };
            if slice.contains(x) {
                return true;
            }
        }

        false
    }

    /// Clears the vector, removing all elements.
    ///
    /// This drops all elements but keeps the allocated memory.
    pub fn clear(&mut self) {
        let old_len = self.len;
        if old_len == 0 {
            return;
        }

        // Reset len BEFORE dropping to prevent double-free if drop panics
        self.len = 0;

        // Reset write_ptr to segment 0 (keep capacity usable)
        if self.buf.segment_count() > 0 {
            let base = unsafe { self.buf.segment_ptr(0) };
            self.write_ptr = base;
            self.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
            self.active_segment_index = 0;
        }

        // Drop all elements
        if std::mem::needs_drop::<T>() {
            let mut remaining = old_len;
            let mut segment_idx = 0;

            while remaining > 0 {
                let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
                let segment_len = segment_cap.min(remaining);
                let base = unsafe { self.buf.segment_ptr(segment_idx) };

                unsafe {
                    std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(base, segment_len));
                }

                segment_idx += 1;
                remaining -= segment_len;
            }
        }
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// Uses chunk-based dropping for better performance by dropping entire
    /// segments at once rather than element by element.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }

        let old_len = self.len;

        // Update state BEFORE dropping to prevent double-free if drop panics
        self.len = len;
        if len == 0 && self.buf.segment_count() > 0 {
            let base = unsafe { self.buf.segment_ptr(0) };
            self.write_ptr = base;
            self.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
            self.active_segment_index = 0;
        } else if len > 0 {
            self.update_write_ptr_for_len();
        }

        // Drop elements beyond new length using chunk-based dropping
        if std::mem::needs_drop::<T>() {
            // Handle ZST - nothing to drop in terms of memory
            if std::mem::size_of::<T>() == 0 {
                for _ in len..old_len {
                    unsafe {
                        std::ptr::drop_in_place(NonNull::<T>::dangling().as_ptr());
                    }
                }
                return;
            }

            // Find segment and offset for new length
            let (start_seg, start_offset) = if len == 0 {
                (0, 0)
            } else {
                let (seg, off) = RawSegmentedVec::<T>::location(len - 1);
                // Start dropping from the next element
                let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg);
                if off + 1 >= seg_cap {
                    (seg + 1, 0)
                } else {
                    (seg, off + 1)
                }
            };

            // Find segment and offset for old length
            let (end_seg, end_offset) = RawSegmentedVec::<T>::location(old_len - 1);
            let end_offset = end_offset + 1; // Convert to exclusive end

            if start_seg == end_seg {
                // All elements to drop are in the same segment
                if start_offset < end_offset {
                    let base = unsafe { self.buf.segment_ptr(start_seg) };
                    unsafe {
                        let ptr = base.add(start_offset);
                        std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                            ptr,
                            end_offset - start_offset,
                        ));
                    }
                }
            } else {
                // Drop partial first segment (from start_offset to end of segment)
                let first_seg_cap = RawSegmentedVec::<T>::segment_capacity(start_seg);
                if start_offset < first_seg_cap {
                    let base = unsafe { self.buf.segment_ptr(start_seg) };
                    unsafe {
                        let ptr = base.add(start_offset);
                        std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                            ptr,
                            first_seg_cap - start_offset,
                        ));
                    }
                }

                // Drop full middle segments
                for seg_idx in (start_seg + 1)..end_seg {
                    let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
                    let base = unsafe { self.buf.segment_ptr(seg_idx) };
                    unsafe {
                        std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(base, seg_cap));
                    }
                }

                // Drop partial last segment (from 0 to end_offset)
                if end_offset > 0 {
                    let base = unsafe { self.buf.segment_ptr(end_seg) };
                    unsafe {
                        std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                            base, end_offset,
                        ));
                    }
                }
            }
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.buf.reserve(self.len + additional);
        self.init_write_ptr_if_needed();
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    ///
    /// Returns `Ok(())` on success, or `Err(TryReserveError)` if allocation fails.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let new_capacity = self
            .len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;
        self.buf.try_reserve(new_capacity)?;
        self.init_write_ptr_if_needed();
        Ok(())
    }

    /// Shrinks the capacity to match the current length.
    pub fn shrink_to_fit(&mut self) {
        self.buf.shrink_to(self.len);
    }

    /// Shrinks the capacity with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.buf.shrink_to(min_capacity.max(self.len));
    }

    /// Initialize write pointer if we didn't have capacity before,
    /// or advance to next segment if at segment boundary.
    fn init_write_ptr_if_needed(&mut self) {
        if self.write_ptr.is_null() && self.buf.segment_count() > 0 {
            // First allocation
            unsafe {
                let base = self.buf.segment_ptr(0);
                self.write_ptr = base;
                self.segment_end = base.add(RawSegmentedVec::<T>::segment_capacity(0));
                self.active_segment_index = 0;
            }
        } else if self.write_ptr == self.segment_end {
            // At segment boundary, advance to next segment if available
            let next_seg = self.active_segment_index.wrapping_add(1);
            if next_seg < self.buf.segment_count() {
                unsafe {
                    let base = self.buf.segment_ptr(next_seg);
                    self.write_ptr = base;
                    self.segment_end = base.add(RawSegmentedVec::<T>::segment_capacity(next_seg));
                    self.active_segment_index = next_seg;
                }
            }
        }
    }

    /// Updates write_ptr based on current len.
    pub(crate) fn update_write_ptr_for_len(&mut self) {
        if self.len == 0 {
            if self.buf.segment_count() > 0 {
                let base = unsafe { self.buf.segment_ptr(0) };
                self.write_ptr = base;
                self.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
                self.active_segment_index = 0;
            }
            return;
        }
        let (segment, offset) = RawSegmentedVec::<T>::location(self.len - 1);
        let segment_size = RawSegmentedVec::<T>::segment_capacity(segment);
        unsafe {
            let base = self.buf.segment_ptr(segment);
            self.active_segment_index = segment;
            self.segment_end = base.add(segment_size);
            self.write_ptr = base.add(offset + 1);
        }
    }

    /// Decrements write_ptr by 1, handling segment boundary.
    #[inline]
    fn decrement_write_ptr(&mut self) {
        let active_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };
        if self.write_ptr == active_base && self.active_segment_index > 0 {
            // Move back to previous segment
            self.active_segment_index -= 1;
            let prev_cap = RawSegmentedVec::<T>::segment_capacity(self.active_segment_index);
            let prev_base = unsafe { self.buf.segment_ptr(self.active_segment_index) };
            self.write_ptr = unsafe { prev_base.add(prev_cap) };
            self.segment_end = self.write_ptr;
        }
        self.write_ptr = unsafe { self.write_ptr.sub(1) };
    }

    /// Sets the length without any checks.
    pub(crate) fn set_len_internal(&mut self, new_len: usize) {
        self.len = new_len;
    }

    /// Returns an iterator over references to the elements.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            vec: self,
            ptr: std::ptr::null(),
            segment_end: std::ptr::null(),
            index: 0,
            segment_index: 0,
        }
    }

    /// Returns an iterator over mutable references to the elements.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            vec: self,
            ptr: std::ptr::null_mut(),
            segment_end: std::ptr::null_mut(),
            index: 0,
            segment_index: 0,
        }
    }

    /// Returns a segmented slice of the entire vector.
    pub fn as_slice(&self) -> SegmentedSlice<'_, T> {
        SegmentedSlice::new(self)
    }

    /// Returns a mutable segmented slice of the entire vector.
    pub fn as_mut_slice(&mut self) -> SegmentedSliceMut<'_, T> {
        SegmentedSliceMut::new(self)
    }

    /// Get an unchecked reference to an element.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn unchecked_at(&self, index: usize) -> &T {
        &*self.buf.ptr_at(index)
    }

    /// Get an unchecked mutable reference to an element.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn unchecked_at_mut(&mut self, index: usize) -> &mut T {
        &mut *self.buf.ptr_at(index)
    }

    /// Read a value from an index without bounds checking.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn unchecked_read(&self, index: usize) -> T {
        std::ptr::read(self.buf.ptr_at(index))
    }

    /// Swaps two elements in the vector.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        assert!(a < self.len && b < self.len, "index out of bounds");
        unsafe {
            let ptr_a = self.buf.ptr_at(a);
            let ptr_b = self.buf.ptr_at(b);
            std::ptr::swap(ptr_a, ptr_b);
        }
    }

    /// Reverses the order of elements in the vector.
    pub fn reverse(&mut self) {
        if self.len < 2 || std::mem::size_of::<T>() == 0 {
            return;
        }

        let mut remaining_total = self.len;

        // Front cursor
        let mut f_segment_index = 0;
        let mut f_segment_capacity = RawSegmentedVec::<T>::MIN_SEGMENT_CAP;
        let mut f_segment_base = unsafe { self.buf.segment_ptr(0) };
        let mut f_remaining = f_segment_capacity;

        // Back cursor
        let mut b_segment_index = self.active_segment_index;
        let mut b_segment_capacity = RawSegmentedVec::<T>::segment_capacity(b_segment_index);
        let mut b_segment_base = unsafe { self.buf.segment_ptr(b_segment_index) };
        let mut b_remaining = unsafe { self.write_ptr.offset_from(b_segment_base) as usize };
        let mut b_segment_end = unsafe { self.write_ptr.sub(1) };

        loop {
            // Front and back in same segment: use slice.reverse
            if f_segment_index == b_segment_index {
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(f_segment_base, remaining_total);
                    slice.reverse();
                }
                return;
            }

            let count = f_remaining.min(b_remaining);

            // Swap chunks in reverse order (LLVM will vectorize this)
            for i in 0..count {
                unsafe {
                    std::ptr::swap(f_segment_base.add(i), b_segment_end.sub(i));
                }
            }

            unsafe {
                f_segment_base = f_segment_base.add(count);
                b_segment_end = b_segment_end.sub(count);
            }

            remaining_total -= count * 2;
            if remaining_total == 0 {
                return;
            }

            // Update front cursor
            f_remaining -= count;
            if f_remaining == 0 {
                f_segment_index += 1;
                f_segment_capacity <<= 1;
                f_remaining = f_segment_capacity;
                f_segment_base = unsafe { self.buf.segment_ptr(f_segment_index) };
            }

            // Update back cursor
            b_remaining -= count;
            if b_remaining == 0 {
                b_segment_index -= 1;
                b_segment_capacity >>= 1;
                b_remaining = b_segment_capacity;
                unsafe {
                    b_segment_base = self.buf.segment_ptr(b_segment_index);
                    b_segment_end = b_segment_base.add(b_segment_capacity - 1);
                }
            }
        }
    }

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len);

        if std::mem::size_of::<T>() == 0 {
            self.len += 1;
            unsafe {
                std::ptr::write(std::ptr::NonNull::dangling().as_ptr(), element);
            }
            return;
        }

        if index == self.len {
            self.push(element);
            return;
        }

        // Ensure capacity (also advances write_ptr to next segment if at boundary)
        if self.write_ptr == self.segment_end {
            self.reserve(1);
        }

        // Find segment containing index
        let (mut seg_idx, offset) = RawSegmentedVec::<T>::location(index);
        let mut seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
        let mut seg_base = unsafe { self.buf.segment_ptr(seg_idx) };

        // Calculate how many elements are in this segment after `offset`
        let active_seg = self.active_segment_index;
        let seg_end = if seg_idx == active_seg {
            unsafe { self.write_ptr.offset_from(seg_base) as usize }
        } else {
            seg_cap
        };

        // Save last element of current segment (will ripple forward)
        let mut carry = unsafe { std::ptr::read(seg_base.add(seg_end - 1)) };

        // Shift elements [offset, seg_end-1) right by 1
        if seg_end - 1 > offset {
            unsafe {
                std::ptr::copy(
                    seg_base.add(offset),
                    seg_base.add(offset + 1),
                    seg_end - 1 - offset,
                );
            }
        }

        // Write new element at insertion point
        unsafe {
            std::ptr::write(seg_base.add(offset), element);
        }

        // Ripple through subsequent full segments (not including active)
        while seg_idx + 1 < active_seg {
            seg_idx += 1;
            seg_cap <<= 1;
            seg_base = unsafe { self.buf.segment_ptr(seg_idx) };

            // Save last element
            let next_carry = unsafe { std::ptr::read(seg_base.add(seg_cap - 1)) };

            // Shift all elements right by 1
            unsafe {
                std::ptr::copy(seg_base, seg_base.add(1), seg_cap - 1);
            }

            // Place carried element at start
            unsafe {
                std::ptr::write(seg_base, carry);
            }

            carry = next_carry;
        }

        // Handle active segment (if we haven't already processed it as the initial segment)
        if seg_idx < active_seg {
            let active_base = unsafe { self.buf.segment_ptr(active_seg) };
            let active_len = unsafe { self.write_ptr.offset_from(active_base) as usize };

            if active_len > 0 {
                // Shift active segment elements right by 1
                unsafe {
                    std::ptr::copy(active_base, active_base.add(1), active_len);
                }
            }
            // Place carry at start (or as only element if empty)
            unsafe {
                std::ptr::write(active_base, carry);
            }
        } else {
            // We inserted into the active segment, write carry at write_ptr
            unsafe {
                std::ptr::write(self.write_ptr, carry);
            }
        }

        self.len += 1;
        self.write_ptr = unsafe { self.write_ptr.add(1) };
    }

    /// Removes and returns the element at position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);

        if std::mem::size_of::<T>() == 0 {
            self.len -= 1;
            return unsafe { std::ptr::read(std::ptr::NonNull::dangling().as_ptr()) };
        }

        // Removing last element is just pop
        if index == self.len - 1 {
            return unsafe { self.pop().unwrap_unchecked() };
        }

        // Find segment containing index
        let (mut seg_idx, offset) = RawSegmentedVec::<T>::location(index);
        let mut seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
        let mut seg_base = unsafe { self.buf.segment_ptr(seg_idx) };

        // Save the element to remove
        let removed = unsafe { std::ptr::read(seg_base.add(offset)) };

        let active_seg = self.active_segment_index;

        // Calculate segment end for initial segment
        let seg_end = if seg_idx == active_seg {
            unsafe { self.write_ptr.offset_from(seg_base) as usize }
        } else {
            seg_cap
        };

        // Shift elements [offset+1, seg_end) left by 1 in initial segment
        if offset + 1 < seg_end {
            unsafe {
                std::ptr::copy(
                    seg_base.add(offset + 1),
                    seg_base.add(offset),
                    seg_end - offset - 1,
                );
            }
        }

        // Ripple through subsequent segments
        while seg_idx < active_seg {
            let next_seg_idx = seg_idx + 1;
            let next_seg_cap = seg_cap << 1;
            let next_seg_base = unsafe { self.buf.segment_ptr(next_seg_idx) };

            let next_seg_len = if next_seg_idx == active_seg {
                unsafe { self.write_ptr.offset_from(next_seg_base) as usize }
            } else {
                next_seg_cap
            };

            // If next segment is empty, stop rippling
            if next_seg_len == 0 {
                break;
            }

            // Carry first element of next segment to last position of current segment
            let carry = unsafe { std::ptr::read(next_seg_base) };
            unsafe {
                std::ptr::write(seg_base.add(seg_cap - 1), carry);
            }

            // Shift next segment elements left by 1
            if next_seg_len > 1 {
                unsafe {
                    std::ptr::copy(next_seg_base.add(1), next_seg_base, next_seg_len - 1);
                }
            }

            seg_idx = next_seg_idx;
            seg_cap = next_seg_cap;
            seg_base = next_seg_base;
        }

        // Update length and write_ptr
        self.len -= 1;
        self.decrement_write_ptr();

        removed
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    /// This does not preserve ordering, but is O(1).
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        if index == self.len - 1 {
            return unsafe { self.pop().unwrap_unchecked() };
        }

        unsafe {
            let ptr_idx = self.unchecked_at_mut(index) as *mut T;
            let value = std::ptr::read(ptr_idx);

            // Write last element to the removed position
            let last_val = self.pop().unwrap_unchecked();
            std::ptr::write(ptr_idx, last_val);

            value
        }
    }

    /// Creates a draining iterator that removes the specified range.
    pub fn drain(&mut self, range: std::ops::Range<usize>) -> Drain<'_, T> {
        assert!(range.start <= range.end);
        assert!(range.end <= self.len);

        let original_len = self.len;

        Drain {
            vec: unsafe { NonNull::new_unchecked(self) },
            range_start: range.start,
            range_end: range.end,
            index: range.start,
            original_len,
            _marker: PhantomData,
        }
    }

    /// Converts the vector into a standard `Vec`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Fills the vector with elements by cloning `value`.
    ///
    /// Optimized based on type characteristics:
    /// - For Copy/POD types: uses `slice.fill()` which optimizes to memset/SIMD
    /// - For complex types: uses `clone_from` to reuse existing heap allocations
    /// - The last element receives the moved value, saving one clone
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        if self.len == 0 {
            return;
        }

        // For types without Drop (Copy/POD), slice.fill is optimal
        // as it can be vectorized to memset/SIMD
        if !std::mem::needs_drop::<T>() {
            let mut remaining = self.len;
            let mut segment_idx = 0;

            while remaining > 0 {
                let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
                let segment_len = segment_cap.min(remaining);
                let base = unsafe { self.buf.segment_ptr(segment_idx) };
                let slice = unsafe { std::slice::from_raw_parts_mut(base, segment_len) };
                slice.fill(value.clone());
                segment_idx += 1;
                remaining -= segment_len;
            }
            return;
        }

        // For complex types with Drop, use clone_from to reuse heap allocations
        // and move the value into the last position to save one clone
        let last_idx = self.len - 1;
        let mut remaining = self.len;
        let mut segment_idx = 0;
        let mut global_idx = 0;

        while remaining > 0 {
            let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
            let segment_len = segment_cap.min(remaining);
            let base = unsafe { self.buf.segment_ptr(segment_idx) };

            for i in 0..segment_len {
                let current_idx = global_idx + i;
                if current_idx == last_idx {
                    // Move value into the last position, saving one clone
                    unsafe {
                        // Drop existing element and write the moved value
                        std::ptr::drop_in_place(base.add(i));
                        std::ptr::write(base.add(i), value);
                    }
                    return;
                }
                // Use clone_from to potentially reuse existing allocations
                unsafe {
                    (*base.add(i)).clone_from(&value);
                }
            }

            global_idx += segment_len;
            segment_idx += 1;
            remaining -= segment_len;
        }
    }

    /// Fills the vector with elements returned by calling a closure.
    ///
    /// Uses chunk-based processing for better performance:
    /// - For Copy/POD types: writes directly without dropping
    /// - For complex types: drops chunk first, then writes new values
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        if self.len == 0 {
            return;
        }

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            for _ in 0..self.len {
                unsafe {
                    std::ptr::drop_in_place(NonNull::<T>::dangling().as_ptr());
                    std::ptr::write(NonNull::<T>::dangling().as_ptr(), f());
                }
            }
            return;
        }

        let mut remaining = self.len;
        let mut segment_idx = 0;

        // For types without Drop, we can skip the drop step entirely
        if !std::mem::needs_drop::<T>() {
            while remaining > 0 {
                let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
                let segment_len = segment_cap.min(remaining);
                let base = unsafe { self.buf.segment_ptr(segment_idx) };

                for i in 0..segment_len {
                    unsafe {
                        std::ptr::write(base.add(i), f());
                    }
                }

                segment_idx += 1;
                remaining -= segment_len;
            }
            return;
        }

        // For types with Drop: drop the chunk first, then write new values
        // This improves cache locality by processing each segment twice
        // rather than interleaving drop/write for each element
        while remaining > 0 {
            let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);
            let segment_len = segment_cap.min(remaining);
            let base = unsafe { self.buf.segment_ptr(segment_idx) };

            // Drop all elements in this segment chunk
            unsafe {
                std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(base, segment_len));
            }

            // Write new values to this segment chunk
            for i in 0..segment_len {
                unsafe {
                    std::ptr::write(base.add(i), f());
                }
            }

            segment_idx += 1;
            remaining -= segment_len;
        }
    }

    /// Resizes the vector to `new_len` elements.
    ///
    /// Uses chunk-based filling for better performance:
    /// - For Copy/POD types: uses `slice.fill()` which optimizes to memset/SIMD
    /// - For complex types: writes clones directly to uninitialized memory
    /// - The last element receives the moved value, saving one clone
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len <= self.len {
            self.truncate(new_len);
            return;
        }

        let old_len = self.len;
        let additional = new_len - old_len;
        self.reserve(additional);

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            for _ in 0..additional {
                unsafe {
                    std::ptr::write(NonNull::<T>::dangling().as_ptr(), value.clone());
                }
            }
            self.len = new_len;
            return;
        }

        // For types without Drop (Copy/POD), use slice.fill for SIMD optimization
        if !std::mem::needs_drop::<T>() {
            // Find starting segment and offset
            let (mut seg_idx, mut offset) = if old_len == 0 {
                (0, 0)
            } else {
                let (seg, off) = RawSegmentedVec::<T>::location(old_len - 1);
                let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg);
                if off + 1 >= seg_cap {
                    (seg + 1, 0)
                } else {
                    (seg, off + 1)
                }
            };

            let mut remaining = additional;

            while remaining > 0 {
                let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
                let available = seg_cap - offset;
                let to_fill = remaining.min(available);
                let base = unsafe { self.buf.segment_ptr(seg_idx) };

                unsafe {
                    let slice = std::slice::from_raw_parts_mut(base.add(offset), to_fill);
                    slice.fill(value.clone());
                }

                remaining -= to_fill;
                seg_idx += 1;
                offset = 0;
            }

            self.len = new_len;
            self.update_write_ptr_for_len();
            return;
        }

        // For complex types with Drop, write clones individually
        // Move value into the last position to save one clone
        let (mut seg_idx, mut offset) = if old_len == 0 {
            (0, 0)
        } else {
            let (seg, off) = RawSegmentedVec::<T>::location(old_len - 1);
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg);
            if off + 1 >= seg_cap {
                (seg + 1, 0)
            } else {
                (seg, off + 1)
            }
        };

        let mut remaining = additional;

        while remaining > 1 {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let available = seg_cap - offset;
            let to_fill = (remaining - 1).min(available); // Reserve 1 for the moved value
            let base = unsafe { self.buf.segment_ptr(seg_idx) };

            for i in 0..to_fill {
                unsafe {
                    std::ptr::write(base.add(offset + i), value.clone());
                }
            }

            remaining -= to_fill;
            offset += to_fill;
            if offset >= seg_cap {
                seg_idx += 1;
                offset = 0;
            }
        }

        // Write the last element by moving the value (saves one clone)
        if remaining == 1 {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            if offset >= seg_cap {
                seg_idx += 1;
                offset = 0;
            }
            let base = unsafe { self.buf.segment_ptr(seg_idx) };
            unsafe {
                std::ptr::write(base.add(offset), value);
            }
        }

        self.len = new_len;
        self.update_write_ptr_for_len();
    }

    /// Resizes the vector using a closure to generate new elements.
    ///
    /// Uses chunk-based filling for better performance by writing
    /// directly to segments rather than using push in a loop.
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        if new_len <= self.len {
            self.truncate(new_len);
            return;
        }

        let old_len = self.len;
        let additional = new_len - old_len;
        self.reserve(additional);

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            for _ in 0..additional {
                unsafe {
                    std::ptr::write(NonNull::<T>::dangling().as_ptr(), f());
                }
            }
            self.len = new_len;
            return;
        }

        // Find starting segment and offset
        let (mut seg_idx, mut offset) = if old_len == 0 {
            (0, 0)
        } else {
            let (seg, off) = RawSegmentedVec::<T>::location(old_len - 1);
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg);
            if off + 1 >= seg_cap {
                (seg + 1, 0)
            } else {
                (seg, off + 1)
            }
        };

        let mut remaining = additional;

        // Write new elements directly to segments
        while remaining > 0 {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let available = seg_cap - offset;
            let to_fill = remaining.min(available);
            let base = unsafe { self.buf.segment_ptr(seg_idx) };

            for i in 0..to_fill {
                unsafe {
                    std::ptr::write(base.add(offset + i), f());
                }
            }

            remaining -= to_fill;
            seg_idx += 1;
            offset = 0;
        }

        self.len = new_len;
        self.update_write_ptr_for_len();
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|elem| f(elem));
    }

    /// Retains only the elements specified by the predicate, with mutable access.
    ///
    /// Uses two independent segment cursors (read and write) that traverse
    /// linearly, avoiding per-element segment/offset calculations.
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        if self.len == 0 {
            return;
        }

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            let mut write_idx = 0;
            for _ in 0..self.len {
                let keep = unsafe { f(&mut *NonNull::<T>::dangling().as_ptr()) };
                if keep {
                    write_idx += 1;
                }
            }
            self.len = write_idx;
            return;
        }

        let total_len = self.len;

        // Initialize read cursor
        let mut read_seg_idx: usize = 0;
        let mut read_offset: usize = 0;
        let mut read_seg_cap = RawSegmentedVec::<T>::segment_capacity(0);
        let mut read_ptr = unsafe { self.buf.segment_ptr(0) };

        // Initialize write cursor (same as read initially)
        let mut write_seg_idx: usize = 0;
        let mut write_offset: usize = 0;
        let mut write_seg_cap = read_seg_cap;
        let mut write_ptr = read_ptr;

        let mut write_count: usize = 0;
        let mut deleted_count: usize = 0;

        for _ in 0..total_len {
            // Get element at read cursor
            let elem_ptr = unsafe { read_ptr.add(read_offset) };
            let keep = unsafe { f(&mut *elem_ptr) };

            if keep {
                // Move element from read to write position if they differ
                if deleted_count > 0 {
                    unsafe {
                        let write_elem_ptr = write_ptr.add(write_offset);
                        // Move the element (read is now garbage, will be overwritten or truncated)
                        std::ptr::copy_nonoverlapping(elem_ptr, write_elem_ptr, 1);
                    }
                }
                write_count += 1;

                // Advance write cursor
                write_offset += 1;
                if write_offset >= write_seg_cap {
                    write_seg_idx += 1;
                    write_offset = 0;
                    if write_seg_idx < self.buf.segment_count() {
                        write_seg_cap = RawSegmentedVec::<T>::segment_capacity(write_seg_idx);
                        write_ptr = unsafe { self.buf.segment_ptr(write_seg_idx) };
                    }
                }
            } else {
                // Drop the element
                unsafe {
                    std::ptr::drop_in_place(elem_ptr);
                }
                deleted_count += 1;
            }

            // Advance read cursor
            read_offset += 1;
            if read_offset >= read_seg_cap {
                read_seg_idx += 1;
                read_offset = 0;
                if read_seg_idx < self.buf.segment_count() {
                    read_seg_cap = RawSegmentedVec::<T>::segment_capacity(read_seg_idx);
                    read_ptr = unsafe { self.buf.segment_ptr(read_seg_idx) };
                }
            }
        }

        // Update length and write pointer
        self.len = write_count;
        if write_count == 0 && self.buf.segment_count() > 0 {
            let base = unsafe { self.buf.segment_ptr(0) };
            self.write_ptr = base;
            self.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
            self.active_segment_index = 0;
        } else if write_count > 0 {
            self.update_write_ptr_for_len();
        }
    }

    /// Removes consecutive duplicate elements.
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes consecutive elements that satisfy the predicate.
    ///
    /// Uses two independent segment cursors (read and write) that traverse
    /// linearly, avoiding per-element segment/offset calculations.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        if self.len <= 1 {
            return;
        }

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            let mut write_count = 1;
            let mut prev_ptr = NonNull::<T>::dangling().as_ptr();
            for _ in 1..self.len {
                let curr_ptr = NonNull::<T>::dangling().as_ptr();
                let is_dup = unsafe { same_bucket(&mut *prev_ptr, &mut *curr_ptr) };
                if !is_dup {
                    write_count += 1;
                    prev_ptr = curr_ptr;
                }
            }
            self.len = write_count;
            return;
        }

        let total_len = self.len;

        // Initialize read cursor - starts at index 1
        let mut read_seg_idx: usize = 0;
        let mut read_offset: usize = 1;
        let mut read_seg_cap = RawSegmentedVec::<T>::segment_capacity(0);
        let mut read_ptr = unsafe { self.buf.segment_ptr(0) };

        // Handle case where first segment has only 1 element
        if read_offset >= read_seg_cap {
            read_seg_idx = 1;
            read_offset = 0;
            read_seg_cap = RawSegmentedVec::<T>::segment_capacity(1);
            read_ptr = unsafe { self.buf.segment_ptr(1) };
        }

        // Initialize write cursor - points to last written element (index 0)
        let mut write_seg_idx: usize = 0;
        let mut write_offset: usize = 0;
        let mut write_seg_cap = RawSegmentedVec::<T>::segment_capacity(0);
        let mut write_ptr = unsafe { self.buf.segment_ptr(0) };

        let mut write_count: usize = 1; // First element is always kept
        let mut deleted_count: usize = 0;

        for _ in 1..total_len {
            // Get pointers to previous (last written) and current (read) elements
            let prev_elem_ptr = unsafe { write_ptr.add(write_offset) };
            let curr_elem_ptr = unsafe { read_ptr.add(read_offset) };

            let is_dup = unsafe { same_bucket(&mut *prev_elem_ptr, &mut *curr_elem_ptr) };

            if !is_dup {
                // Advance write cursor to next position
                write_offset += 1;
                if write_offset >= write_seg_cap {
                    write_seg_idx += 1;
                    write_offset = 0;
                    write_seg_cap = RawSegmentedVec::<T>::segment_capacity(write_seg_idx);
                    write_ptr = unsafe { self.buf.segment_ptr(write_seg_idx) };
                }

                // Move element from read to write position if they differ
                if deleted_count > 0 {
                    unsafe {
                        let write_elem_ptr = write_ptr.add(write_offset);
                        std::ptr::copy_nonoverlapping(curr_elem_ptr, write_elem_ptr, 1);
                    }
                }
                write_count += 1;
            } else {
                // Drop the duplicate element
                unsafe {
                    std::ptr::drop_in_place(curr_elem_ptr);
                }
                deleted_count += 1;
            }

            // Advance read cursor
            read_offset += 1;
            if read_offset >= read_seg_cap {
                read_seg_idx += 1;
                read_offset = 0;
                if read_seg_idx < self.buf.segment_count() {
                    read_seg_cap = RawSegmentedVec::<T>::segment_capacity(read_seg_idx);
                    read_ptr = unsafe { self.buf.segment_ptr(read_seg_idx) };
                }
            }
        }

        // Update length and write pointer
        self.len = write_count;
        self.update_write_ptr_for_len();
    }

    /// Removes consecutive elements with duplicate keys.
    pub fn dedup_by_key<K, F>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Moves all elements from `other` into `self`.
    ///
    /// Uses chunk-based moving with `ptr::copy_nonoverlapping` for better
    /// performance by copying contiguous memory regions.
    pub fn append(&mut self, other: &mut Self) {
        if other.is_empty() {
            return;
        }

        let other_len = other.len;
        self.reserve(other_len);

        // Handle ZST - no actual copying needed
        if std::mem::size_of::<T>() == 0 {
            self.len += other_len;
            other.len = 0;
            return;
        }

        let old_len = self.len;

        // Initialize write cursor for self (where to start writing)
        let (mut write_seg_idx, mut write_offset) = if old_len == 0 {
            (0, 0)
        } else {
            let (seg, off) = RawSegmentedVec::<T>::location(old_len - 1);
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg);
            if off + 1 >= seg_cap {
                (seg + 1, 0)
            } else {
                (seg, off + 1)
            }
        };

        // Initialize read cursor for other (where to start reading)
        let mut read_seg_idx: usize = 0;
        let mut read_offset: usize = 0;

        let mut remaining = other_len;

        while remaining > 0 {
            // Get segment info for both cursors
            let write_seg_cap = RawSegmentedVec::<T>::segment_capacity(write_seg_idx);
            let write_available = write_seg_cap - write_offset;
            let write_ptr = unsafe { self.buf.segment_ptr(write_seg_idx) };

            let read_seg_cap = RawSegmentedVec::<T>::segment_capacity(read_seg_idx);
            let read_seg_len = if read_seg_idx == other.active_segment_index {
                // Last segment of other may be partially filled
                unsafe {
                    other
                        .write_ptr
                        .offset_from(other.buf.segment_ptr(read_seg_idx))
                        as usize
                }
            } else {
                read_seg_cap
            };
            let read_available = read_seg_len - read_offset;
            let read_ptr = unsafe { other.buf.segment_ptr(read_seg_idx) };

            // Copy the minimum of what's available in both segments
            let to_copy = remaining.min(write_available).min(read_available);

            unsafe {
                std::ptr::copy_nonoverlapping(
                    read_ptr.add(read_offset),
                    write_ptr.add(write_offset),
                    to_copy,
                );
            }

            remaining -= to_copy;
            read_offset += to_copy;
            write_offset += to_copy;

            // Advance read cursor if segment exhausted
            if read_offset >= read_seg_len {
                read_seg_idx += 1;
                read_offset = 0;
            }

            // Advance write cursor if segment exhausted
            if write_offset >= write_seg_cap {
                write_seg_idx += 1;
                write_offset = 0;
            }
        }

        // Update self's length and write pointer
        self.len = old_len + other_len;
        self.update_write_ptr_for_len();

        // Reset other without dropping elements (they've been moved)
        other.len = 0;
        if other.buf.segment_count() > 0 {
            let base = unsafe { other.buf.segment_ptr(0) };
            other.write_ptr = base;
            other.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
            other.active_segment_index = 0;
        } else {
            other.write_ptr = std::ptr::null_mut();
            other.segment_end = std::ptr::null_mut();
            other.active_segment_index = 0;
        }
    }

    /// Splits the vector into two at the given index.
    ///
    /// Uses chunk-based moving with `ptr::copy_nonoverlapping` for better
    /// performance by copying contiguous memory regions.
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len);

        let mut other = Self::new();
        if at == self.len {
            return other;
        }

        let move_count = self.len - at;
        other.reserve(move_count);

        // Handle ZST - no actual copying needed
        if std::mem::size_of::<T>() == 0 {
            other.len = move_count;
            self.len = at;
            return other;
        }

        // Initialize read cursor for self (starting at index `at`)
        let (mut read_seg_idx, mut read_offset) = RawSegmentedVec::<T>::location(at);
        let mut read_seg_cap = RawSegmentedVec::<T>::segment_capacity(read_seg_idx);

        // Initialize write cursor for other (starting at 0)
        let mut write_seg_idx: usize = 0;
        let mut write_offset: usize = 0;

        let mut remaining = move_count;

        while remaining > 0 {
            // Get segment info for read cursor
            let read_seg_len = if read_seg_idx == self.active_segment_index {
                // Last segment of self may be partially filled
                unsafe {
                    self.write_ptr
                        .offset_from(self.buf.segment_ptr(read_seg_idx))
                        as usize
                }
            } else {
                read_seg_cap
            };
            let read_available = read_seg_len - read_offset;
            let read_ptr = unsafe { self.buf.segment_ptr(read_seg_idx) };

            // Get segment info for write cursor
            let write_seg_cap = RawSegmentedVec::<T>::segment_capacity(write_seg_idx);
            let write_available = write_seg_cap - write_offset;
            let write_ptr = unsafe { other.buf.segment_ptr(write_seg_idx) };

            // Copy the minimum of what's available in both segments
            let to_copy = remaining.min(read_available).min(write_available);

            unsafe {
                std::ptr::copy_nonoverlapping(
                    read_ptr.add(read_offset),
                    write_ptr.add(write_offset),
                    to_copy,
                );
            }

            remaining -= to_copy;
            read_offset += to_copy;
            write_offset += to_copy;

            // Advance read cursor if segment exhausted
            if read_offset >= read_seg_len {
                read_seg_idx += 1;
                read_offset = 0;
                if read_seg_idx < self.buf.segment_count() {
                    read_seg_cap = RawSegmentedVec::<T>::segment_capacity(read_seg_idx);
                }
            }

            // Advance write cursor if segment exhausted
            if write_offset >= write_seg_cap {
                write_seg_idx += 1;
                write_offset = 0;
            }
        }

        // Update other's length and write pointer
        other.len = move_count;
        other.update_write_ptr_for_len();

        // Update self's length and write pointer
        self.len = at;
        if at == 0 && self.buf.segment_count() > 0 {
            let base = unsafe { self.buf.segment_ptr(0) };
            self.write_ptr = base;
            self.segment_end = unsafe { base.add(RawSegmentedVec::<T>::segment_capacity(0)) };
            self.active_segment_index = 0;
        } else if at > 0 {
            self.update_write_ptr_for_len();
        }

        other
    }
}

// Search operations
impl<T> SegmentedVec<T> {
    /// Returns `true` if `needle` is a prefix of the vector.
    ///
    /// Uses chunk-based comparison for better performance by comparing
    /// contiguous slices rather than element by element.
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let needle_len = needle.len();
        if needle_len > self.len {
            return false;
        }
        if needle_len == 0 {
            return true;
        }

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            // For ZST, all instances are equal
            return true;
        }

        let mut seg_idx: usize = 0;
        let mut seg_offset: usize = 0;
        let mut needle_offset: usize = 0;

        while needle_offset < needle_len {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let seg_ptr = unsafe { self.buf.segment_ptr(seg_idx) };

            // How many elements available in this segment
            let seg_available = seg_cap - seg_offset;
            // How many elements left to compare in needle
            let needle_remaining = needle_len - needle_offset;
            // Compare the minimum of both
            let to_compare = seg_available.min(needle_remaining);

            // Create slices and compare
            let seg_slice =
                unsafe { std::slice::from_raw_parts(seg_ptr.add(seg_offset), to_compare) };
            let needle_slice = &needle[needle_offset..needle_offset + to_compare];

            if seg_slice != needle_slice {
                return false;
            }

            needle_offset += to_compare;
            seg_offset += to_compare;

            // Move to next segment if current is exhausted
            if seg_offset >= seg_cap {
                seg_idx += 1;
                seg_offset = 0;
            }
        }

        true
    }

    /// Returns `true` if `needle` is a suffix of the vector.
    ///
    /// Uses chunk-based comparison for better performance by comparing
    /// contiguous slices rather than element by element.
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        let needle_len = needle.len();
        if needle_len > self.len {
            return false;
        }
        if needle_len == 0 {
            return true;
        }

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            // For ZST, all instances are equal
            return true;
        }

        let start_idx = self.len - needle_len;

        // Initialize cursor at start_idx
        let (mut seg_idx, mut seg_offset) = RawSegmentedVec::<T>::location(start_idx);
        let mut needle_offset: usize = 0;

        while needle_offset < needle_len {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let seg_ptr = unsafe { self.buf.segment_ptr(seg_idx) };

            // How many elements available in this segment from current offset
            // For the last segment, limit to actual elements
            let seg_len = if seg_idx == self.active_segment_index {
                unsafe { self.write_ptr.offset_from(seg_ptr) as usize }
            } else {
                seg_cap
            };
            let seg_available = seg_len - seg_offset;

            // How many elements left to compare in needle
            let needle_remaining = needle_len - needle_offset;
            // Compare the minimum of both
            let to_compare = seg_available.min(needle_remaining);

            // Create slices and compare
            let seg_slice =
                unsafe { std::slice::from_raw_parts(seg_ptr.add(seg_offset), to_compare) };
            let needle_slice = &needle[needle_offset..needle_offset + to_compare];

            if seg_slice != needle_slice {
                return false;
            }

            needle_offset += to_compare;
            seg_offset += to_compare;

            // Move to next segment if current is exhausted
            if seg_offset >= seg_len {
                seg_idx += 1;
                seg_offset = 0;
            }
        }

        true
    }

    /// Binary search for `x` in a sorted vector.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary search using a comparison function.
    ///
    /// Uses reverse linear scan of segments followed by native binary search
    /// on the target segment's slice for better cache performance.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        if self.len == 0 {
            return Err(0);
        }

        // Handle ZST - fall back to simple binary search
        if std::mem::size_of::<T>() == 0 {
            return self.binary_search_by_simple(&mut f);
        }

        // Reverse linear scan of segments to find target segment
        let mut segment_idx = self.active_segment_index;
        let mut segment_start_idx = 0;

        // Calculate starting index of the active segment
        if segment_idx > 0 {
            segment_start_idx = RawSegmentedVec::<T>::segment_capacity(0);
            for i in 1..segment_idx {
                segment_start_idx += RawSegmentedVec::<T>::segment_capacity(i);
            }
        }

        loop {
            let segment_base = unsafe { self.buf.segment_ptr(segment_idx) };
            let segment_cap = RawSegmentedVec::<T>::segment_capacity(segment_idx);

            // Calculate how many elements are in this segment
            let segment_len = if segment_idx == self.active_segment_index {
                unsafe { self.write_ptr.offset_from(segment_base) as usize }
            } else {
                segment_cap
            };

            if segment_len == 0 {
                // Empty segment (shouldn't happen in normal use, but handle it)
                if segment_idx == 0 {
                    return Err(0);
                }
                segment_idx -= 1;
                segment_start_idx -= RawSegmentedVec::<T>::segment_capacity(segment_idx);
                continue;
            }

            // Check first element of this segment
            let first_cmp = f(unsafe { &*segment_base });

            match first_cmp {
                Ordering::Greater => {
                    // Target is less than first element of this segment
                    // Move to previous segment
                    if segment_idx == 0 {
                        return Err(0);
                    }
                    segment_idx -= 1;
                    segment_start_idx -= RawSegmentedVec::<T>::segment_capacity(segment_idx);
                }
                Ordering::Equal => {
                    // Found at first element
                    return Ok(segment_start_idx);
                }
                Ordering::Less => {
                    // Target >= first element, search in this segment
                    let slice = unsafe { std::slice::from_raw_parts(segment_base, segment_len) };
                    match slice.binary_search_by(&mut f) {
                        Ok(pos) => return Ok(segment_start_idx + pos),
                        Err(pos) => return Err(segment_start_idx + pos),
                    }
                }
            }
        }
    }

    /// Simple element-by-element binary search (fallback for ZST)
    fn binary_search_by_simple<F>(&self, f: &mut F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let mut left = 0;
        let mut right = self.len;

        while left < right {
            let mid = left + (right - left) / 2;
            let cmp = f(unsafe { self.unchecked_at(mid) });
            match cmp {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    /// Binary search using a key extraction function.
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }
}

// Sorting operations
impl<T> SegmentedVec<T> {
    /// Sorts the vector in place.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        let len = self.len;
        sort::merge_sort(self, 0, len, &mut |a, b| a < b);
    }

    /// Sorts the vector with a comparison function.
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        let len = self.len;
        sort::merge_sort(self, 0, len, &mut |a, b| compare(a, b) == Ordering::Less);
    }

    /// Sorts the vector with a key extraction function.
    pub fn sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_by(|a, b| f(a).cmp(&f(b)));
    }

    /// Sorts the vector with an unstable sort.
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        let len = self.len;
        sort::heapsort(self, 0, len, &mut |a, b| a < b);
    }

    /// Sorts the vector with an unstable sort using a comparison function.
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        let len = self.len;
        sort::heapsort(self, 0, len, &mut |a, b| compare(a, b) == Ordering::Less);
    }

    /// Sorts the vector with an unstable sort using a key extraction function.
    pub fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_unstable_by(|a, b| f(a).cmp(&f(b)));
    }

    /// Returns `true` if the vector is sorted.
    pub fn is_sorted(&self) -> bool
    where
        T: Ord,
    {
        self.is_sorted_by(|a, b| a <= b)
    }

    /// Returns `true` if the vector is sorted according to the comparison function.
    pub fn is_sorted_by<F>(&self, mut compare: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        for i in 1..self.len {
            if !compare(unsafe { self.unchecked_at(i - 1) }, unsafe {
                self.unchecked_at(i)
            }) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if the vector is sorted by the given key.
    pub fn is_sorted_by_key<K, F>(&self, mut f: F) -> bool
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.is_sorted_by(|a, b| f(a) <= f(b))
    }

    /// Returns the index of the partition point.
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.binary_search_by(|x| {
            if pred(x) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .unwrap_or_else(|i| i)
    }

    /// Rotates the vector left by `mid` elements.
    pub fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len);
        self.as_mut_slice().rotate_left(mid);
    }

    /// Rotates the vector right by `k` elements.
    pub fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len);
        self.as_mut_slice().rotate_right(k);
    }
}

// Slice operations
impl<T> SegmentedVec<T> {
    /// Returns an iterator over chunks of `chunk_size` elements.
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        self.as_slice().chunks(chunk_size)
    }

    /// Returns an iterator over windows of `size` elements.
    pub fn windows(&self, size: usize) -> Windows<'_, T> {
        self.as_slice().windows(size)
    }

    /// Returns an iterator over chunks from the end.
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'_, T> {
        self.as_slice().rchunks(chunk_size)
    }

    /// Returns an iterator over exact chunks.
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        self.as_slice().chunks_exact(chunk_size)
    }

    /// Returns a slice view of a range.
    pub fn slice(&self, range: std::ops::Range<usize>) -> SegmentedSlice<'_, T> {
        assert!(range.end <= self.len);
        SegmentedSlice::from_range(self, range.start, range.end)
    }

    /// Returns a mutable slice view of a range.
    pub fn slice_mut(&mut self, range: std::ops::Range<usize>) -> SegmentedSliceMut<'_, T> {
        assert!(range.end <= self.len);
        SegmentedSliceMut::from_range(self, range.start, range.end)
    }
}

// Extend operations
impl<T> SegmentedVec<T> {
    /// Extends the vector with elements from a slice.
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.reserve(other.len());
        for item in other {
            self.push(item.clone());
        }
    }

    /// Extends the vector with elements from a Copy slice.
    pub fn extend_from_copy_slice(&mut self, other: &[T])
    where
        T: Copy,
    {
        self.reserve(other.len());
        for &item in other {
            self.push(item);
        }
    }
}

// Generic implementation for all allocators (needed for Drop)
impl<T, A: Allocator> SegmentedVec<T, A> {
    /// Clears the vector, removing all elements.
    ///
    /// This drops all elements but keeps the allocated memory.
    ///
    /// Optimized to:
    /// - Handle ZST separately (no segments involved)
    /// - Split full segments (0..active_idx) from partial segment (active_idx)
    /// - Derive active segment base from registers (segment_end - capacity) instead of memory lookups
    fn clear_internal(&mut self) {
        let old_len = self.len;
        if old_len == 0 {
            return;
        }

        // Handle ZST - no actual memory to deal with
        if std::mem::size_of::<T>() == 0 {
            if std::mem::needs_drop::<T>() {
                for _ in 0..old_len {
                    unsafe {
                        std::ptr::drop_in_place(NonNull::<T>::dangling().as_ptr());
                    }
                }
            }
            self.len = 0;
            return;
        }

        // Capture drop parameters from registers BEFORE resetting state
        let active_idx = self.active_segment_index;
        let active_cap = RawSegmentedVec::<T, A>::segment_capacity(active_idx);
        // Derive active segment base from segment_end (in register) instead of array lookup
        let active_base = unsafe { self.segment_end.sub(active_cap) };
        // Elements in active segment = write_ptr - base
        let active_len = unsafe { self.write_ptr.offset_from(active_base) as usize };

        // Reset len BEFORE dropping to prevent double-free if drop panics
        self.len = 0;

        // Reset write_ptr to segment 0 (keep capacity usable)
        if self.buf.segment_count() > 0 {
            let base = unsafe { self.buf.segment_ptr(0) };
            self.write_ptr = base;
            self.segment_end = unsafe { base.add(RawSegmentedVec::<T, A>::segment_capacity(0)) };
            self.active_segment_index = 0;
        }

        // Drop all elements
        if std::mem::needs_drop::<T>() {
            // Drop full segments (0..active_idx) - no min() needed, they're all full
            for seg_idx in 0..active_idx {
                let seg_cap = RawSegmentedVec::<T, A>::segment_capacity(seg_idx);
                let base = unsafe { self.buf.segment_ptr(seg_idx) };
                unsafe {
                    std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(base, seg_cap));
                }
            }

            // Drop partial active segment (already captured active_base and active_len)
            if active_len > 0 {
                unsafe {
                    std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                        active_base,
                        active_len,
                    ));
                }
            }
        }
    }
}

// Trait implementations
impl<T, A: Allocator> Drop for SegmentedVec<T, A> {
    fn drop(&mut self) {
        self.clear_internal();
        // RawSegmentedVec will be dropped automatically and free the memory
    }
}

impl<T: Clone> Clone for SegmentedVec<T> {
    /// Clones the vector segment-by-segment for better performance.
    ///
    /// - For Copy types: uses slice operations that can be vectorized (SIMD)
    /// - For Clone types: clones within contiguous segments, avoiding per-element segment lookups
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self::new();
        }

        let mut new_vec = Self::with_capacity(self.len);

        // Handle ZST
        if std::mem::size_of::<T>() == 0 {
            for _ in 0..self.len {
                unsafe {
                    let val = (*NonNull::<T>::dangling().as_ptr()).clone();
                    std::ptr::write(NonNull::<T>::dangling().as_ptr(), val);
                }
            }
            new_vec.len = self.len;
            return new_vec;
        }

        let total_len = self.len;
        let active_idx = self.active_segment_index;

        // Clone full segments (0..active_idx)
        for seg_idx in 0..active_idx {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let src_base = unsafe { self.buf.segment_ptr(seg_idx) };
            let dst_base = unsafe { new_vec.buf.segment_ptr(seg_idx) };

            // For Copy types, this can be vectorized
            if !std::mem::needs_drop::<T>() {
                unsafe {
                    let src_slice = std::slice::from_raw_parts(src_base, seg_cap);
                    let dst_slice = std::slice::from_raw_parts_mut(dst_base, seg_cap);
                    dst_slice.clone_from_slice(src_slice);
                }
            } else {
                // For non-Copy types, clone element by element within segment
                for i in 0..seg_cap {
                    unsafe {
                        let val = (*src_base.add(i)).clone();
                        std::ptr::write(dst_base.add(i), val);
                    }
                }
            }
        }

        // Clone partial active segment
        let active_cap = RawSegmentedVec::<T>::segment_capacity(active_idx);
        let active_base = unsafe { self.segment_end.sub(active_cap) };
        let active_len = unsafe { self.write_ptr.offset_from(active_base) as usize };

        if active_len > 0 {
            let src_base = active_base;
            let dst_base = unsafe { new_vec.buf.segment_ptr(active_idx) };

            if !std::mem::needs_drop::<T>() {
                unsafe {
                    let src_slice = std::slice::from_raw_parts(src_base, active_len);
                    let dst_slice = std::slice::from_raw_parts_mut(dst_base, active_len);
                    dst_slice.clone_from_slice(src_slice);
                }
            } else {
                for i in 0..active_len {
                    unsafe {
                        let val = (*src_base.add(i)).clone();
                        std::ptr::write(dst_base.add(i), val);
                    }
                }
            }
        }

        // Update new_vec state
        new_vec.len = total_len;
        new_vec.update_write_ptr_for_len();

        new_vec
    }
}

impl<T: PartialEq> PartialEq for SegmentedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        let len = self.len;
        if len == 0 {
            return true;
        }

        // ZST: lengths are equal, so they're equal
        if size_of::<T>() == 0 {
            return true;
        }

        // Both vecs have same length, so same segment layout
        let active_idx = self.active_segment_index;

        // Compare full segments (0..active_idx)
        for seg_idx in 0..active_idx {
            let seg_cap = RawSegmentedVec::<T>::segment_capacity(seg_idx);
            let self_slice = unsafe {
                let ptr = self.buf.segment_ptr(seg_idx) as *const T;
                std::slice::from_raw_parts(ptr, seg_cap)
            };
            let other_slice = unsafe {
                let ptr = other.buf.segment_ptr(seg_idx) as *const T;
                std::slice::from_raw_parts(ptr, seg_cap)
            };
            if self_slice != other_slice {
                return false;
            }
        }

        // Compare partial active segment
        let active_cap = RawSegmentedVec::<T>::segment_capacity(active_idx);
        let self_base = unsafe { self.segment_end.sub(active_cap) };
        let other_base = unsafe { other.segment_end.sub(active_cap) };
        let active_len = unsafe { self.write_ptr.offset_from(self_base) as usize };

        let self_active = unsafe { std::slice::from_raw_parts(self_base, active_len) };
        let other_active = unsafe { std::slice::from_raw_parts(other_base, active_len) };

        self_active == other_active
    }
}

impl<T: Eq> Eq for SegmentedVec<T> {}

impl<T: PartialOrd> PartialOrd for SegmentedVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        for i in 0..self.len.min(other.len) {
            match unsafe { self.unchecked_at(i).partial_cmp(other.unchecked_at(i)) } {
                Some(Ordering::Equal) => continue,
                non_eq => return non_eq,
            }
        }
        Some(self.len.cmp(&other.len))
    }
}

impl<T: Ord> Ord for SegmentedVec<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in 0..self.len.min(other.len) {
            match unsafe { self.unchecked_at(i).cmp(other.unchecked_at(i)) } {
                Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }
        self.len.cmp(&other.len)
    }
}

impl<T: std::hash::Hash> std::hash::Hash for SegmentedVec<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for i in 0..self.len {
            unsafe { self.unchecked_at(i).hash(state) };
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SegmentedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Default for SegmentedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<usize> for SegmentedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T> IndexMut<usize> for SegmentedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T> Extend<T> for SegmentedVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<'a, T: Clone + 'a> Extend<&'a T> for SegmentedVec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item.clone());
        }
    }
}

impl<T> FromIterator<T> for SegmentedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::new();
        vec.extend(iter);
        vec
    }
}

impl<T> IntoIterator for SegmentedVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            vec: self,
            index: 0,
        }
    }
}

impl<'a, T> IntoIterator for &'a SegmentedVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut SegmentedVec<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// IndexedAccess implementation for sorting
impl<T> sort::IndexedAccess<T> for SegmentedVec<T> {
    #[inline]
    fn get_ref(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { self.unchecked_at(index) }
    }

    #[inline]
    fn get_ptr(&self, index: usize) -> *const T {
        debug_assert!(index < self.len);
        unsafe { self.buf.ptr_at(index) }
    }

    #[inline]
    fn get_ptr_mut(&mut self, index: usize) -> *mut T {
        debug_assert!(index < self.len);
        unsafe { self.buf.ptr_at(index) }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        SegmentedVec::swap(self, a, b);
    }
}

// Safety implementations
unsafe impl<T: Send> Send for SegmentedVec<T> {}
unsafe impl<T: Sync> Sync for SegmentedVec<T> {}

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

    #[test]
    fn test_sort() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]);
        vec.sort();

        let expected = vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9];
        assert_eq!(vec.iter().copied().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_drain() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let drained: Vec<i32> = vec.drain(3..7).collect();
        assert_eq!(drained, vec![3, 4, 5, 6]);
        assert_eq!(vec.len(), 6);
        assert_eq!(
            vec.iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 7, 8, 9]
        );
    }

    #[test]
    fn test_with_capacity() {
        let vec: SegmentedVec<i32> = SegmentedVec::with_capacity(100);
        assert!(vec.capacity() >= 100);
        assert!(vec.is_empty());
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
