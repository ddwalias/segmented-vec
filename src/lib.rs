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

mod slice;
mod sort;

pub use slice::{
    Chunks, ChunksExact, RChunks, SegmentedSlice, SegmentedSliceMut, SliceIter, SliceIterMut,
    Windows,
};

use std::alloc::{self, Layout};
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Maximum number of dynamic segments supported.
/// With exponentially growing segments, 64 segments can hold more than 2^64 elements.
const MAX_SEGMENTS: usize = 64;

/// A segmented vector with stable pointers.
#[repr(C)]
pub struct SegmentedVec<T> {
    /// Cached pointer to the next write position (for fast push)
    write_ptr: *mut T,
    /// Pointer to the end of the current segment
    segment_end: *mut T,
    len: usize,
    /// Index of the current segment
    active_segment_index: usize,
    segment_count: usize,
    dynamic_segments: [*mut T; MAX_SEGMENTS],
    _marker: PhantomData<T>,
}

impl<T> SegmentedVec<T> {
    /// Minimum capacity for the first dynamic segment
    /// Avoids tiny allocations that heap allocators round up anyway.
    /// - 8 for 1-byte elements (allocators round up small requests)
    /// - 4 for moderate elements (<= 1 KiB)
    /// - 1 for large elements (avoid wasting space)
    const MIN_NON_ZERO_CAP: usize = {
        let size = std::mem::size_of::<T>();
        if size == 1 {
            8
        } else if size <= 1024 {
            4
        } else {
            1
        }
    };

    const MIN_CAP_EXP: u32 = Self::MIN_NON_ZERO_CAP.trailing_zeros();

    /// Creates a new empty `SegmentedVec`.
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
            dynamic_segments: [std::ptr::null_mut(); MAX_SEGMENTS],
            segment_count: 0,
            len: 0,
            write_ptr: std::ptr::null_mut(),
            segment_end: std::ptr::null_mut(),
            // Optimization: Initialize to MAX so the first increment wrap around and make it 0
            active_segment_index: usize::MAX,
            _marker: PhantomData,
        }
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
        Self::compute_capacity(self.segment_count as u32)
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
        self.active_segment_index = self.active_segment_index.wrapping_add(1);

        if self.active_segment_index >= self.segment_count {
            self.grow_once();
        }

        let idx = self.active_segment_index;

        // Load the base pointer for the new segment
        // SAFETY: grow_once guarantees this slot is populated.
        let base = unsafe { *self.dynamic_segments.get_unchecked(idx) };
        let shelf_size = Self::MIN_NON_ZERO_CAP << idx;

        unsafe {
            std::ptr::write(base, value);
            self.write_ptr = base.add(1);
            self.segment_end = base.add(shelf_size);
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

        // Fast path: current segment still has elements
        // SAFETY: len > 0 means active_segment_index is valid
        let segment_base = unsafe {
            *self
                .dynamic_segments
                .get_unchecked(self.active_segment_index)
        };
        if self.write_ptr > segment_base {
            self.len -= 1;
            unsafe {
                // Move back to the last written element
                self.write_ptr = self.write_ptr.sub(1);
                // Return the value
                Some(std::ptr::read(self.write_ptr))
            }
        } else {
            // Slow path: current segment doesn't have any element left, cross the boundary
            self.pop_slow_path()
        }
    }

    #[cold]
    #[inline(never)]
    fn pop_slow_path(&mut self) -> Option<T> {
        // Safe because caller checked len > 0, and we're at segment boundary,
        // so active_segment_index must be > 0
        self.active_segment_index -= 1;
        let idx = self.active_segment_index;

        let base = unsafe { *self.dynamic_segments.get_unchecked(idx) };
        let capacity = Self::MIN_NON_ZERO_CAP << idx;

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
            // SAFETY: len > 0 means segment 0 exists with at least one element
            Some(unsafe { &*self.dynamic_segments[0] })
        }
    }

    /// Returns a mutable reference to the first element, or `None` if empty.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: len > 0 means segment 0 exists with at least one element
            Some(unsafe { &mut *self.dynamic_segments[0] })
        }
    }

    /// Returns a reference to the last element, or `None` if empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }

        // Fast path: check if current segment has elements
        // SAFETY: len > 0 means active_segment_index is valid
        let segment_base = unsafe {
            *self
                .dynamic_segments
                .get_unchecked(self.active_segment_index)
        };

        if self.write_ptr > segment_base {
            // Last element is in the current segment
            Some(unsafe { &*self.write_ptr.sub(1) })
        } else {
            // Current segment is empty, last element is in previous segment
            self.get(self.len - 1)
        }
    }

    /// Returns a mutable reference to the last element, or `None` if empty.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }

        // Fast path: check if current segment has elements
        // SAFETY: len > 0 means active_segment_index is valid
        let segment_base = unsafe {
            *self
                .dynamic_segments
                .get_unchecked(self.active_segment_index)
        };

        if self.write_ptr > segment_base {
            // Last element is in the current segment
            Some(unsafe { &mut *self.write_ptr.sub(1) })
        } else {
            // Current segment is empty, last element is in previous segment
            self.get_mut(self.len - 1)
        }
    }

    /// Returns `true` if the slice contains an element with the given value.
    ///
    /// This iterates over the contiguous memory segments and uses the
    /// standard library's vectorized `contains` on each segment.
    #[inline]
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        let mut remaining = self.len;
        let mut segment_idx = 0;

        while remaining > 0 {
            let segment_capacity = Self::MIN_NON_ZERO_CAP << segment_idx;
            let segment_len = segment_capacity.min(remaining);

            // SAFETY: segment_idx is valid while remaining > 0
            let base = unsafe { *self.dynamic_segments.get_unchecked(segment_idx) };
            let slice = unsafe { std::slice::from_raw_parts(base, segment_len) };

            if slice.contains(x) {
                return true;
            }

            segment_idx += 1;
            remaining -= segment_len;
        }

        false
    }

    /// Returns `true` if `needle` is a prefix of the vector.
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        if needle.len() > self.len {
            return false;
        }

        if needle.is_empty() {
            return true;
        }

        let mut current_segment_capacity = Self::MIN_NON_ZERO_CAP;
        let mut needle_cursor = 0;
        let mut idx = 0;

        while needle_cursor < needle.len() {
            unsafe {
                let segment_base = *self.dynamic_segments.get_unchecked(idx);

                let remaining = needle.len() - needle_cursor;
                let match_len = remaining.min(current_segment_capacity);

                let haystack_slice = std::slice::from_raw_parts(segment_base, match_len);

                let needle_slice = &needle[needle_cursor..needle_cursor + match_len];

                if haystack_slice != needle_slice {
                    return false;
                }

                needle_cursor += match_len;
                idx += 1;
                current_segment_capacity <<= 1;
            }
        }

        true
    }

    /// Returns `true` if `needle` is a suffix of the vector.
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        if needle.len() > self.len {
            return false;
        }

        if needle.is_empty() {
            return true;
        }

        let mut remaining = needle.len();
        let mut idx = self.active_segment_index;

        // Check active segment
        unsafe {
            let segment_base = *self.dynamic_segments.get_unchecked(idx);

            // Calculate the length using pointer math
            let segment_len = self.write_ptr.offset_from(segment_base) as usize;
            let match_len = remaining.min(segment_len);

            let haystack_start = self.write_ptr.sub(match_len);
            let haystack_slice = std::slice::from_raw_parts(haystack_start, match_len);

            let needle_start = remaining - match_len;
            let needle_slice = &needle[needle_start..remaining];

            if haystack_slice != needle_slice {
                return false;
            }

            remaining -= match_len;
            if remaining == 0 {
                return true;
            }

            idx -= 1;
        }

        // Check other previous segments
        loop {
            unsafe {
                let segment_base = *self.dynamic_segments.get_unchecked(idx);

                let segment_capacity = Self::MIN_NON_ZERO_CAP << idx;
                let match_len = remaining.min(segment_capacity);

                let haystack_start = segment_base.add(segment_capacity - match_len);
                let haystack_slice = std::slice::from_raw_parts(haystack_start, match_len);

                let needle_start = remaining - match_len;
                let needle_slice = &needle[needle_start..remaining];

                if haystack_slice != needle_slice {
                    return false;
                }

                remaining -= match_len;
                if remaining == 0 {
                    return true;
                }

                idx -= 1;
            }
        }
    }

    /// Binary searches this vector for a given element.
    ///
    /// If the value is found, returns `Ok(index)`. If not found, returns
    /// `Err(index)` where `index` is the position where the element could be inserted.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|p| p.cmp(x))
    }

    /// Binary searches this vector with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        if self.len == 0 {
            return Err(0);
        }

        // Check Active Segment (50% Probability)
        // We calculate base pointer using ALU math because we need capacity later
        let active_capacity = Self::MIN_NON_ZERO_CAP << self.active_segment_index;

        // SAFETY: segment_end is valid because len > 0
        let segment_base = unsafe { self.segment_end.sub(active_capacity) };
        let first_elem = unsafe { &*segment_base };

        // If the first element is <= Target, the target is in this segment
        // (or this segment is where we need to insert).
        if f(first_elem) != Ordering::Greater {
            unsafe {
                // IMPORTANT: Active segment length is determined by 'write_ptr', NOT capacity.
                let segment_len = self.write_ptr.offset_from(segment_base) as usize;
                let slice = std::slice::from_raw_parts(segment_base, segment_len);

                let offset = active_capacity - Self::MIN_NON_ZERO_CAP;

                return match slice.binary_search_by(f) {
                    Ok(idx) => Ok(offset + idx),
                    Err(idx) => Err(offset + idx),
                };
            }
        }

        // Reverse Linear Scan
        // Target is smaller than the active segment. Check previous segments.
        // Previous segments are GUARANTEED to be full.
        let mut idx = self.active_segment_index;
        while idx > 0 {
            idx -= 1;
            unsafe {
                let segment_base = *self.dynamic_segments.get_unchecked(idx);
                if f(&*segment_base) != Ordering::Greater {
                    let capacity = Self::MIN_NON_ZERO_CAP << idx;
                    let slice = std::slice::from_raw_parts(segment_base, capacity);

                    let global_offset = capacity - Self::MIN_NON_ZERO_CAP;

                    return match slice.binary_search_by(f) {
                        Ok(i) => Ok(global_offset + i),
                        Err(i) => Err(global_offset + i),
                    };
                }
            }
        }

        Err(0)
    }

    /// Binary searches this vector with a key extraction function.
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|k| f(k).cmp(b))
    }

    /// Swaps two elements in the vector.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        let ptr_b = &raw mut self[a];
        let ptr_a = &raw mut self[b];
        // Note that accessing the elements behind `a` and `b` is checked and will
        // panic when out of bounds.
        unsafe {
            std::ptr::swap(ptr_b, ptr_a);
        }
    }

    /// Reverses the order of elements in the vector.
    pub fn reverse(&mut self) {
        if self.len < 2 {
            return;
        }

        let mut remaining_total = self.len;

        // front cursor
        let mut f_segment_index = 0;
        let mut f_segment_capacity = Self::MIN_NON_ZERO_CAP;
        let mut f_segment_base = unsafe { *self.dynamic_segments.get_unchecked(0) };
        let mut f_remaining = f_segment_capacity;

        // back cursor
        let mut b_segment_index = self.active_segment_index;
        let mut b_segment_capacity = Self::MIN_NON_ZERO_CAP << b_segment_index;
        let mut b_segment_base = unsafe { *self.dynamic_segments.get_unchecked(b_segment_index) };
        let mut b_remaining = unsafe { self.write_ptr.offset_from(b_segment_base) as usize };
        let mut b_segment_end = unsafe { self.write_ptr.sub(1) };

        loop {
            // front and back is in the same slice, use slice.reverse
            if f_segment_index == b_segment_index {
                unsafe {
                    let slice = std::slice::from_raw_parts_mut(f_segment_base, remaining_total);
                    slice.reverse();
                }
                return;
            }

            let count = f_remaining.min(b_remaining);

            unsafe {
                self.swap_chunks_reversed_simd(f_segment_base, b_segment_end, count);

                f_segment_base = f_segment_base.add(count);
                b_segment_end = b_segment_end.sub(count);
            }

            remaining_total -= count * 2;
            if remaining_total == 0 {
                return;
            }

            // Update Front
            f_remaining -= count;
            if f_remaining == 0 {
                f_segment_index += 1;
                f_segment_capacity <<= 1;
                f_remaining = f_segment_capacity;
                // Load next front segment
                unsafe {
                    f_segment_base = *self.dynamic_segments.get_unchecked(f_segment_index);
                }
            }

            // Update Back
            b_remaining -= count;
            if b_remaining == 0 {
                b_segment_index -= 1;
                b_segment_capacity >>= 1;
                b_remaining = b_segment_capacity;
                unsafe {
                    b_segment_base = *self.dynamic_segments.get_unchecked(b_segment_index);
                    b_segment_end = b_segment_base.add(b_segment_capacity - 1);
                }
            }
        }
    }

    /// SIMD swap helper.
    #[inline(always)]
    unsafe fn swap_chunks_reversed_simd(&mut self, base_a: *mut T, base_b: *mut T, count: usize) {
        // LLVM will generate SIMD instructions
        for i in 0..count {
            std::ptr::swap(base_a.add(i), base_b.sub(i));
        }
    }

    /// Fills the vector with the given value.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        if self.len == 0 {
            return;
        }

        // Fill non-active segments
        for i in 0..self.active_segment_index {
            unsafe {
                let base = *self.dynamic_segments.get_unchecked(i);
                let capacity = Self::MIN_NON_ZERO_CAP << i;

                // Create a standard mutable slice
                let slice = std::slice::from_raw_parts_mut(base, capacity);

                // We must clone here because we need 'value' for subsequent segments
                slice.fill(value.clone());
            }
        }

        // Fill the active segment
        unsafe {
            let base = *self
                .dynamic_segments
                .get_unchecked(self.active_segment_index);

            let count = self.write_ptr.offset_from(base) as usize;
            let slice = std::slice::from_raw_parts_mut(base, count);

            // Optimization: Move the final 'value' in, avoiding one clone
            slice.fill(value);
        }
    }

    /// Fills the vector with values produced by a function.
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        if self.len == 0 {
            return;
        }

        // Fill non-active segments
        for i in 0..self.active_segment_index {
            unsafe {
                let base = *self.dynamic_segments.get_unchecked(i);
                let capacity = Self::MIN_NON_ZERO_CAP << i;

                // Create a standard mutable slice
                let slice = std::slice::from_raw_parts_mut(base, capacity);
                slice.fill_with(&mut f);
            }
        }

        // Fill the active segment
        unsafe {
            let base = *self
                .dynamic_segments
                .get_unchecked(self.active_segment_index);

            let count = self.write_ptr.offset_from(base) as usize;
            let slice = std::slice::from_raw_parts_mut(base, count);
            slice.fill_with(&mut f);
        }
    }

    /// Clears the vector, removing all elements.
    ///
    /// This does not deallocate the dynamic segments.
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }

        if std::mem::needs_drop::<T>() {
            // Clear non-active segments
            for i in 0..self.active_segment_index {
                unsafe {
                    let base = *self.dynamic_segments.get_unchecked(i);
                    let capacity = Self::MIN_NON_ZERO_CAP << i;

                    // Create a standard mutable slice
                    let slice = std::slice::from_raw_parts_mut(base, capacity);
                    std::ptr::drop_in_place(slice);
                }
            }

            // Clear the active segment
            unsafe {
                let base = *self
                    .dynamic_segments
                    .get_unchecked(self.active_segment_index);

                let count = self.write_ptr.offset_from(base) as usize;
                let slice = std::slice::from_raw_parts_mut(base, count);
                std::ptr::drop_in_place(slice);
            }
        }

        self.len = 0;
        //Because original len > 0, we know that there's at least a segment
        unsafe {
            // Point to Segment 0 so next push is O(1)
            let base = *self.dynamic_segments.get_unchecked(0);
            self.write_ptr = base;
            self.segment_end = base.add(Self::MIN_NON_ZERO_CAP);
            self.active_segment_index = 0;
        }
    }

    /// Shortens the vector, keeping the first `new_len` elements.
    ///
    /// If `new_len` is greater than or equal to the current length, this has no effect.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }

        if std::mem::needs_drop::<T>() {
            let dynamic_new_len = new_len;
            let dynamic_old_len = self.len;

            if dynamic_new_len < dynamic_old_len {
                let mut pos = 0usize;
                for shelf in 0..self.segment_count {
                    let size = Self::shelf_size(shelf as u32);
                    let seg_end = pos + size;

                    // Calculate overlap with [dynamic_new_len, dynamic_old_len)
                    let drop_start = dynamic_new_len.max(pos);
                    let drop_end = dynamic_old_len.min(seg_end);

                    if drop_start < drop_end {
                        let offset = drop_start - pos;
                        let count = drop_end - drop_start;
                        let ptr =
                            unsafe { (*self.dynamic_segments.get_unchecked(shelf)).add(offset) };
                        unsafe {
                            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(ptr, count))
                        };
                    }

                    pos = seg_end;
                    if pos >= dynamic_old_len {
                        break;
                    }
                }
            }
        }

        self.len = new_len;
        // Update write pointer cache
        if new_len > 0 {
            let biased = new_len + Self::MIN_NON_ZERO_CAP;
            let msb = biased.ilog2();
            let idx = (msb - Self::MIN_CAP_EXP) as usize;
            self.active_segment_index = idx;

            let capacity = 1usize << msb;
            let offset = biased ^ capacity;
            unsafe {
                let base = *self.dynamic_segments.get_unchecked(idx);
                self.write_ptr = base.add(offset);
                self.segment_end = base.add(capacity);
            }
        } else {
            // Reset to "Genesis" state
            self.write_ptr = std::ptr::null_mut();
            self.segment_end = std::ptr::null_mut();
            // Set to -1 so the next push wraps to 0
            self.active_segment_index = usize::MAX;
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.grow_capacity(self.len + additional);
        // Initialize write pointer if we didn't have capacity before
        if self.write_ptr.is_null() {
            unsafe {
                // We just guaranteed segment 0 exists in the loop above.
                let base = *self.dynamic_segments.get_unchecked(0);
                self.write_ptr = base;
                self.segment_end = base.add(Self::MIN_NON_ZERO_CAP);

                // Set to 0 so we are valid.
                // Note: push_slow expects MAX->0 transition, but since we set
                // write_ptr here, the first push will hit the FAST path (write_ptr < end)
                // and skip push_slow entirely.
                self.active_segment_index = 0;
            }
        }
    }

    /// Shrinks the capacity to match the current length.
    ///
    /// This deallocates unused dynamic segments.
    pub fn shrink_to_fit(&mut self) {
        self.shrink_capacity(self.len);
    }

    /// Returns an iterator over references to the elements.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        // Initialize with null pointers - first next() call will set up the segment
        Iter {
            vec: self,
            ptr: std::ptr::null(),
            segment_end: std::ptr::null(),
            index: 0,
            shelf_index: 0,
        }
    }

    /// Returns an iterator over mutable references to the elements.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        // Initialize with null pointers - first next() call will set up the segment
        IterMut {
            vec: self,
            ptr: std::ptr::null_mut(),
            segment_end: std::ptr::null_mut(),
            index: 0,
            shelf_index: 0,
        }
    }

    /// Returns an immutable slice view of the entire vector.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend(0..10);
    ///
    /// let slice = vec.as_slice();
    /// assert_eq!(slice.len(), 10);
    /// assert_eq!(slice[0], 0);
    /// ```
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'_, T> {
        SegmentedSlice::new(self)
    }

    /// Returns a mutable slice view of the entire vector.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend(0..10);
    ///
    /// let mut slice = vec.as_mut_slice();
    /// slice[0] = 100;
    /// assert_eq!(vec[0], 100);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> SegmentedSliceMut<'_, T> {
        SegmentedSliceMut::new(self)
    }

    /// Returns an immutable slice view of a range of the vector.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend(0..10);
    ///
    /// let slice = vec.slice(2..5);
    /// assert_eq!(slice.len(), 3);
    /// assert_eq!(slice[0], 2);
    /// ```
    #[inline]
    pub fn slice(&self, range: std::ops::Range<usize>) -> SegmentedSlice<'_, T> {
        assert!(range.start <= range.end && range.end <= self.len);
        SegmentedSlice::from_range(self, range.start, range.end)
    }

    /// Returns a mutable slice view of a range of the vector.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend(0..10);
    ///
    /// let mut slice = vec.slice_mut(2..5);
    /// slice[0] = 100;
    /// assert_eq!(vec[2], 100);
    /// ```
    #[inline]
    pub fn slice_mut(&mut self, range: std::ops::Range<usize>) -> SegmentedSliceMut<'_, T> {
        assert!(range.start <= range.end && range.end <= self.len);
        SegmentedSliceMut::from_range(self, range.start, range.end)
    }

    /// Extends the vector by cloning elements from a slice.
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        if other.is_empty() {
            return;
        }
        self.reserve(other.len());

        let mut src = other;

        // Fill dynamic segments
        while !src.is_empty() {
            let (shelf, box_idx, remaining) = Self::location_with_capacity(self.len);
            let to_copy = src.len().min(remaining);
            let dest = unsafe { (*self.dynamic_segments.get_unchecked(shelf)).add(box_idx) };
            for (i, item) in src.iter().take(to_copy).enumerate() {
                unsafe { std::ptr::write(dest.add(i), item.clone()) };
            }
            self.len += to_copy;
            src = &src[to_copy..];
        }

        // Set up write pointer cache for next push
        if self.len > 0 {
            let biased = self.len + Self::MIN_NON_ZERO_CAP;
            let msb = biased.ilog2();
            let idx = (msb - Self::MIN_CAP_EXP) as usize;
            self.active_segment_index = idx;

            let capacity = 1usize << msb;
            let offset = biased ^ capacity;
            unsafe {
                let base = *self.dynamic_segments.get_unchecked(idx);
                self.write_ptr = base.add(offset);
                self.segment_end = base.add(capacity);
            }
        }
    }

    /// Extends the vector by copying elements from a slice (for `Copy` types).
    ///
    /// This is more efficient than `extend_from_slice` for `Copy` types
    /// as it uses bulk memory copy operations.
    pub fn extend_from_copy_slice(&mut self, other: &[T])
    where
        T: Copy,
    {
        if other.is_empty() {
            return;
        }
        self.reserve(other.len());

        let mut src = other;

        // Fill dynamic segments
        while !src.is_empty() {
            let (shelf, box_idx, remaining) = Self::location_with_capacity(self.len);
            let to_copy = src.len().min(remaining);
            unsafe {
                let dest = (*self.dynamic_segments.get_unchecked(shelf)).add(box_idx);
                std::ptr::copy_nonoverlapping(src.as_ptr(), dest, to_copy);
            };
            self.len += to_copy;
            src = &src[to_copy..];
        }

        // Set up write pointer cache for next push
        if self.len > 0 {
            let biased = self.len + Self::MIN_NON_ZERO_CAP;
            let msb = biased.ilog2();
            let idx = (msb - Self::MIN_CAP_EXP) as usize;
            self.active_segment_index = idx;

            let capacity = 1usize << msb;
            let offset = biased ^ capacity;
            unsafe {
                let base = *self.dynamic_segments.get_unchecked(idx);
                self.write_ptr = base.add(offset);
                self.segment_end = base.add(capacity);
            }
        }
    }

    /// Sorts the vector in place using a stable sorting algorithm.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(n * log(n)) worst-case.
    ///
    /// The algorithm is a merge sort adapted from the Rust standard library.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([3, 1, 4, 1, 5, 9, 2, 6]);
    /// vec.sort();
    /// assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
    /// ```
    #[inline]
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().sort();
    }

    /// Sorts the vector in place with a comparator function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(n * log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([3, 1, 4, 1, 5, 9, 2, 6]);
    /// vec.sort_by(|a, b| b.cmp(a)); // reverse order
    /// assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![9, 6, 5, 4, 3, 2, 1, 1]);
    /// ```
    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.as_mut_slice().sort_by(compare);
    }

    /// Sorts the vector in place with a key extraction function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and O(m * n * log(n)) worst-case,
    /// where the key function is O(m).
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([-3, 1, -4, 1, 5, -9, 2, 6]);
    /// vec.sort_by_key(|k| k.abs());
    /// assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![1, 1, 2, -3, -4, 5, 6, -9]);
    /// ```
    #[inline]
    pub fn sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.as_mut_slice().sort_by_key(f);
    }

    /// Sorts the vector in place using an unstable sorting algorithm.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place, and O(n * log(n)) worst-case.
    ///
    /// The algorithm is a quicksort with heapsort fallback, adapted from the Rust standard library.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([3, 1, 4, 1, 5, 9, 2, 6]);
    /// vec.sort_unstable();
    /// assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
    /// ```
    #[inline]
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.as_mut_slice().sort_unstable();
    }

    /// Sorts the vector in place with a comparator function using an unstable sorting algorithm.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place, and O(n * log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([3, 1, 4, 1, 5, 9, 2, 6]);
    /// vec.sort_unstable_by(|a, b| b.cmp(a)); // reverse order
    /// assert_eq!(vec.iter().copied().collect::<Vec<_>>(), vec![9, 6, 5, 4, 3, 2, 1, 1]);
    /// ```
    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.as_mut_slice().sort_unstable_by(compare);
    }

    /// Sorts the vector in place with a key extraction function using an unstable sorting algorithm.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place, and O(n * log(n)) worst-case,
    /// where the key function is O(m).
    ///
    /// # Examples
    ///
    /// ```
    /// use segmented_vec::SegmentedVec;
    ///
    /// let mut vec: SegmentedVec<i32> = SegmentedVec::new();
    /// vec.extend([-3, 1, -4, 1, 5, -9, 2, 6]);
    /// vec.sort_unstable_by_key(|k| k.abs());
    /// // Note: unstable sort may reorder equal elements, so we just check it's sorted by abs value
    /// let result: Vec<i32> = vec.iter().copied().collect();
    /// for i in 1..result.len() {
    ///     assert!(result[i-1].abs() <= result[i].abs());
    /// }
    /// ```
    #[inline]
    pub fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.as_mut_slice().sort_unstable_by_key(f);
    }

    /// Checks if the elements of this vector are sorted.
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.as_slice().is_sorted()
    }

    /// Checks if the elements of this vector are sorted using the given comparator function.
    pub fn is_sorted_by<F>(&self, compare: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        self.as_slice().is_sorted_by(compare)
    }

    /// Checks if the elements of this vector are sorted using the given key extraction function.
    pub fn is_sorted_by_key<K, F>(&self, f: F) -> bool
    where
        F: FnMut(&T) -> K,
        K: PartialOrd,
    {
        self.as_slice().is_sorted_by_key(f)
    }

    /// Returns the index of the partition point according to the given predicate.
    pub fn partition_point<P>(&self, pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.as_slice().partition_point(pred)
    }

    /// Rotates the vector in-place such that the first `mid` elements move to the end.
    pub fn rotate_left(&mut self, mid: usize) {
        self.as_mut_slice().rotate_left(mid);
    }

    /// Rotates the vector in-place such that the last `k` elements move to the front.
    pub fn rotate_right(&mut self, k: usize) {
        self.as_mut_slice().rotate_right(k);
    }

    /// Creates a new `SegmentedVec` with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
    }

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len);
        self.push(element);
        // Rotate the new element into place
        if index < self.len - 1 {
            for i in (index..self.len - 1).rev() {
                unsafe {
                    let ptr_a = self.unchecked_at_mut(i) as *mut T;
                    let ptr_b = self.unchecked_at_mut(i + 1) as *mut T;
                    std::ptr::swap(ptr_a, ptr_b);
                }
            }
        }
    }

    /// Removes and returns the element at position `index`, shifting all elements after it to the left.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        // Shift elements left
        for i in index..self.len - 1 {
            unsafe {
                let ptr_a = self.unchecked_at_mut(i) as *mut T;
                let ptr_b = self.unchecked_at_mut(i + 1) as *mut T;
                std::ptr::swap(ptr_a, ptr_b);
            }
        }
        self.pop().unwrap()
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
        let last_index = self.len - 1;
        if index != last_index {
            unsafe {
                let ptr_a = self.unchecked_at_mut(index) as *mut T;
                // Can't use write_ptr.sub(1) here because write_ptr might be at
                // a segment boundary (start of segment), in which case the last
                // element is in the previous segment.
                let ptr_b = self.unchecked_at_mut(last_index) as *mut T;
                std::ptr::swap(ptr_a, ptr_b);
            }
        }
        self.pop().unwrap()
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// Removes all elements for which `f(&element)` returns `false`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut i = 0;
        while i < self.len {
            if f(unsafe { self.unchecked_at(i) }) {
                i += 1;
            } else {
                self.remove(i);
            }
        }
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference.
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let mut i = 0;
        while i < self.len {
            if f(unsafe { self.unchecked_at_mut(i) }) {
                i += 1;
            } else {
                self.remove(i);
            }
        }
    }

    /// Removes consecutive duplicate elements.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes consecutive elements that satisfy the given equality relation.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        if self.len <= 1 {
            return;
        }
        let mut write = 1;
        for read in 1..self.len {
            let should_keep = unsafe {
                let prev_ptr = self.unchecked_at_mut(write - 1) as *mut T;
                let curr_ptr = self.unchecked_at_mut(read) as *mut T;
                !same_bucket(&mut *prev_ptr, &mut *curr_ptr)
            };
            if should_keep {
                if read != write {
                    unsafe {
                        let ptr_src = self.unchecked_at_mut(read) as *mut T;
                        let ptr_dst = self.unchecked_at_mut(write) as *mut T;
                        std::ptr::swap(ptr_dst, ptr_src);
                    }
                }
                write += 1;
            } else {
                // Drop the duplicate
                unsafe {
                    std::ptr::drop_in_place(self.unchecked_at_mut(read));
                }
            }
        }
        self.len = write;
    }

    /// Removes consecutive elements that map to the same key.
    pub fn dedup_by_key<K, F>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by the difference,
    /// with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the vector is simply truncated.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            while self.len < new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the vector is extended by the difference,
    /// with each additional slot filled with the result of calling the closure `f`.
    pub fn resize_with<F>(&mut self, new_len: usize, mut f: F)
    where
        F: FnMut() -> T,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            while self.len < new_len {
                self.push(f());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        let other_len = other.len;
        self.reserve(other_len);
        let start = self.len;
        while let Some(item) = other.pop() {
            self.push(item);
        }
        // Reverse the appended portion since pop() returns in reverse order
        let mut left = start;
        let mut right = self.len;
        while left < right {
            right -= 1;
            if left < right {
                unsafe {
                    let ptr_a = self.unchecked_at_mut(left) as *mut T;
                    let ptr_b = self.unchecked_at_mut(right) as *mut T;
                    std::ptr::swap(ptr_a, ptr_b);
                }
                left += 1;
            }
        }
    }

    /// Splits the vector into two at the given index.
    ///
    /// Returns a newly allocated vector containing the elements in the range `[at, len)`.
    /// After the call, the original vector will contain elements `[0, at)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len);
        let mut other = Self::new();
        other.reserve(self.len - at);
        for i in at..self.len {
            other.push(unsafe { self.unchecked_read(i) });
        }
        self.len = at;
        other
    }

    /// Returns an iterator over `chunk_size` elements of the vector at a time.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        self.as_slice().chunks(chunk_size)
    }

    /// Returns an iterator over all contiguous windows of length `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn windows(&self, size: usize) -> Windows<'_, T> {
        self.as_slice().windows(size)
    }

    /// Returns an iterator over `chunk_size` elements at a time, starting from the end.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'_, T> {
        self.as_slice().rchunks(chunk_size)
    }

    /// Returns an iterator over exactly `chunk_size` elements at a time.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        self.as_slice().chunks_exact(chunk_size)
    }

    /// Creates a draining iterator that removes the specified range and yields the removed items.
    ///
    /// # Panics
    ///
    /// Panics if the starting point is greater than the end point or if the end point is greater than the length.
    pub fn drain(&mut self, range: std::ops::Range<usize>) -> Drain<'_, T> {
        assert!(range.start <= range.end && range.end <= self.len);
        Drain {
            vec: self,
            range_start: range.start,
            range_end: range.end,
            index: range.start,
        }
    }

    /// Copies the elements to a new `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    // --- Internal helper methods ---

    /// Calculate the number of shelves needed for a given capacity.
    #[inline]
    fn shelf_count(box_count: usize) -> u32 {
        if box_count == 0 {
            return 0;
        }
        let val = box_count + Self::MIN_NON_ZERO_CAP;
        val.next_power_of_two().trailing_zeros() - Self::MIN_CAP_EXP
    }

    /// Calculate the size of a shelf at a given index.
    #[inline]
    fn shelf_size(shelf_index: u32) -> usize {
        Self::MIN_NON_ZERO_CAP << shelf_index
    }

    /// Calculate which shelf and box index a list index falls into.
    /// Returns (shelf_index, box_index).
    #[inline]
    fn location(list_index: usize) -> (usize, usize) {
        let biased = list_index + Self::MIN_NON_ZERO_CAP;
        let msb = biased.ilog2();
        let shelf = msb - Self::MIN_CAP_EXP;
        // Clear the most significant bit to get box_index
        let box_idx = biased ^ (1usize << msb);
        (shelf as usize, box_idx)
    }

    /// Calculate shelf, box index, and remaining capacity in one go.
    /// Returns (shelf_index, box_index, segment_remaining).
    #[inline]
    fn location_with_capacity(list_index: usize) -> (usize, usize, usize) {
        let biased = list_index + Self::MIN_NON_ZERO_CAP;
        let msb = biased.ilog2();
        let shelf = msb - Self::MIN_CAP_EXP;
        let box_idx = biased ^ (1usize << msb);
        // segment_remaining = shelf_size - box_idx = (1 << msb) - box_idx
        //                   = (1 << msb) - (biased - (1 << msb))
        //                   = (2 << msb) - biased
        let segment_remaining = (2usize << msb) - biased;
        (shelf as usize, box_idx, segment_remaining)
    }

    /// Get an unchecked reference to an element.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn unchecked_at(&self, index: usize) -> &T {
        unsafe {
            let (shelf, box_idx) = Self::location(index);
            &*(*self.dynamic_segments.get_unchecked(shelf)).add(box_idx)
        }
    }

    /// Get an unchecked mutable reference to an element.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`.
    #[inline]
    pub(crate) unsafe fn unchecked_at_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            let (shelf, box_idx) = Self::location(index);
            &mut *(*self.dynamic_segments.get_unchecked(shelf)).add(box_idx)
        }
    }

    /// Read a value from an index.
    ///
    /// # Safety
    ///
    /// `index` must be less than `self.len`, and the value must not be read again.
    #[inline]
    unsafe fn unchecked_read(&self, index: usize) -> T {
        unsafe {
            let (shelf, box_idx) = Self::location(index);
            std::ptr::read((*self.dynamic_segments.get_unchecked(shelf)).add(box_idx))
        }
    }

    fn grow_once(&mut self) {
        assert!(
            self.segment_count < MAX_SEGMENTS,
            "Maximum segment count exceeded"
        );

        let size = Self::shelf_size(self.segment_count as u32);
        let layout = Layout::array::<T>(size).expect("Layout overflow");
        let ptr = if layout.size() == 0 {
            std::ptr::dangling_mut::<T>()
        } else {
            let ptr = unsafe { alloc::alloc(layout) };
            if ptr.is_null() {
                panic!("Allocation failed");
            }
            ptr as *mut T
        };
        self.dynamic_segments[self.segment_count] = ptr;
        self.segment_count += 1;
    }

    /// Grow capacity to accommodate at least `new_capacity` elements.
    fn grow_capacity(&mut self, new_capacity: usize) {
        let new_shelf_count = Self::shelf_count(new_capacity) as usize;
        let old_shelf_count = self.segment_count;

        if new_shelf_count > old_shelf_count {
            assert!(
                new_shelf_count <= MAX_SEGMENTS,
                "Maximum segment count exceeded"
            );

            for i in old_shelf_count..new_shelf_count {
                let size = Self::shelf_size(i as u32);
                let layout = Layout::array::<T>(size).expect("Layout overflow");
                let ptr = if layout.size() == 0 {
                    std::ptr::dangling_mut::<T>()
                } else {
                    let ptr = unsafe { alloc::alloc(layout) };
                    if ptr.is_null() {
                        panic!("Allocation failed");
                    }
                    ptr as *mut T
                };
                self.dynamic_segments[i] = ptr;
            }
            self.segment_count = new_shelf_count;
        }
    }

    /// Compute total capacity given the number of dynamic shelves.
    #[inline]
    fn compute_capacity(shelf_count: u32) -> usize {
        (Self::MIN_NON_ZERO_CAP << shelf_count) - Self::MIN_NON_ZERO_CAP
    }

    /// Shrink capacity to the minimum needed for `new_capacity` elements.
    fn shrink_capacity(&mut self, new_capacity: usize) {
        let new_shelf_count = Self::shelf_count(new_capacity);
        let old_shelf_count = self.segment_count as u32;

        if new_shelf_count < old_shelf_count {
            self.free_shelves(old_shelf_count, new_shelf_count);
            self.segment_count = new_shelf_count as usize;
        }
    }

    /// Free shelves from `from_count` down to `to_count` (exclusive).
    fn free_shelves(&mut self, from_count: u32, to_count: u32) {
        for i in (to_count..from_count).rev() {
            let size = Self::shelf_size(i);
            let layout = Layout::array::<T>(size).expect("Layout overflow");
            if layout.size() > 0 {
                unsafe {
                    alloc::dealloc(self.dynamic_segments[i as usize] as *mut u8, layout);
                }
            }
            self.dynamic_segments[i as usize] = std::ptr::null_mut();
        }
    }
}

impl<T> Drop for SegmentedVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        self.clear();
        // Free all dynamic segments
        self.free_shelves(self.segment_count as u32, 0);
    }
}

impl<T> sort::IndexedAccess<T> for SegmentedVec<T> {
    #[inline]
    fn get_ref(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        unsafe { self.unchecked_at(index) }
    }

    #[inline]
    fn get_ptr(&self, index: usize) -> *const T {
        debug_assert!(index < self.len);
        let (shelf, box_idx) = Self::location(index);
        unsafe { (*self.dynamic_segments.get_unchecked(shelf)).add(box_idx) }
    }

    #[inline]
    fn get_ptr_mut(&mut self, index: usize) -> *mut T {
        debug_assert!(index < self.len);
        let (shelf, box_idx) = Self::location(index);
        unsafe { (*self.dynamic_segments.get_unchecked(shelf)).add(box_idx) }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        debug_assert!(a < self.len && b < self.len);
        unsafe {
            let ptr_a = self.get_ptr_mut(a);
            let ptr_b = self.get_ptr_mut(b);
            std::ptr::swap(ptr_a, ptr_b);
        }
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

impl<T: Clone> Clone for SegmentedVec<T> {
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self::new();
        }

        let mut new_vec = Self::new();
        new_vec.reserve(self.len);

        let mut remaining = self.len;
        for shelf in 0..self.segment_count {
            let size = Self::shelf_size(shelf as u32);
            let count = remaining.min(size);
            let src = unsafe { *self.dynamic_segments.get_unchecked(shelf) };
            let dst = unsafe { *new_vec.dynamic_segments.get_unchecked(shelf) };
            for i in 0..count {
                unsafe { std::ptr::write(dst.add(i), (*src.add(i)).clone()) };
            }
            new_vec.len += count;
            remaining -= count;
            if remaining == 0 {
                break;
            }
        }

        // Set up write pointer
        if new_vec.len > 0 {
            let biased = new_vec.len + Self::MIN_NON_ZERO_CAP;
            let msb = biased.ilog2();
            let idx = (msb - Self::MIN_CAP_EXP) as usize;
            new_vec.active_segment_index = idx;

            let capacity = 1usize << msb;
            let offset = biased ^ capacity;
            unsafe {
                let base = *new_vec.dynamic_segments.get_unchecked(idx);
                new_vec.write_ptr = base.add(offset);
                new_vec.segment_end = base.add(capacity);
            }
        }

        new_vec
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SegmentedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq> PartialEq for SegmentedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            if unsafe { self.unchecked_at(i) } != unsafe { other.unchecked_at(i) } {
                return false;
            }
        }
        true
    }
}

impl<T: Eq> Eq for SegmentedVec<T> {}

impl<T> FromIterator<T> for SegmentedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut vec = Self::new();
        vec.reserve(lower);
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<T> Extend<T> for SegmentedVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        for item in iter {
            self.push(item);
        }
    }
}

// --- Iterator implementations ---

/// An iterator over references to elements of a `SegmentedVec`.
pub struct Iter<'a, T> {
    vec: &'a SegmentedVec<T>,
    /// Current pointer within segment
    ptr: *const T,
    /// End of current segment (min of segment capacity and vec.len)
    segment_end: *const T,
    /// Current logical index
    index: usize,
    /// Current shelf index (for dynamic segments)
    shelf_index: u32,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr < self.segment_end {
            let result = unsafe { &*self.ptr };
            self.ptr = unsafe { self.ptr.add(1) };
            self.index += 1;
            return Some(result);
        }
        self.next_segment()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T> Iter<'a, T> {
    #[inline]
    fn next_segment(&mut self) -> Option<&'a T> {
        if self.index >= self.vec.len {
            return None;
        }

        let shelf_idx = self.shelf_index as usize;
        let shelf_size = SegmentedVec::<T>::shelf_size(self.shelf_index);
        let ptr = self.vec.dynamic_segments[shelf_idx];
        let segment_len = shelf_size.min(self.vec.len - self.index);
        self.ptr = ptr;
        self.segment_end = unsafe { ptr.add(segment_len) };
        self.shelf_index += 1;

        let result = unsafe { &*self.ptr };
        self.ptr = unsafe { self.ptr.add(1) };
        self.index += 1;
        Some(result)
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

/// An iterator over mutable references to elements of a `SegmentedVec`.
pub struct IterMut<'a, T> {
    vec: &'a mut SegmentedVec<T>,
    /// Current pointer within segment
    ptr: *mut T,
    /// End of current segment (min of segment capacity and vec.len)
    segment_end: *mut T,
    /// Current logical index
    index: usize,
    /// Current shelf index (for dynamic segments)
    shelf_index: u32,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr < self.segment_end {
            let result = self.ptr;
            self.ptr = unsafe { self.ptr.add(1) };
            self.index += 1;
            return Some(unsafe { &mut *result });
        }
        self.next_segment()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T> IterMut<'a, T> {
    #[inline]
    fn next_segment(&mut self) -> Option<&'a mut T> {
        if self.index >= self.vec.len {
            return None;
        }

        let shelf_idx = self.shelf_index as usize;
        let shelf_size = SegmentedVec::<T>::shelf_size(self.shelf_index);
        let ptr = self.vec.dynamic_segments[shelf_idx];
        let segment_len = shelf_size.min(self.vec.len - self.index);
        self.ptr = ptr;
        self.segment_end = unsafe { ptr.add(segment_len) };
        self.shelf_index += 1;

        let result = self.ptr;
        self.ptr = unsafe { self.ptr.add(1) };
        self.index += 1;
        Some(unsafe { &mut *result })
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

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

/// An owning iterator over elements of a `SegmentedVec`.
pub struct IntoIter<T> {
    vec: SegmentedVec<T>,
    index: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.len {
            return None;
        }
        let value = unsafe { self.vec.unchecked_read(self.index) };
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop remaining elements that weren't consumed
        for i in self.index..self.vec.len {
            unsafe {
                std::ptr::drop_in_place(self.vec.unchecked_at_mut(i));
            }
        }
        // Prevent the Vec from dropping elements again
        self.vec.len = 0;
    }
}

/// A draining iterator for `SegmentedVec`.
///
/// This struct is created by the `drain` method on `SegmentedVec`.
pub struct Drain<'a, T> {
    vec: &'a mut SegmentedVec<T>,
    range_start: usize,
    range_end: usize,
    index: usize,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.range_end {
            None
        } else {
            let value = unsafe { std::ptr::read(self.vec.unchecked_at(self.index)) };
            self.index += 1;
            Some(value)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.range_end - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.range_end {
            None
        } else {
            self.range_end -= 1;
            Some(unsafe { std::ptr::read(self.vec.unchecked_at(self.range_end)) })
        }
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // Drop any remaining elements in the range
        for i in self.index..self.range_end {
            unsafe {
                std::ptr::drop_in_place(self.vec.unchecked_at_mut(i));
            }
        }

        // Shift elements after the range to fill the gap
        let original_range_end = self.range_end;
        let original_len = self.vec.len;
        let drain_count = original_range_end - self.range_start;

        for i in 0..(original_len - original_range_end) {
            unsafe {
                let src = self.vec.unchecked_at(original_range_end + i) as *const T;
                let dst = self.vec.unchecked_at_mut(self.range_start + i) as *mut T;
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }
        }

        self.vec.len = original_len - drain_count;
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

        // Push many more elements to trigger segment allocations
        for i in 2..1000 {
            vec.push(i);
        }

        // The pointer should still be valid
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
    fn test_debug() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..3);
        let debug_str = format!("{:?}", vec);
        assert_eq!(debug_str, "[0, 1, 2]");
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
    fn test_drop_elements() {
        use std::cell::Cell;
        use std::rc::Rc;

        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter {
            counter: Rc<Cell<i32>>,
        }

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.counter.set(self.counter.get() + 1);
            }
        }

        {
            let mut vec: SegmentedVec<DropCounter> = SegmentedVec::new();
            for _ in 0..10 {
                vec.push(DropCounter {
                    counter: Rc::clone(&drop_count),
                });
            }
        }

        assert_eq!(drop_count.get(), 10);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..100);
        vec.truncate(5);
        vec.shrink_to_fit();
        assert_eq!(vec.len(), 5);
        for i in 0..5 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_sort() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
        vec.sort();
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_sort_empty() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.sort();
        assert!(vec.is_empty());
    }

    #[test]
    fn test_sort_single() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(42);
        vec.sort();
        assert_eq!(vec[0], 42);
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        vec.sort();
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_sort_reverse_sorted() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend((0..10).rev());
        vec.sort();
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_sort_by() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
        vec.sort_by(|a, b| b.cmp(a)); // reverse order
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_sort_by_key() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([-5, 2, -8, 1, -9, 3, -7, 4, -6, 0]);
        vec.sort_by_key(|k| k.abs());
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![0, 1, 2, 3, 4, -5, -6, -7, -8, -9]);
    }

    #[test]
    fn test_sort_stable() {
        // Test that sort is stable (equal elements maintain their relative order)
        #[derive(Debug, Clone, Eq, PartialEq)]
        struct Item {
            key: i32,
            order: usize,
        }

        let mut vec: SegmentedVec<Item> = SegmentedVec::new();
        vec.push(Item { key: 1, order: 0 });
        vec.push(Item { key: 2, order: 1 });
        vec.push(Item { key: 1, order: 2 });
        vec.push(Item { key: 2, order: 3 });
        vec.push(Item { key: 1, order: 4 });

        vec.sort_by_key(|item| item.key);

        // All items with key=1 should come first, in original order
        assert_eq!(vec[0], Item { key: 1, order: 0 });
        assert_eq!(vec[1], Item { key: 1, order: 2 });
        assert_eq!(vec[2], Item { key: 1, order: 4 });
        // Then items with key=2, in original order
        assert_eq!(vec[3], Item { key: 2, order: 1 });
        assert_eq!(vec[4], Item { key: 2, order: 3 });
    }

    #[test]
    fn test_sort_unstable() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
        vec.sort_unstable();
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_sort_unstable_by() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]);
        vec.sort_unstable_by(|a, b| b.cmp(a)); // reverse order
        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_sort_unstable_by_key() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([-5, 2, -8, 1, -9, 3, -7, 4, -6, 0]);
        vec.sort_unstable_by_key(|k| k.abs());
        // Just verify it's sorted by absolute value (unstable may reorder equal elements)
        let result: Vec<i32> = vec.iter().copied().collect();
        for i in 1..result.len() {
            assert!(result[i - 1].abs() <= result[i].abs());
        }
    }

    #[test]
    fn test_sort_large() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        // Add numbers in reverse order
        vec.extend((0..1000).rev());
        vec.sort();
        for i in 0..1000 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_sort_unstable_large() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        // Add numbers in reverse order
        vec.extend((0..1000).rev());
        vec.sort_unstable();
        for i in 0..1000 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn test_sort_pointers_remain_stable_after_sort() {
        // Verify that after sorting, pointers to elements are still valid
        // (they point to different values, but the memory locations are stable)
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 2, 8, 1, 9]);

        // Get pointer to first element before sort
        let ptr = &vec[0] as *const i32;

        vec.sort();

        // After sort, the pointer still points to valid memory (now contains sorted value)
        assert_eq!(unsafe { *ptr }, 1); // First element after sort is 1
    }

    // --- Slice tests ---

    #[test]
    fn test_as_slice() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert_eq!(slice.len(), 10);
        assert_eq!(slice[0], 0);
        assert_eq!(slice[9], 9);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        {
            let mut slice = vec.as_mut_slice();
            slice[0] = 100;
            slice[9] = 200;
        }

        assert_eq!(vec[0], 100);
        assert_eq!(vec[9], 200);
    }

    #[test]
    fn test_slice_range() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.slice(2..5);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 2);
        assert_eq!(slice[1], 3);
        assert_eq!(slice[2], 4);
    }

    #[test]
    fn test_slice_mut_range() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        {
            let mut slice = vec.slice_mut(2..5);
            slice[0] = 100;
            slice[2] = 200;
        }

        assert_eq!(vec[2], 100);
        assert_eq!(vec[4], 200);
    }

    #[test]
    fn test_slice_first_last() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert_eq!(slice.first(), Some(&0));
        assert_eq!(slice.last(), Some(&9));
    }

    #[test]
    fn test_slice_iter() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        let collected: Vec<i32> = slice.iter().copied().collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_slice_iter_rev() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        let collected: Vec<i32> = slice.iter().rev().copied().collect();
        assert_eq!(collected, (0..10).rev().collect::<Vec<_>>());
    }

    #[test]
    fn test_slice_contains() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert!(slice.contains(&5));
        assert!(!slice.contains(&100));
    }

    #[test]
    fn test_slice_binary_search() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..100);

        let slice = vec.as_slice();
        assert_eq!(slice.binary_search(&50), Ok(50));
        assert_eq!(slice.binary_search(&0), Ok(0));
        assert_eq!(slice.binary_search(&99), Ok(99));
        assert_eq!(slice.binary_search(&100), Err(100));
    }

    #[test]
    fn test_slice_split_at() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        let (left, right) = slice.split_at(5);
        assert_eq!(left.len(), 5);
        assert_eq!(right.len(), 5);
        assert_eq!(left[0], 0);
        assert_eq!(right[0], 5);
    }

    #[test]
    fn test_slice_to_vec() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        let converted = slice.to_vec();
        assert_eq!(converted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_slice_mut_sort() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend([5, 3, 1, 4, 2, 8, 7, 6, 0, 9]);

        // Sort only the middle part
        {
            let mut slice = vec.slice_mut(2..8);
            slice.sort();
        }

        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![5, 3, 1, 2, 4, 6, 7, 8, 0, 9]);
    }

    #[test]
    fn test_slice_mut_reverse() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        {
            let mut slice = vec.slice_mut(2..8);
            slice.reverse();
        }

        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![0, 1, 7, 6, 5, 4, 3, 2, 8, 9]);
    }

    #[test]
    fn test_slice_mut_fill() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        {
            let mut slice = vec.slice_mut(2..5);
            slice.fill(99);
        }

        let result: Vec<i32> = vec.iter().copied().collect();
        assert_eq!(result, vec![0, 1, 99, 99, 99, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_slice_starts_with() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert!(slice.starts_with(&[0, 1, 2]));
        assert!(!slice.starts_with(&[1, 2, 3]));
    }

    #[test]
    fn test_slice_ends_with() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert!(slice.ends_with(&[7, 8, 9]));
        assert!(!slice.ends_with(&[6, 7, 8]));
    }

    #[test]
    fn test_starts_with_edge_cases() {
        // Empty vector
        let empty: SegmentedVec<i32> = SegmentedVec::new();
        assert!(empty.starts_with(&[])); // Empty needle always matches
        assert!(!empty.starts_with(&[1])); // Non-empty needle on empty vec

        // Empty needle
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        assert!(vec.starts_with(&[])); // Empty needle always matches

        // Needle equals entire vector
        let full: Vec<i32> = (0..10).collect();
        assert!(vec.starts_with(&full));

        // Needle longer than vector
        let longer: Vec<i32> = (0..20).collect();
        assert!(!vec.starts_with(&longer));

        // Single element
        assert!(vec.starts_with(&[0]));
        assert!(!vec.starts_with(&[1]));
    }

    #[test]
    fn test_ends_with_edge_cases() {
        // Empty vector
        let empty: SegmentedVec<i32> = SegmentedVec::new();
        assert!(empty.ends_with(&[])); // Empty needle always matches
        assert!(!empty.ends_with(&[1])); // Non-empty needle on empty vec

        // Empty needle
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);
        assert!(vec.ends_with(&[])); // Empty needle always matches

        // Needle equals entire vector
        let full: Vec<i32> = (0..10).collect();
        assert!(vec.ends_with(&full));

        // Needle longer than vector
        let longer: Vec<i32> = (0..20).collect();
        assert!(!vec.ends_with(&longer));

        // Single element
        assert!(vec.ends_with(&[9]));
        assert!(!vec.ends_with(&[8]));
    }

    #[test]
    fn test_starts_with_across_segments() {
        // Create a vector that spans multiple segments
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..1000);

        // Test prefix within first segment
        assert!(vec.starts_with(&[0, 1, 2, 3, 4]));

        // Test prefix that spans segments (first segment is 4 elements)
        let prefix: Vec<i32> = (0..100).collect();
        assert!(vec.starts_with(&prefix));

        // Test wrong prefix
        assert!(!vec.starts_with(&[1, 2, 3]));
    }

    #[test]
    fn test_ends_with_across_segments() {
        // Create a vector that spans multiple segments
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..1000);

        // Test suffix within last segment
        assert!(vec.ends_with(&[997, 998, 999]));

        // Test suffix that spans segments
        let suffix: Vec<i32> = (900..1000).collect();
        assert!(vec.ends_with(&suffix));

        // Test wrong suffix
        assert!(!vec.ends_with(&[996, 997, 998]));
    }

    #[test]
    fn test_slice_eq() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend(0..10);

        let slice = vec.as_slice();
        assert_eq!(slice, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_min_non_zero_cap() {
        // For u8 (1 byte), first segment should be 8 elements
        let mut vec_u8: SegmentedVec<u8> = SegmentedVec::new();
        vec_u8.push(1);
        assert_eq!(vec_u8.capacity(), 8);

        // For i32 (4 bytes, <= 1024), first segment should be 4 elements
        let mut vec_i32: SegmentedVec<i32> = SegmentedVec::new();
        vec_i32.push(1);
        assert_eq!(vec_i32.capacity(), 4);

        // Verify indexing still works correctly with the larger first segment
        for i in 0u8..100 {
            vec_u8.push(i);
        }
        for i in 0..101 {
            assert_eq!(vec_u8[i], if i == 0 { 1 } else { (i - 1) as u8 });
        }
    }

    #[test]
    fn test_extend_from_copy_slice() {
        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        let data: Vec<i32> = (0..100).collect();
        vec.extend_from_copy_slice(&data);
        assert_eq!(vec.len(), 100);
        for i in 0..100 {
            assert_eq!(vec[i], i as i32);
        }

        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.push(999);
        vec.push(998);
        vec.extend_from_copy_slice(&data[..10]);
        assert_eq!(vec.len(), 12);
        assert_eq!(vec[0], 999);
        assert_eq!(vec[1], 998);
        for i in 0..10 {
            assert_eq!(vec[i + 2], i as i32);
        }

        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend_from_copy_slice(&[]);
        assert!(vec.is_empty());

        let mut vec: SegmentedVec<i32> = SegmentedVec::new();
        vec.extend_from_copy_slice(&[1, 2, 3]); // 3 elements, not at boundary
        assert_eq!(vec.len(), 3);
        vec.push(4); // Should use fast path, not crash in push_slow
        vec.push(5);
        vec.push(6);
        assert_eq!(vec.len(), 6);
        for i in 0..6 {
            assert_eq!(vec[i], (i + 1) as i32);
        }
    }
}
