//! Raw segment allocation management for `SegmentedVec`.
//!
//! This module handles low-level memory allocation for segmented vectors,
//! similar to how `RawVec` works for `Vec` in the standard library.

use std::alloc::{self, Layout};
use std::marker::PhantomData;

use crate::TryReserveError;

/// Maximum number of segments supported.
/// With exponentially growing segments, 64 segments can hold more than 2^64 elements.
pub(crate) const MAX_SEGMENTS: usize = 64;

/// Raw segmented vector that handles segment allocation without element management.
///
/// This is the low-level allocation primitive used by `SegmentedVec`.
/// It manages segment pointers and capacity but does not track element count
/// or handle element initialization/destruction.
#[repr(C)]
pub(crate) struct RawSegmentedVec<T> {
    /// Array of segment pointers
    segments: [*mut T; MAX_SEGMENTS],
    /// Number of allocated segments
    segment_count: usize,
    /// Marker for type ownership
    _marker: PhantomData<T>,
}

impl<T> RawSegmentedVec<T> {
    /// Minimum capacity for the first segment.
    /// Avoids tiny allocations that heap allocators round up anyway.
    /// - 8 for 1-byte elements (allocators round up small requests)
    /// - 4 for moderate elements (<= 1 KiB)
    /// - 1 for large elements (avoid wasting space)
    pub(crate) const MIN_SEGMENT_CAP: usize = {
        let size = std::mem::size_of::<T>();
        if size == 0 {
            // ZST: use 1 for consistent math, no actual allocation happens
            usize::MAX / 2 + 1
        } else if size == 1 {
            8
        } else if size <= 1024 {
            4
        } else {
            1
        }
    };

    /// Exponent of MIN_SEGMENT_CAP (log2)
    pub(crate) const MIN_CAP_EXP: u32 = if std::mem::size_of::<T>() == 0 {
        usize::BITS - 1
    } else {
        Self::MIN_SEGMENT_CAP.trailing_zeros()
    };

    /// Whether T is a zero-sized type
    const IS_ZST: bool = std::mem::size_of::<T>() == 0;

    /// Creates a new `RawSegmentedVec` without allocating.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            segments: [std::ptr::null_mut(); MAX_SEGMENTS],
            segment_count: 0,
            _marker: PhantomData,
        }
    }

    /// Returns the number of allocated segments.
    #[inline]
    pub(crate) const fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// Returns a pointer to the segment at the given index.
    ///
    /// # Safety
    ///
    /// `index` must be less than `segment_count`.
    #[inline]
    pub(crate) unsafe fn segment_ptr(&self, index: usize) -> *mut T {
        debug_assert!(index < self.segment_count);
        *self.segments.get_unchecked(index)
    }

    /// Returns the capacity of a segment at the given index.
    #[inline]
    pub(crate) const fn segment_capacity(index: usize) -> usize {
        if Self::IS_ZST {
            usize::MAX
        } else {
            Self::MIN_SEGMENT_CAP << index
        }
    }

    /// Computes total capacity given the number of segments.
    #[inline]
    pub(crate) const fn compute_capacity(segment_count: usize) -> usize {
        if Self::IS_ZST {
            if segment_count == 0 {
                0
            } else {
                usize::MAX
            }
        } else {
            (Self::MIN_SEGMENT_CAP << segment_count).wrapping_sub(Self::MIN_SEGMENT_CAP)
        }
    }

    /// Returns the total capacity across all allocated segments.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        Self::compute_capacity(self.segment_count)
    }

    /// Computes the number of segments needed to hold `element_count` elements.
    #[inline]
    pub(crate) fn segments_for_capacity(element_count: usize) -> usize {
        if element_count == 0 {
            return 0;
        }
        if Self::IS_ZST {
            return 1;
        }
        let biased = element_count.saturating_add(Self::MIN_SEGMENT_CAP - 1);
        let msb = biased.ilog2();
        (msb - Self::MIN_CAP_EXP + 1) as usize
    }

    /// Calculates which segment and offset a list index falls into.
    /// Returns (segment_index, offset_within_segment).
    #[inline]
    pub(crate) fn location(list_index: usize) -> (usize, usize) {
        if Self::IS_ZST {
            return (0, 0);
        }
        let biased = list_index + Self::MIN_SEGMENT_CAP;
        let msb = biased.ilog2();
        let segment = msb - Self::MIN_CAP_EXP;
        let offset = biased ^ (1usize << msb);
        (segment as usize, offset)
    }

    /// Allocates a single new segment.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or MAX_SEGMENTS is exceeded.
    pub(crate) fn grow_one(&mut self) {
        assert!(
            self.segment_count < MAX_SEGMENTS,
            "Maximum segment count exceeded"
        );

        if Self::IS_ZST {
            self.segments[self.segment_count] = std::ptr::NonNull::dangling().as_ptr();
            self.segment_count += 1;
            return;
        }

        let size = Self::segment_capacity(self.segment_count);
        let layout = Layout::array::<T>(size).expect("Layout overflow");

        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            panic!("Allocation failed");
        }

        self.segments[self.segment_count] = ptr as *mut T;
        self.segment_count += 1;
    }

    /// Ensures capacity for at least `needed_capacity` elements.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or capacity exceeds limits.
    pub(crate) fn reserve(&mut self, needed_capacity: usize) {
        if needed_capacity == 0 {
            return;
        }

        if Self::IS_ZST {
            if self.segment_count == 0 {
                self.segments[0] = std::ptr::NonNull::dangling().as_ptr();
                self.segment_count = 1;
            }
            return;
        }

        let needed_segments = Self::segments_for_capacity(needed_capacity);

        if needed_segments > self.segment_count {
            assert!(
                needed_segments <= MAX_SEGMENTS,
                "Maximum segment count exceeded"
            );

            for i in self.segment_count..needed_segments {
                let size = Self::segment_capacity(i);
                let layout = Layout::array::<T>(size).expect("Layout overflow");

                let ptr = unsafe { alloc::alloc(layout) };
                if ptr.is_null() {
                    panic!("Allocation failed");
                }

                self.segments[i] = ptr as *mut T;
            }
            self.segment_count = needed_segments;
        }
    }

    /// Tries to ensure capacity for at least `needed_capacity` elements.
    ///
    /// Returns `Err` if allocation fails or capacity would overflow.
    /// On error, any segments allocated during this call are freed.
    pub(crate) fn try_reserve(&mut self, needed_capacity: usize) -> Result<(), TryReserveError> {
        if needed_capacity == 0 {
            return Ok(());
        }

        if Self::IS_ZST {
            if self.segment_count == 0 {
                self.segments[0] = std::ptr::NonNull::dangling().as_ptr();
                self.segment_count = 1;
            }
            return Ok(());
        }

        let needed_segments = Self::segments_for_capacity(needed_capacity);
        let old_segment_count = self.segment_count;

        if needed_segments > old_segment_count {
            if needed_segments > MAX_SEGMENTS {
                return Err(TryReserveError::capacity_overflow());
            }

            for i in old_segment_count..needed_segments {
                let size = Self::segment_capacity(i);
                let layout =
                    Layout::array::<T>(size).map_err(|_| TryReserveError::capacity_overflow())?;

                let ptr = unsafe { alloc::alloc(layout) };
                if ptr.is_null() {
                    // Free any segments we allocated in this call
                    self.free_segments(i, old_segment_count);
                    return Err(TryReserveError::alloc_error(layout));
                }

                self.segments[i] = ptr as *mut T;
                self.segment_count = i + 1;
            }
        }

        Ok(())
    }

    /// Shrinks to hold at most `new_capacity` elements.
    ///
    /// Does not drop elements - caller must ensure elements beyond new capacity
    /// have already been dropped.
    pub(crate) fn shrink_to(&mut self, new_capacity: usize) {
        let new_segment_count = Self::segments_for_capacity(new_capacity);

        if new_segment_count < self.segment_count {
            self.free_segments(self.segment_count, new_segment_count);
            self.segment_count = new_segment_count;
        }
    }

    /// Frees segments from `from_count` down to `to_count` (exclusive).
    fn free_segments(&mut self, from_count: usize, to_count: usize) {
        if Self::IS_ZST {
            return;
        }

        for i in (to_count..from_count).rev() {
            let size = Self::segment_capacity(i);
            let layout = Layout::array::<T>(size).expect("Layout overflow");

            unsafe {
                alloc::dealloc(self.segments[i] as *mut u8, layout);
            }
            self.segments[i] = std::ptr::null_mut();
        }
    }

    /// Deallocates all segments without dropping elements.
    ///
    /// # Safety
    ///
    /// All elements must have been dropped before calling this.
    pub(crate) unsafe fn deallocate(&mut self) {
        self.free_segments(self.segment_count, 0);
        self.segment_count = 0;
    }

    /// Returns a raw pointer to an element at the given index.
    ///
    /// # Safety
    ///
    /// The index must be within allocated capacity.
    #[inline]
    pub(crate) unsafe fn ptr_at(&self, index: usize) -> *mut T {
        let (segment, offset) = Self::location(index);
        (*self.segments.get_unchecked(segment)).add(offset)
    }
}

impl<T> Drop for RawSegmentedVec<T> {
    fn drop(&mut self) {
        // Note: This only frees memory, it doesn't drop elements.
        // SegmentedVec must drop elements before RawSegmentedVec is dropped.
        unsafe {
            self.deallocate();
        }
    }
}

// Safety: RawSegmentedVec owns its allocations and T determines thread safety
unsafe impl<T: Send> Send for RawSegmentedVec<T> {}
unsafe impl<T: Sync> Sync for RawSegmentedVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let raw: RawSegmentedVec<i32> = RawSegmentedVec::new();
        assert_eq!(raw.segment_count(), 0);
        assert_eq!(raw.capacity(), 0);
    }

    #[test]
    fn test_grow_one() {
        let mut raw: RawSegmentedVec<i32> = RawSegmentedVec::new();
        raw.grow_one();
        assert_eq!(raw.segment_count(), 1);
        assert!(raw.capacity() >= 4);
    }

    #[test]
    fn test_reserve() {
        let mut raw: RawSegmentedVec<i32> = RawSegmentedVec::new();
        raw.reserve(100);
        assert!(raw.capacity() >= 100);
    }

    #[test]
    fn test_location() {
        // For i32, MIN_SEGMENT_CAP = 4
        // Segment 0: indices 0..4 (capacity 4)
        // Segment 1: indices 4..12 (capacity 8)
        // Segment 2: indices 12..28 (capacity 16)
        assert_eq!(RawSegmentedVec::<i32>::location(0), (0, 0));
        assert_eq!(RawSegmentedVec::<i32>::location(3), (0, 3));
        assert_eq!(RawSegmentedVec::<i32>::location(4), (1, 0));
        assert_eq!(RawSegmentedVec::<i32>::location(11), (1, 7));
        assert_eq!(RawSegmentedVec::<i32>::location(12), (2, 0));
    }

    #[test]
    fn test_shrink() {
        let mut raw: RawSegmentedVec<i32> = RawSegmentedVec::new();
        raw.reserve(100);
        let old_count = raw.segment_count();
        raw.shrink_to(10);
        assert!(raw.segment_count() < old_count);
        assert!(raw.capacity() >= 10);
    }

    #[test]
    fn test_zst() {
        let mut raw: RawSegmentedVec<()> = RawSegmentedVec::new();
        assert_eq!(raw.capacity(), 0);
        raw.grow_one();
        assert_eq!(raw.capacity(), usize::MAX);
    }
}
