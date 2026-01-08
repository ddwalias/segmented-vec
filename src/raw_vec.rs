//! Raw segment allocation management for `SegmentedVec`.
//!
//! This module handles low-level memory allocation for segmented vectors,
//! similar to how `RawVec` works for `Vec` in the standard library.

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

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
pub(crate) struct RawSegmentedVec<T, A: Allocator = Global> {
    /// Type-erased inner implementation
    inner: RawSegmentedVecInner<A>,
    /// Marker for type ownership
    _marker: PhantomData<T>,
}

/// Computes the minimum segment capacity exponent for a given element size.
/// Returns log2(min_segment_cap).
#[inline]
const fn min_cap_exp_for_size(elem_size: usize) -> u32 {
    if elem_size == 0 {
        usize::BITS - 1
    } else if elem_size == 1 {
        3 // 8 elements
    } else if elem_size <= 1024 {
        2 // 4 elements
    } else {
        0 // 1 element
    }
}

/// Computes the minimum segment capacity for a given element size.
#[inline]
const fn min_segment_cap_for_size(elem_size: usize) -> usize {
    if elem_size == 0 {
        usize::MAX / 2 + 1
    } else {
        1 << min_cap_exp_for_size(elem_size)
    }
}

/// Computes the capacity of a segment at a given index for elements of the given size.
#[inline]
const fn segment_capacity_for_size(index: usize, elem_size: usize) -> usize {
    if elem_size == 0 {
        usize::MAX
    } else {
        min_segment_cap_for_size(elem_size) << index
    }
}

/// Computes total capacity given the number of segments and element size.
#[inline]
const fn compute_capacity_for_size(segment_count: usize, elem_size: usize) -> usize {
    if elem_size == 0 {
        if segment_count == 0 {
            0
        } else {
            usize::MAX
        }
    } else {
        let min_cap = min_segment_cap_for_size(elem_size);
        (min_cap << segment_count).wrapping_sub(min_cap)
    }
}

/// Computes the number of segments needed to hold `element_count` elements.
#[inline]
fn segments_for_capacity_inner(element_count: usize, elem_size: usize) -> usize {
    if element_count == 0 {
        return 0;
    }
    if elem_size == 0 {
        return 1;
    }
    let min_cap = min_segment_cap_for_size(elem_size);
    let min_cap_exp = min_cap_exp_for_size(elem_size);
    let biased = element_count.saturating_add(min_cap - 1);
    let msb = biased.ilog2();
    (msb - min_cap_exp + 1) as usize
}

/// Type-erased inner implementation for RawSegmentedVec.
/// Stores segment pointers as `*mut u8` and takes element layout as parameter.
struct RawSegmentedVecInner<A> {
    /// Array of segment pointers (type-erased)
    segments: [*mut u8; MAX_SEGMENTS],
    /// Number of allocated segments
    segment_count: usize,
    /// Allocator
    alloc: A,
}

impl<A: Allocator> RawSegmentedVecInner<A> {
    /// Creates a new empty `RawSegmentedVecInner`.
    #[inline]
    const fn new_in(alloc: A) -> Self {
        Self {
            segments: [std::ptr::null_mut(); MAX_SEGMENTS],
            segment_count: 0,
            alloc,
        }
    }

    /// Returns the number of allocated segments.
    #[inline]
    const fn segment_count(&self) -> usize {
        self.segment_count
    }

    /// Returns a pointer to the segment at the given index.
    ///
    /// # Safety
    ///
    /// `index` must be less than `segment_count`.
    #[inline]
    unsafe fn segment_ptr(&self, index: usize) -> *mut u8 {
        debug_assert!(index < self.segment_count);
        *self.segments.get_unchecked(index)
    }

    /// Returns the total capacity across all allocated segments.
    #[inline]
    fn capacity(&self, elem_size: usize) -> usize {
        compute_capacity_for_size(self.segment_count, elem_size)
    }

    /// Allocates a single new segment.
    ///
    /// # Safety
    ///
    /// `elem_layout` must match the element type this vec is used for.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or MAX_SEGMENTS is exceeded.
    unsafe fn grow_one(&mut self, elem_layout: Layout) {
        assert!(
            self.segment_count < MAX_SEGMENTS,
            "Maximum segment count exceeded"
        );

        // ZST: no actual allocation needed
        if elem_layout.size() == 0 {
            self.segments[self.segment_count] = NonNull::dangling().as_ptr();
            self.segment_count += 1;
            return;
        }

        let size = segment_capacity_for_size(self.segment_count, elem_layout.size());
        let layout = Layout::from_size_align(size * elem_layout.size(), elem_layout.align())
            .expect("Layout overflow");

        let ptr = match self.alloc.allocate(layout) {
            Ok(ptr) => ptr.as_ptr() as *mut u8,
            Err(_) => panic!("Allocation failed"),
        };

        self.segments[self.segment_count] = ptr;
        self.segment_count += 1;
    }

    /// Ensures capacity for at least `needed_capacity` elements.
    ///
    /// # Safety
    ///
    /// `elem_layout` must match the element type this vec is used for.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or capacity exceeds limits.
    unsafe fn reserve(&mut self, needed_capacity: usize, elem_layout: Layout) {
        if needed_capacity == 0 {
            return;
        }

        // ZST: only need one segment ever
        if elem_layout.size() == 0 {
            if self.segment_count == 0 {
                self.segments[0] = NonNull::dangling().as_ptr();
                self.segment_count = 1;
            }
            return;
        }

        let needed_segments = segments_for_capacity_inner(needed_capacity, elem_layout.size());

        if needed_segments > self.segment_count {
            assert!(
                needed_segments <= MAX_SEGMENTS,
                "Maximum segment count exceeded"
            );

            for i in self.segment_count..needed_segments {
                let size = segment_capacity_for_size(i, elem_layout.size());
                let layout =
                    Layout::from_size_align(size * elem_layout.size(), elem_layout.align())
                        .expect("Layout overflow");

                let ptr = match self.alloc.allocate(layout) {
                    Ok(ptr) => ptr.as_ptr() as *mut u8,
                    Err(_) => panic!("Allocation failed"),
                };

                self.segments[i] = ptr;
            }
            self.segment_count = needed_segments;
        }
    }

    /// Tries to ensure capacity for at least `needed_capacity` elements.
    ///
    /// # Safety
    ///
    /// `elem_layout` must match the element type this vec is used for.
    ///
    /// Returns `Err` if allocation fails or capacity would overflow.
    /// On error, any segments allocated during this call are freed.
    unsafe fn try_reserve(
        &mut self,
        needed_capacity: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        if needed_capacity == 0 {
            return Ok(());
        }

        // ZST
        if elem_layout.size() == 0 {
            if self.segment_count == 0 {
                self.segments[0] = NonNull::dangling().as_ptr();
                self.segment_count = 1;
            }
            return Ok(());
        }

        let needed_segments = segments_for_capacity_inner(needed_capacity, elem_layout.size());
        let old_segment_count = self.segment_count;

        if needed_segments > old_segment_count {
            if needed_segments > MAX_SEGMENTS {
                return Err(TryReserveError::capacity_overflow());
            }

            for i in old_segment_count..needed_segments {
                let size = segment_capacity_for_size(i, elem_layout.size());
                let layout =
                    Layout::from_size_align(size * elem_layout.size(), elem_layout.align())
                        .map_err(|_| TryReserveError::capacity_overflow())?;

                let ptr = match self.alloc.allocate(layout) {
                    Ok(ptr) => ptr.as_ptr() as *mut u8,
                    Err(_) => {
                        // Free any segments we allocated in this call
                        self.free_segments(i, old_segment_count, elem_layout);
                        return Err(TryReserveError::alloc_error(layout));
                    }
                };

                self.segments[i] = ptr;
                self.segment_count = i + 1;
            }
        }

        Ok(())
    }

    /// Shrinks to hold at most `new_capacity` elements.
    ///
    /// # Safety
    ///
    /// `elem_layout` must match the element type.
    /// Caller must ensure elements beyond new capacity have already been dropped.
    unsafe fn shrink_to(&mut self, new_capacity: usize, elem_layout: Layout) {
        let new_segment_count = segments_for_capacity_inner(new_capacity, elem_layout.size());

        if new_segment_count < self.segment_count {
            self.free_segments(self.segment_count, new_segment_count, elem_layout);
            self.segment_count = new_segment_count;
        }
    }

    /// Frees segments from `from_count` down to `to_count` (exclusive).
    ///
    /// # Safety
    ///
    /// `elem_layout` must match the element type.
    unsafe fn free_segments(&mut self, from_count: usize, to_count: usize, elem_layout: Layout) {
        if elem_layout.size() == 0 {
            return;
        }

        for i in (to_count..from_count).rev() {
            let size = segment_capacity_for_size(i, elem_layout.size());
            let layout =
                Layout::from_size_align_unchecked(size * elem_layout.size(), elem_layout.align());

            if let Some(ptr) = NonNull::new(self.segments[i]) {
                self.alloc.deallocate(ptr, layout);
            }
            self.segments[i] = std::ptr::null_mut();
        }
    }

    /// Deallocates all segments without dropping elements.
    ///
    /// # Safety
    ///
    /// All elements must have been dropped before calling this.
    /// `elem_layout` must match the element type.
    unsafe fn deallocate(&mut self, elem_layout: Layout) {
        self.free_segments(self.segment_count, 0, elem_layout);
        self.segment_count = 0;
    }

    /// Returns a raw pointer to an element at the given index.
    ///
    /// # Safety
    ///
    /// The index must be within allocated capacity.
    /// `elem_layout` must match the element type.
    #[inline]
    unsafe fn ptr_at(&self, index: usize, elem_layout: Layout) -> *mut u8 {
        let (segment, offset) = Self::location(index, elem_layout.size());
        (*self.segments.get_unchecked(segment)).add(offset * elem_layout.size())
    }

    /// Calculates which segment and offset a list index falls into.
    /// Returns (segment_index, offset_within_segment).
    #[inline]
    fn location(list_index: usize, elem_size: usize) -> (usize, usize) {
        if elem_size == 0 {
            return (0, 0);
        }
        let min_cap = min_segment_cap_for_size(elem_size);
        let min_cap_exp = min_cap_exp_for_size(elem_size);
        let biased = list_index + min_cap;
        let msb = biased.ilog2();
        let segment = msb - min_cap_exp;
        let offset = biased ^ (1usize << msb);
        (segment as usize, offset)
    }
}

impl<T, A: Allocator> RawSegmentedVec<T, A> {
    /// Minimum capacity for the first segment.
    /// Avoids tiny allocations that heap allocators round up anyway.
    /// - 8 for 1-byte elements (allocators round up small requests)
    /// - 4 for moderate elements (<= 1 KiB)
    /// - 1 for large elements (avoid wasting space)
    pub(crate) const MIN_SEGMENT_CAP: usize = min_segment_cap_for_size(std::mem::size_of::<T>());

    /// Returns the layout for type T
    #[inline]
    const fn elem_layout() -> Layout {
        // SAFETY: Layout::new::<T>() is always valid
        unsafe {
            Layout::from_size_align_unchecked(std::mem::size_of::<T>(), std::mem::align_of::<T>())
        }
    }

    /// Returns the number of allocated segments.
    #[inline]
    pub(crate) const fn segment_count(&self) -> usize {
        self.inner.segment_count()
    }

    /// Returns a pointer to the segment at the given index.
    ///
    /// # Safety
    ///
    /// `index` must be less than `segment_count`.
    #[inline]
    pub(crate) unsafe fn segment_ptr(&self, index: usize) -> *mut T {
        self.inner.segment_ptr(index) as *mut T
    }

    /// Returns the capacity of a segment at the given index.
    #[inline]
    pub(crate) const fn segment_capacity(index: usize) -> usize {
        segment_capacity_for_size(index, std::mem::size_of::<T>())
    }

    /// Returns the total capacity across all allocated segments.
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.inner.capacity(std::mem::size_of::<T>())
    }

    /// Calculates which segment and offset a list index falls into.
    /// Returns (segment_index, offset_within_segment).
    #[inline]
    pub(crate) fn location(list_index: usize) -> (usize, usize) {
        RawSegmentedVecInner::<Global>::location(list_index, std::mem::size_of::<T>())
    }

    /// Allocates a single new segment.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or MAX_SEGMENTS is exceeded.
    pub(crate) fn grow_one(&mut self) {
        // SAFETY: elem_layout matches type T
        unsafe { self.inner.grow_one(Self::elem_layout()) }
    }

    /// Ensures capacity for at least `needed_capacity` elements.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails or capacity exceeds limits.
    pub(crate) fn reserve(&mut self, needed_capacity: usize) {
        // SAFETY: elem_layout matches type T
        unsafe { self.inner.reserve(needed_capacity, Self::elem_layout()) }
    }

    /// Tries to ensure capacity for at least `needed_capacity` elements.
    ///
    /// Returns `Err` if allocation fails or capacity would overflow.
    /// On error, any segments allocated during this call are freed.
    pub(crate) fn try_reserve(&mut self, needed_capacity: usize) -> Result<(), TryReserveError> {
        // SAFETY: elem_layout matches type T
        unsafe { self.inner.try_reserve(needed_capacity, Self::elem_layout()) }
    }

    /// Shrinks to hold at most `new_capacity` elements.
    ///
    /// Does not drop elements - caller must ensure elements beyond new capacity
    /// have already been dropped.
    pub(crate) fn shrink_to(&mut self, new_capacity: usize) {
        // SAFETY: elem_layout matches type T
        unsafe { self.inner.shrink_to(new_capacity, Self::elem_layout()) }
    }

    /// Returns a raw pointer to an element at the given index.
    ///
    /// # Safety
    ///
    /// The index must be within allocated capacity.
    #[inline]
    pub(crate) unsafe fn ptr_at(&self, index: usize) -> *mut T {
        self.inner.ptr_at(index, Self::elem_layout()) as *mut T
    }
}

impl<T> RawSegmentedVec<T> {
    /// Creates a new `RawSegmentedVec` without allocating.
    #[inline]
    pub(crate) const fn new() -> Self {
        Self {
            inner: RawSegmentedVecInner::new_in(Global),
            _marker: PhantomData,
        }
    }
}

impl<T, A: Allocator> Drop for RawSegmentedVec<T, A> {
    fn drop(&mut self) {
        // Note: This only frees memory, it doesn't drop elements.
        // SegmentedVec must drop elements before RawSegmentedVec is dropped.
        // SAFETY: We're dropping, so no more accesses will happen
        unsafe {
            self.inner.deallocate(Self::elem_layout());
        }
    }
}

// Safety: RawSegmentedVec owns its allocations and T determines thread safety
unsafe impl<T: Send, A: Allocator + Send> Send for RawSegmentedVec<T, A> {}
unsafe impl<T: Sync, A: Allocator + Sync> Sync for RawSegmentedVec<T, A> {}

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
