//! Raw segment allocation management for `SegmentedVec`.
//!
//! This module handles low-level memory allocation for segmented vectors,
//! similar to how `RawVec` works for `Vec` in the standard library.

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::TryReserveError;

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
        usize::MAX
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
        return 0; // ZSTs don't need segments
    }
    let min_cap = min_segment_cap_for_size(elem_size);
    let min_cap_exp = min_cap_exp_for_size(elem_size);
    let biased = element_count.saturating_add(min_cap - 1);
    let msb = biased.ilog2();
    (msb - min_cap_exp + 1) as usize
}

/// Raw segmented vector that handles segment allocation without element management.
///
/// This is the low-level allocation primitive used by `SegmentedVec`.
/// It manages segment pointers and capacity but does not track element count
/// or handle element initialization/destruction.
#[repr(C)]
pub(crate) struct RawSegmentedVec<T, A: Allocator = Global> {
    inner: RawSegmentedVecInner<A>,
    _marker: PhantomData<T>,
}

#[repr(C)]
struct RawSegmentedVecInner<A: Allocator> {
    /// Pointer to the array of segment pointers.
    /// This is allocated using `A`, just like the segments themselves.
    segments: *mut *mut u8,
    /// Number of *allocated* segments (aka length of the backbone array).
    segment_count: usize,
    /// Capacity of the backbone array (how many segment pointers we can store before reallocating).
    segments_cap: usize,
    /// Allocator
    alloc: A,
}

impl<T> RawSegmentedVec<T, Global> {}

impl<T, A: Allocator> RawSegmentedVec<T, A> {
    #[inline]
    pub(crate) fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        // Optimization for ZSTs: capacity is infinite/irrelevant
        if std::mem::size_of::<T>() == 0 {
            return Self::new_in(alloc);
        }

        Self {
            inner: RawSegmentedVecInner::with_capacity_in(
                capacity,
                alloc,
                std::alloc::Layout::new::<T>(),
            ),
            _marker: PhantomData,
        }
    }

    /// Grows by one segment, returning (segment_ptr, segment_capacity).
    #[inline(never)]
    pub(crate) fn grow_one(&mut self) -> (*mut T, usize) {
        let (ptr, cap) = unsafe { self.inner.grow_one(std::alloc::Layout::new::<T>()) };
        (ptr as *mut T, cap)
    }

    /// Like `new`, but parameterized over the choice of allocator for
    /// the returned `RawVec`.
    #[inline]
    pub(crate) const fn new_in(alloc: A) -> Self {
        Self {
            inner: RawSegmentedVecInner::new_in(alloc),
            _marker: PhantomData,
        }
    }

    /// Like `try_with_capacity`, but parameterized over the choice of
    /// allocator for the returned `RawVec`.
    #[inline]
    pub(crate) fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Self, TryReserveError> {
        if std::mem::size_of::<T>() == 0 {
            return Ok(Self::new_in(alloc));
        }

        match RawSegmentedVecInner::try_with_capacity_in(
            capacity,
            alloc,
            std::alloc::Layout::new::<T>(),
        ) {
            Ok(inner) => Ok(Self {
                inner,
                _marker: PhantomData,
            }),
            Err(e) => Err(e),
        }
    }

    /// Gets the capacity of the allocation.
    ///
    /// This will always be `usize::MAX` if `T` is zero-sized.
    #[inline]
    pub(crate) const fn capacity(&self) -> usize {
        self.inner.capacity(std::mem::size_of::<T>())
    }

    /// Returns a shared reference to the allocator backing this `RawSegmentedVec`.
    #[inline]
    pub(crate) fn allocator(&self) -> &A {
        self.inner.allocator()
    }

    /// Ensures that the buffer contains at least enough space to hold `len +
    /// additional` elements. If it doesn't already have enough capacity, will
    /// allocate new segments as needed.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM.
    #[inline]
    pub(crate) fn reserve(&mut self, len: usize, additional: usize) -> Option<(*mut T, usize)> {
        unsafe {
            self.inner
                .reserve(len, additional, std::alloc::Layout::new::<T>())
                .map(|(ptr, cap)| (ptr.as_ptr() as *mut T, cap))
        }
    }

    /// The same as `reserve`, but returns on errors instead of panicking or aborting.
    /// The same as `reserve`, but returns on errors instead of panicking or aborting.
    pub(crate) fn try_reserve(
        &mut self,
        len: usize,
        additional: usize,
    ) -> Result<Option<(*mut T, usize)>, TryReserveError> {
        let ret = unsafe {
            self.inner
                .try_reserve(len, additional, std::alloc::Layout::new::<T>())?
        };
        Ok(ret.map(|(ptr, cap)| (ptr.as_ptr() as *mut T, cap)))
    }

    /// Shrinks the buffer down to the specified capacity. If the given amount
    /// is 0, actually completely deallocates.
    ///
    /// # Panics
    ///
    /// Panics if the given amount is *larger* than the current capacity.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM.
    #[inline]
    pub(crate) fn shrink_to_fit(&mut self, cap: usize) {
        unsafe {
            self.inner
                .shrink_to_fit(cap, std::alloc::Layout::new::<T>())
        }
    }

    /// Shrink to exactly `target_segments` count.
    ///
    /// # Safety
    /// `target_segments` must be <= current segment count.
    #[inline]
    pub(crate) unsafe fn shrink_to_fit_segments(&mut self, target_segments: usize) {
        self.inner
            .shrink_to_fit_segments(target_segments, std::alloc::Layout::new::<T>())
    }

    /// Returns the number of allocated segments.
    #[inline]
    pub(crate) fn segment_count(&self) -> usize {
        self.inner.segment_count
    }

    /// Returns a pointer to the start of segment at the given index.
    ///
    /// # Safety
    ///
    /// The segment at `index` must have been allocated.
    #[inline]
    pub(crate) unsafe fn segment_ptr(&self, index: usize) -> *mut T {
        if std::mem::size_of::<T>() == 0 {
            return std::ptr::NonNull::<T>::dangling().as_ptr();
        }
        debug_assert!(index < self.inner.segment_count);
        self.inner.segments.add(index).read() as *mut T
    }

    /// Returns the capacity of a segment at the given index.
    #[inline]
    pub(crate) fn segment_capacity(index: usize) -> usize {
        segment_capacity_for_size(index, std::mem::size_of::<T>())
    }

    /// Computes the segment index and offset within segment for a given logical index.
    /// Returns `(segment_index, offset_within_segment)`.
    #[inline]
    pub(crate) fn location(index: usize) -> (usize, usize) {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            // For ZSTs, everything is in segment 0
            return (0, index);
        }

        let min_cap = min_segment_cap_for_size(elem_size);
        let min_cap_exp = min_cap_exp_for_size(elem_size);

        // biased = index + min_cap gives us the position in a virtual array
        // where segment 0 starts at index min_cap
        let biased = index + min_cap;
        let segment_index = (biased.ilog2() - min_cap_exp) as usize;

        // The start of this segment in logical index space
        let segment_start = compute_capacity_for_size(segment_index, elem_size);
        let offset = index - segment_start;

        (segment_index, offset)
    }

    /// Returns a pointer to the element at the given logical index.
    ///
    /// # Safety
    ///
    /// The index must be within the allocated capacity.
    #[inline]
    pub(crate) unsafe fn ptr_at(&self, index: usize) -> *mut T {
        if std::mem::size_of::<T>() == 0 {
            return std::ptr::NonNull::<T>::dangling().as_ptr();
        }
        let (segment_idx, offset) = Self::location(index);
        debug_assert!(segment_idx < self.inner.segment_count);
        let segment_ptr = self.inner.segments.add(segment_idx).read() as *mut T;
        segment_ptr.add(offset)
    }
}

impl<T, A: Allocator> Drop for RawSegmentedVec<T, A> {
    /// Frees the memory owned by the `RawSegmentedVec` *without* trying to drop its contents.
    fn drop(&mut self) {
        unsafe { self.inner.deallocate(std::alloc::Layout::new::<T>()) }
    }
}

impl<A: Allocator> RawSegmentedVecInner<A> {
    #[inline]
    fn with_capacity_in(capacity: usize, alloc: A, elem_layout: Layout) -> Self {
        match Self::try_with_capacity_in(capacity, alloc, elem_layout) {
            Ok(this) => this,
            Err(_) => handle_alloc_error(elem_layout),
        }
    }

    /// Allocates segments for the given capacity, with optional zeroing.
    ///
    /// For segmented storage, zeroed allocation allocates segments using
    /// `allocate_zeroed` instead of `allocate`.
    fn try_allocate_in(
        capacity: usize,
        alloc: A,
        elem_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        let mut this = Self::new_in(alloc);
        let segments_needed = segments_for_capacity_inner(capacity, elem_layout.size());

        for _ in 0..segments_needed {
            if let Err(e) = unsafe { this.grow_one_inner_impl(elem_layout) } {
                // Free any already allocated segments/backbone to avoid leak
                unsafe {
                    this.deallocate(elem_layout);
                }
                return Err(e);
            }
        }

        Ok(this)
    }

    #[inline]
    const fn new_in(alloc: A) -> Self {
        Self {
            segments: std::ptr::null_mut(),
            segment_count: 0,
            segments_cap: 0,
            alloc,
        }
    }

    #[inline]
    fn try_with_capacity_in(
        capacity: usize,
        alloc: A,
        elem_layout: Layout,
    ) -> Result<Self, TryReserveError> {
        Self::try_allocate_in(capacity, alloc, elem_layout)
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    ///   Returns (segment_ptr, segment_capacity)
    #[inline]
    unsafe fn grow_one(&mut self, elem_layout: Layout) -> (*mut u8, usize) {
        match unsafe { self.grow_one_inner(elem_layout) } {
            Ok(result) => result,
            Err(_) => handle_alloc_error(elem_layout),
        }
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - The sum of `len` and `additional` must be greater than the current capacity
    ///   Returns the (ptr, cap) of the *first* newly allocated segment, if any.
    unsafe fn grow_amortized(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<Option<(NonNull<u8>, usize)>, TryReserveError> {
        // This is ensured by the calling contexts.
        debug_assert!(additional > 0);

        if elem_layout.size() == 0 {
            // Since we return a capacity of `usize::MAX` when `elem_size` is
            // 0, getting to here necessarily means the `RawSegmentedVec` is overfull,
            // UNLESS we haven't allocated any segments yet.
            if self.segment_count > 0 {
                return Err(TryReserveError::capacity_overflow());
            }
        }

        // Nothing we can really do about these checks, sadly.
        let required_cap = len
            .checked_add(additional)
            .ok_or_else(TryReserveError::capacity_overflow)?;

        // For segmented storage, we just need to ensure we have enough segments
        let segments_needed = segments_for_capacity_inner(required_cap, elem_layout.size());

        let mut first_alloc = None;
        while self.segment_count < segments_needed {
            unsafe {
                let (ptr, cap) = self.grow_one_inner(elem_layout)?;
                if first_alloc.is_none() {
                    first_alloc = Some((NonNull::new_unchecked(ptr), cap));
                }
            }
        }

        Ok(first_alloc)
    }

    /// Allocates one more segment, returning an error on failure.
    ///
    /// # Safety
    /// - `elem_layout` must be valid for `self`
    ///   Returns (segment_ptr, segment_capacity)
    unsafe fn grow_one_inner(
        &mut self,
        elem_layout: Layout,
    ) -> Result<(*mut u8, usize), TryReserveError> {
        unsafe { self.grow_one_inner_impl(elem_layout) }
    }

    /// Allocates one more segment with the specified initialization.
    /// # Safety
    /// - `elem_layout` must be valid for `self`
    unsafe fn grow_one_inner_impl(
        &mut self,
        elem_layout: Layout,
    ) -> Result<(*mut u8, usize), TryReserveError> {
        // Limit max segments to avoid overflow in capacity calculation
        if self.segment_count >= usize::BITS as usize {
            return Err(TryReserveError::capacity_overflow());
        }

        // Ensure we have space in the segments array
        if self.segment_count == self.segments_cap {
            let new_cap = if self.segments_cap == 0 {
                4 // Start with 4 segments to avoid small copies
            } else {
                self.segments_cap
                    .checked_mul(2)
                    .ok_or_else(TryReserveError::capacity_overflow)?
            };

            let backbone_layout = Layout::array::<*mut u8>(new_cap)
                .map_err(|_| TryReserveError::capacity_overflow())?;

            let new_segments = if self.segments_cap == 0 {
                self.alloc.allocate(backbone_layout)
            } else {
                let old_layout = Layout::array::<*mut u8>(self.segments_cap).unwrap();
                unsafe {
                    self.alloc.grow(
                        NonNull::new_unchecked(self.segments).cast(),
                        old_layout,
                        backbone_layout,
                    )
                }
            };

            let new_ptr = new_segments
                .map_err(|_| TryReserveError::alloc_error(backbone_layout))?
                .cast::<*mut u8>();
            self.segments = new_ptr.as_ptr();
            self.segments_cap = new_cap;
        }

        // For ZSTs, we use a dangling pointer and don't actually allocate
        if elem_layout.size() == 0 {
            let ptr = NonNull::<u8>::dangling().as_ptr();
            self.segments.add(self.segment_count).write(ptr);
            self.segment_count += 1;
            return Ok((ptr, usize::MAX));
        }

        let segment_cap = segment_capacity_for_size(self.segment_count, elem_layout.size());
        let layout = Layout::from_size_align(
            elem_layout
                .size()
                .checked_mul(segment_cap)
                .ok_or_else(TryReserveError::capacity_overflow)?,
            elem_layout.align(),
        )
        .map_err(|_| TryReserveError::capacity_overflow())?;

        let ptr = self
            .alloc
            .allocate(layout)
            .map_err(|_| TryReserveError::alloc_error(layout))?;

        let segment_ptr = ptr.as_ptr() as *mut u8;
        self.segments.add(self.segment_count).write(segment_ptr);
        self.segment_count += 1;
        Ok((segment_ptr, segment_cap))
    }

    #[inline]
    const fn capacity(&self, elem_size: usize) -> usize {
        compute_capacity_for_size(self.segment_count, elem_size)
    }

    #[inline]
    fn allocator(&self) -> &A {
        &self.alloc
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    #[inline]
    ///   Returns the (ptr, cap) of the *first* newly allocated segment, if any.
    pub(crate) unsafe fn reserve(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Option<(NonNull<u8>, usize)> {
        if self.needs_to_grow(len, additional, elem_layout) {
            match self.grow_amortized(len, additional, elem_layout) {
                Ok(res) => res, // Option<(NonNull<u8>, usize)>
                Err(_) => handle_alloc_error(elem_layout),
            }
        } else {
            None
        }
    }

    /// The same as `reserve`, but returns on errors instead of panicking or aborting.
    /// Returns the (ptr, cap) of the *first* newly allocated segment, if any.
    pub(crate) unsafe fn try_reserve(
        &mut self,
        len: usize,
        additional: usize,
        elem_layout: Layout,
    ) -> Result<Option<(NonNull<u8>, usize)>, TryReserveError> {
        if self.needs_to_grow(len, additional, elem_layout) {
            self.grow_amortized(len, additional, elem_layout)
        } else {
            Ok(None)
        }
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`
    #[inline]
    unsafe fn shrink_to_fit(&mut self, cap: usize, elem_layout: Layout) {
        if unsafe { self.shrink(cap, elem_layout) }.is_err() {
            handle_alloc_error(elem_layout);
        }
    }

    /// Shrink to exactly `target_segments` count.
    ///
    /// # Safety
    /// `target_segments` must be <= current segment count.
    #[inline]
    pub(crate) unsafe fn shrink_to_fit_segments(
        &mut self,
        target_segments: usize,
        elem_layout: Layout,
    ) {
        if unsafe { self.shrink_segments(target_segments, elem_layout) }.is_err() {
            handle_alloc_error(elem_layout);
        }
    }

    #[inline]
    const fn needs_to_grow(&self, len: usize, additional: usize, elem_layout: Layout) -> bool {
        additional > self.capacity(elem_layout.size()).wrapping_sub(len)
    }

    /// # Safety
    /// - `elem_layout` must be valid for `self`, i.e. it must be the same `elem_layout` used to
    ///   initially construct `self`
    /// - `elem_layout`'s size must be a multiple of its alignment
    /// - `cap` must be less than or equal to `self.capacity(elem_layout.size())`
    #[inline]
    unsafe fn shrink(&mut self, cap: usize, elem_layout: Layout) -> Result<(), TryReserveError> {
        let segments_needed = if cap == 0 {
            0
        } else {
            segments_for_capacity_inner(cap, elem_layout.size())
        };
        unsafe { self.shrink_segments(segments_needed, elem_layout) }
    }

    /// Internal logic for shrinking segments directly by count.
    unsafe fn shrink_segments(
        &mut self,
        segments_needed: usize,
        elem_layout: Layout,
    ) -> Result<(), TryReserveError> {
        // For ZSTs, don't deallocate segments (they're dangling pointers anyway)
        if elem_layout.size() == 0 {
            return Ok(());
        }

        // Deallocate excess segments
        while self.segment_count > segments_needed {
            self.segment_count -= 1;
            let segment_idx = self.segment_count;
            let segment_cap = segment_capacity_for_size(segment_idx, elem_layout.size());

            let layout = Layout::from_size_align_unchecked(
                elem_layout.size() * segment_cap,
                elem_layout.align(),
            );

            let ptr = self.segments.add(segment_idx).read();
            if let Some(ptr) = NonNull::new(ptr) {
                self.alloc.deallocate(ptr, layout);
            }
            self.segments.add(segment_idx).write(std::ptr::null_mut());
        }

        Ok(())
    }

    unsafe fn deallocate(&mut self, elem_layout: Layout) {
        // Free all allocated segments
        if elem_layout.size() != 0 {
            for i in 0..self.segment_count {
                let segment_cap = segment_capacity_for_size(i, elem_layout.size());
                let layout = Layout::from_size_align_unchecked(
                    elem_layout.size() * segment_cap,
                    elem_layout.align(),
                );
                let ptr = self.segments.add(i).read();
                if let Some(ptr) = NonNull::new(ptr) {
                    self.alloc.deallocate(ptr, layout);
                }
            }
        }

        if self.segments_cap > 0 {
            let backbone_layout = Layout::array::<*mut u8>(self.segments_cap).unwrap();
            self.alloc.deallocate(
                NonNull::new_unchecked(self.segments).cast(),
                backbone_layout,
            );
        }
    }
}

/// Handle allocation errors by panicking.
#[cold]
#[inline(never)]
fn handle_alloc_error(layout: Layout) -> ! {
    panic!("memory allocation of {} bytes failed", layout.size());
}
