use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::slice::{SegmentedSlice, SegmentedSliceMut};
use allocator_api2::alloc::Allocator;

/// A helper trait for generic indexing into [`SegmentedVec`] or [`SegmentedSlice`].
pub unsafe trait SliceIndex<T: ?Sized> {
    type Output<'a>
    where
        Self: 'a,
        T: 'a;
    type OutputMut<'a>
    where
        Self: 'a,
        T: 'a;

    fn get(self, container: &T) -> Option<Self::Output<'_>>;
    fn get_mut(self, container: &mut T) -> Option<Self::OutputMut<'_>>;

    unsafe fn get_unchecked(self, container: &T) -> Self::Output<'_>;
    unsafe fn get_unchecked_mut(self, container: &mut T) -> Self::OutputMut<'_>;

    fn index(self, container: &T) -> Self::Output<'_>;
    fn index_mut(self, container: &mut T) -> Self::OutputMut<'_>;
}

// --- usize ---

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>> for usize {
    type Output<'b>
        = &'a T
    where
        'a: 'b;
    type OutputMut<'b>
        = &'a mut T
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, _slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        None
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        &*slice.buf().ptr_at(slice.start + self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(
        self,
        _slice: &mut SegmentedSlice<'a, T, A>,
    ) -> Self::OutputMut<'a> {
        std::hint::unreachable_unchecked()
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("index out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSliceMut<'a, T, A>> for usize {
    type Output<'b>
        = &'a T
    where
        'a: 'b;
    type OutputMut<'b>
        = &'a mut T
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSliceMut<'a, T, A>) -> Option<Self::Output<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSliceMut<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSliceMut<'a, T, A>) -> Self::Output<'a> {
        &*slice.buf().ptr_at(slice.start + self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(
        self,
        slice: &mut SegmentedSliceMut<'a, T, A>,
    ) -> Self::OutputMut<'a> {
        &mut *slice.buf_mut().ptr_at(slice.start + self)
    }

    #[inline]
    fn index(self, slice: &SegmentedSliceMut<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("index out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSliceMut<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("index out of bounds")
    }
}

// --- Ranges ---

macro_rules! impl_range_index {
    ($SliceType:ident, $RangeType:ty, $MutRet:ty, $normalize:expr) => {
        unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<$SliceType<'a, T, A>> for $RangeType {
            type Output<'b>
                = SegmentedSlice<'a, T, A>
            where
                'a: 'b;
            type OutputMut<'b>
                = $MutRet
            where
                'a: 'b;

            #[inline]
            fn get(self, slice: &$SliceType<'a, T, A>) -> Option<Self::Output<'a>> {
                let range = $normalize(self.clone(), slice.len())?;
                unsafe { Some(self.get_unchecked(slice)) }
            }

            #[inline]
            fn get_mut(self, slice: &mut $SliceType<'a, T, A>) -> Option<Self::OutputMut<'a>> {
                let range = $normalize(self.clone(), slice.len())?;
                unsafe { Some(self.get_unchecked_mut(slice)) }
            }

            #[inline]
            unsafe fn get_unchecked(self, slice: &$SliceType<'a, T, A>) -> Self::Output<'a> {
                let range = $normalize(self, slice.len()).unwrap_unchecked();
                SegmentedSlice::new(
                    slice.buf(),
                    slice.start + range.start,
                    slice.start + range.end,
                )
            }

            #[inline]
            unsafe fn get_unchecked_mut(
                self,
                slice: &mut $SliceType<'a, T, A>,
            ) -> Self::OutputMut<'a> {
                let range = $normalize(self, slice.len()).unwrap_unchecked();
                <$MutRet>::new(
                    slice.buf(),
                    slice.start + range.start,
                    slice.start + range.end,
                )
            }

            #[inline]
            fn index(self, slice: &$SliceType<'a, T, A>) -> Self::Output<'a> {
                self.get(slice).expect("range out of bounds")
            }

            #[inline]
            fn index_mut(self, slice: &mut $SliceType<'a, T, A>) -> Self::OutputMut<'a> {
                self.get_mut(slice).expect("range out of bounds")
            }
        }
    };
}

// Range normalization helpers
#[inline]
fn range_bounds(range: Range<usize>, len: usize) -> Option<Range<usize>> {
    if range.start <= range.end && range.end <= len {
        Some(range)
    } else {
        None
    }
}
#[inline]
fn range_to(range: RangeTo<usize>, len: usize) -> Option<Range<usize>> {
    if range.end <= len {
        Some(0..range.end)
    } else {
        None
    }
}
#[inline]
fn range_from(range: RangeFrom<usize>, len: usize) -> Option<Range<usize>> {
    if range.start <= len {
        Some(range.start..len)
    } else {
        None
    }
}
#[inline]
fn range_full(_: RangeFull, len: usize) -> Option<Range<usize>> {
    Some(0..len)
}
#[inline]
fn range_inclusive(range: RangeInclusive<usize>, len: usize) -> Option<Range<usize>> {
    if *range.end() == usize::MAX {
        None
    } else {
        range_bounds(*range.start()..(*range.end() + 1), len)
    }
}
#[inline]
fn range_to_inclusive(range: RangeToInclusive<usize>, len: usize) -> Option<Range<usize>> {
    if range.end == usize::MAX {
        None
    } else {
        range_to(..range.end + 1, len)
    }
}

// Implementations for SegmentedSlice (immutable)
// MutRet is SegmentedSlice (immutable view)
impl_range_index!(
    SegmentedSlice,
    Range<usize>,
    SegmentedSlice<'a, T, A>,
    range_bounds
);
impl_range_index!(
    SegmentedSlice,
    RangeTo<usize>,
    SegmentedSlice<'a, T, A>,
    range_to
);
impl_range_index!(
    SegmentedSlice,
    RangeFrom<usize>,
    SegmentedSlice<'a, T, A>,
    range_from
);
impl_range_index!(
    SegmentedSlice,
    RangeFull,
    SegmentedSlice<'a, T, A>,
    range_full
);
impl_range_index!(
    SegmentedSlice,
    RangeInclusive<usize>,
    SegmentedSlice<'a, T, A>,
    range_inclusive
);
impl_range_index!(
    SegmentedSlice,
    RangeToInclusive<usize>,
    SegmentedSlice<'a, T, A>,
    range_to_inclusive
);

// Implementations for SegmentedSliceMut (mutable)
// MutRet is SegmentedSliceMut
impl_range_index!(
    SegmentedSliceMut,
    Range<usize>,
    SegmentedSliceMut<'a, T, A>,
    range_bounds
);
impl_range_index!(
    SegmentedSliceMut,
    RangeTo<usize>,
    SegmentedSliceMut<'a, T, A>,
    range_to
);
impl_range_index!(
    SegmentedSliceMut,
    RangeFrom<usize>,
    SegmentedSliceMut<'a, T, A>,
    range_from
);
impl_range_index!(
    SegmentedSliceMut,
    RangeFull,
    SegmentedSliceMut<'a, T, A>,
    range_full
);
impl_range_index!(
    SegmentedSliceMut,
    RangeInclusive<usize>,
    SegmentedSliceMut<'a, T, A>,
    range_inclusive
);
impl_range_index!(
    SegmentedSliceMut,
    RangeToInclusive<usize>,
    SegmentedSliceMut<'a, T, A>,
    range_to_inclusive
);
