use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::slice::{SegmentedSlice, SegmentedSliceMut};
use crate::SegmentedVec; // Import SegmentedVec
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

unsafe impl<'a, T> SliceIndex<SegmentedSlice<'a, T>> for usize {
    type Output<'b>
        = &'a T
    where
        'a: 'b;
    type OutputMut<'b>
        = &'a mut T
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T>) -> Option<Self::Output<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, _slice: &mut SegmentedSlice<'a, T>) -> Option<Self::OutputMut<'a>> {
        None
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T>) -> Self::Output<'a> {
        &*slice.ptr_at(slice.start + self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, _slice: &mut SegmentedSlice<'a, T>) -> Self::OutputMut<'a> {
        std::hint::unreachable_unchecked()
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T>) -> Self::Output<'a> {
        self.get(slice).expect("index out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<'a, T> SliceIndex<SegmentedSliceMut<'a, T>> for usize {
    type Output<'b>
        = &'a T
    where
        'a: 'b;
    type OutputMut<'b>
        = &'a mut T
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSliceMut<'a, T>) -> Option<Self::Output<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSliceMut<'a, T>) -> Option<Self::OutputMut<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSliceMut<'a, T>) -> Self::Output<'a> {
        &*slice.ptr_at(slice.start + self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSliceMut<'a, T>) -> Self::OutputMut<'a> {
        &mut *slice.ptr_at(slice.start + self)
    }

    #[inline]
    fn index(self, slice: &SegmentedSliceMut<'a, T>) -> Self::Output<'a> {
        self.get(slice).expect("index out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSliceMut<'a, T>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("index out of bounds")
    }
}

// --- Ranges ---

macro_rules! impl_range_index {
    ($SliceType:ident, $RangeType:ty, $MutRet:ty, $normalize:expr) => {
        unsafe impl<'a, T> SliceIndex<$SliceType<'a, T>> for $RangeType {
            type Output<'b>
                = SegmentedSlice<'a, T>
            where
                'a: 'b;
            type OutputMut<'b>
                = $MutRet
            where
                'a: 'b;

            #[inline]
            fn get(self, slice: &$SliceType<'a, T>) -> Option<Self::Output<'a>> {
                if $normalize(self.clone(), slice.len()).is_none() {
                    return None;
                }
                unsafe { Some(self.get_unchecked(slice)) }
            }

            #[inline]
            fn get_mut(self, slice: &mut $SliceType<'a, T>) -> Option<Self::OutputMut<'a>> {
                if $normalize(self.clone(), slice.len()).is_none() {
                    return None;
                }
                unsafe { Some(self.get_unchecked_mut(slice)) }
            }

            #[inline]
            unsafe fn get_unchecked(self, slice: &$SliceType<'a, T>) -> Self::Output<'a> {
                let range = $normalize(self, slice.len()).unwrap_unchecked();
                SegmentedSlice::new(
                    slice.segments,
                    slice.start + range.start,
                    slice.start + range.end,
                )
            }

            #[inline]
            unsafe fn get_unchecked_mut(
                self,
                slice: &mut $SliceType<'a, T>,
            ) -> Self::OutputMut<'a> {
                let range = $normalize(self, slice.len()).unwrap_unchecked();
                <$MutRet>::new(
                    slice.segments,
                    slice.start + range.start,
                    slice.start + range.end,
                )
            }

            #[inline]
            fn index(self, slice: &$SliceType<'a, T>) -> Self::Output<'a> {
                self.get(slice).expect("range out of bounds")
            }

            #[inline]
            fn index_mut(self, slice: &mut $SliceType<'a, T>) -> Self::OutputMut<'a> {
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
    SegmentedSlice<'a, T>,
    range_bounds
);
impl_range_index!(
    SegmentedSlice,
    RangeTo<usize>,
    SegmentedSlice<'a, T>,
    range_to
);
impl_range_index!(
    SegmentedSlice,
    RangeFrom<usize>,
    SegmentedSlice<'a, T>,
    range_from
);
impl_range_index!(SegmentedSlice, RangeFull, SegmentedSlice<'a, T>, range_full);
impl_range_index!(
    SegmentedSlice,
    RangeInclusive<usize>,
    SegmentedSlice<'a, T>,
    range_inclusive
);
impl_range_index!(
    SegmentedSlice,
    RangeToInclusive<usize>,
    SegmentedSlice<'a, T>,
    range_to_inclusive
);

// Implementations for SegmentedSliceMut (mutable)
// MutRet is SegmentedSliceMut
impl_range_index!(
    SegmentedSliceMut,
    Range<usize>,
    SegmentedSliceMut<'a, T>,
    range_bounds
);
impl_range_index!(
    SegmentedSliceMut,
    RangeTo<usize>,
    SegmentedSliceMut<'a, T>,
    range_to
);
impl_range_index!(
    SegmentedSliceMut,
    RangeFrom<usize>,
    SegmentedSliceMut<'a, T>,
    range_from
);
impl_range_index!(
    SegmentedSliceMut,
    RangeFull,
    SegmentedSliceMut<'a, T>,
    range_full
);
impl_range_index!(
    SegmentedSliceMut,
    RangeInclusive<usize>,
    SegmentedSliceMut<'a, T>,
    range_inclusive
);
impl_range_index!(
    SegmentedSliceMut,
    RangeToInclusive<usize>,
    SegmentedSliceMut<'a, T>,
    range_to_inclusive
);

// --- SegmentedVec ---

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for usize {
    type Output<'a>
        = &'a T
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = &'a mut T
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        vec.as_slice().get(self)
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        vec.as_mut_slice().get_mut(self)
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        vec.as_slice().get_unchecked(self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        vec.as_mut_slice().get_unchecked_mut(self)
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("index out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("index out of bounds")
    }
}

macro_rules! impl_vec_range_index {
    ($RangeType:ty, $normalize:expr) => {
        unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for $RangeType {
            type Output<'a>
                = SegmentedSlice<'a, T>
            where
                T: 'a,
                A: 'a;
            type OutputMut<'a>
                = SegmentedSliceMut<'a, T>
            where
                T: 'a,
                A: 'a;

            #[inline]
            fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
                vec.as_slice().get(self)
            }

            #[inline]
            fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
                vec.as_mut_slice().get_mut(self)
            }

            #[inline]
            unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
                vec.as_slice().get_unchecked(self)
            }

            #[inline]
            unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
                vec.as_mut_slice().get_unchecked_mut(self)
            }

            #[inline]
            fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
                self.index(&vec.as_slice())
            }

            #[inline]
            fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
                self.index_mut(&mut vec.as_mut_slice())
            }
        }
    };
}

impl_vec_range_index!(Range<usize>, range_bounds);
impl_vec_range_index!(RangeTo<usize>, range_to);
impl_vec_range_index!(RangeFrom<usize>, range_from);
impl_vec_range_index!(RangeFull, range_full);
impl_vec_range_index!(RangeInclusive<usize>, range_inclusive);
impl_vec_range_index!(RangeToInclusive<usize>, range_to_inclusive);
