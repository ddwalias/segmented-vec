use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::slice::SegmentedSlice;
use crate::SegmentedVec;
use allocator_api2::alloc::Allocator;
use std::marker::PhantomData;

/// A helper trait for generic indexing into [`SegmentedVec`] or [`SegmentedSlice`].
/// # Safety
/// This trait is unsafe because it allows implementation of unchecked indexing.
pub unsafe trait SliceIndex<T: ?Sized> {
    /// The output of the indexing operation.
    type Output<'a>
    where
        Self: 'a,
        T: 'a;

    /// The mutable output of the indexing operation.
    type OutputMut<'a>
    where
        Self: 'a,
        T: 'a;

    /// Returns a reference to the element at the given index.
    fn get(self, container: &T) -> Option<Self::Output<'_>>;

    /// Returns a mutable reference to the element at the given index.
    fn get_mut(self, container: &mut T) -> Option<Self::OutputMut<'_>>;

    /// Returns a reference to the element at the given index, without bounds checking.
    /// # Safety
    /// The caller must ensure that the index is within bounds.
    unsafe fn get_unchecked(self, container: &T) -> Self::Output<'_>;

    /// Returns a mutable reference to the element at the given index, without bounds checking.
    /// # Safety
    /// The caller must ensure that the index is within bounds.
    unsafe fn get_unchecked_mut(self, container: &mut T) -> Self::OutputMut<'_>;

    /// Returns a reference to the element at the given index, or panics if the index is out of bounds.
    fn index(self, container: &T) -> Self::Output<'_>;

    /// Returns a mutable reference to the element at the given index, or panics if the index is out of bounds.
    fn index_mut(self, container: &mut T) -> Self::OutputMut<'_>;
}

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
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self < slice.len() {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        &*slice.buf().ptr_at(slice.start + self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        &mut *slice.buf_mut().ptr_at(slice.start + self)
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

// --- Range (start..end) ---

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>> for Range<usize> {
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if self.start <= self.end && self.end <= slice.len {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self.start <= self.end && self.end <= slice.len {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        SegmentedSlice::new(slice.buf, slice.start + self.start, slice.start + self.end)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        SegmentedSlice::new(slice.buf, slice.start + self.start, slice.start + self.end)
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("range out of bounds")
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>> for RangeTo<usize> {
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if self.end <= slice.len {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self.end <= slice.len {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        SegmentedSlice::new(slice.buf, slice.start, slice.start + self.end)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        SegmentedSlice::new(slice.buf, slice.start, slice.start + self.end)
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("range end out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("range end out of bounds")
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>> for RangeFrom<usize> {
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if self.start <= slice.len {
            unsafe { Some(self.get_unchecked(slice)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self.start <= slice.len {
            unsafe { Some(self.get_unchecked_mut(slice)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        let new_len = slice.len - self.start;
        SegmentedSlice {
            buf: slice.buf,
            start: slice.start + self.start,
            len: new_len,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
            _marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        let new_len = slice.len - self.start;
        SegmentedSlice {
            buf: slice.buf,
            start: slice.start + self.start,
            len: new_len,
            end_ptr: slice.end_ptr,
            end_seg: slice.end_seg,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("range start out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("range start out of bounds")
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>> for RangeFull {
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        // RangeFull is always in bounds
        unsafe { Some(self.get_unchecked(slice)) }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        // RangeFull is always in bounds
        unsafe { Some(self.get_unchecked_mut(slice)) }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        SegmentedSlice::new(slice.buf, slice.start, slice.start + slice.len)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        SegmentedSlice::new(slice.buf, slice.start, slice.start + slice.len)
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).unwrap()
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).unwrap()
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>>
    for RangeInclusive<usize>
{
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if *self.end() == usize::MAX {
            None
        } else {
            (*self.start()..*self.end() + 1).get(slice)
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if *self.end() == usize::MAX {
            None
        } else {
            (*self.start()..*self.end() + 1).get_mut(slice)
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        (*self.start()..*self.end() + 1).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        (*self.start()..*self.end() + 1).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("range out of bounds")
    }
}

unsafe impl<'a, T, A: Allocator + 'a> SliceIndex<SegmentedSlice<'a, T, A>>
    for RangeToInclusive<usize>
{
    type Output<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;
    type OutputMut<'b>
        = SegmentedSlice<'a, T, A>
    where
        'a: 'b;

    #[inline]
    fn get(self, slice: &SegmentedSlice<'a, T, A>) -> Option<Self::Output<'a>> {
        if self.end == usize::MAX {
            None
        } else {
            (..self.end + 1).get(slice)
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Option<Self::OutputMut<'a>> {
        if self.end == usize::MAX {
            None
        } else {
            (..self.end + 1).get_mut(slice)
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        (..self.end + 1).get_unchecked(slice)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        (..self.end + 1).get_unchecked_mut(slice)
    }

    #[inline]
    fn index(self, slice: &SegmentedSlice<'a, T, A>) -> Self::Output<'a> {
        self.get(slice).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, slice: &mut SegmentedSlice<'a, T, A>) -> Self::OutputMut<'a> {
        self.get_mut(slice).expect("range out of bounds")
    }
}

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
        if self < vec.len {
            unsafe { Some(self.get_unchecked(vec)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if self < vec.len {
            unsafe { Some(self.get_unchecked_mut(vec)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        &*vec.buf.ptr_at(self)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        &mut *vec.buf.ptr_at(self)
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

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for Range<usize> {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        if self.start <= self.end && self.end <= vec.len {
            unsafe { Some(self.get_unchecked(vec)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if self.start <= self.end && self.end <= vec.len {
            unsafe { Some(self.get_unchecked_mut(vec)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        SegmentedSlice::new(&vec.buf, self.start, self.end)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        SegmentedSlice::new(&vec.buf, self.start, self.end)
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("range out of bounds")
    }
}

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for RangeTo<usize> {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        if self.end <= vec.len {
            unsafe { Some(self.get_unchecked(vec)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if self.end <= vec.len {
            unsafe { Some(self.get_unchecked_mut(vec)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        SegmentedSlice::new(&vec.buf, 0, self.end)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        SegmentedSlice::new(&vec.buf, 0, self.end)
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("range end out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("range end out of bounds")
    }
}

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for RangeFrom<usize> {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        if self.start <= vec.len {
            unsafe { Some(self.get_unchecked(vec)) }
        } else {
            None
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if self.start <= vec.len {
            unsafe { Some(self.get_unchecked_mut(vec)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        SegmentedSlice {
            buf: &vec.buf,
            start: self.start,
            len: vec.len - self.start,
            end_ptr: vec.write_ptr,
            end_seg: vec.active_segment_index,
            _marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        SegmentedSlice {
            buf: &vec.buf,
            start: self.start,
            len: vec.len - self.start,
            end_ptr: vec.write_ptr,
            end_seg: vec.active_segment_index,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("range start out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("range start out of bounds")
    }
}

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for RangeFull {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        unsafe { Some(self.get_unchecked(vec)) }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        unsafe { Some(self.get_unchecked_mut(vec)) }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        vec.as_slice()
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        vec.as_mut_slice()
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        vec.as_slice()
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        vec.as_mut_slice()
    }
}

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for RangeInclusive<usize> {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        if *self.end() == usize::MAX {
            None
        } else {
            (*self.start()..*self.end() + 1).get(vec)
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if *self.end() == usize::MAX {
            None
        } else {
            (*self.start()..*self.end() + 1).get_mut(vec)
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        (*self.start()..*self.end() + 1).get_unchecked(vec)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        (*self.start()..*self.end() + 1).get_unchecked_mut(vec)
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("range out of bounds")
    }
}

unsafe impl<T, A: Allocator> SliceIndex<SegmentedVec<T, A>> for RangeToInclusive<usize> {
    type Output<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;
    type OutputMut<'a>
        = SegmentedSlice<'a, T, A>
    where
        T: 'a,
        A: 'a;

    #[inline]
    fn get(self, vec: &SegmentedVec<T, A>) -> Option<Self::Output<'_>> {
        if self.end == usize::MAX {
            None
        } else {
            (..self.end + 1).get(vec)
        }
    }

    #[inline]
    fn get_mut(self, vec: &mut SegmentedVec<T, A>) -> Option<Self::OutputMut<'_>> {
        if self.end == usize::MAX {
            None
        } else {
            (..self.end + 1).get_mut(vec)
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        (..self.end + 1).get_unchecked(vec)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        (..self.end + 1).get_unchecked_mut(vec)
    }

    #[inline]
    fn index(self, vec: &SegmentedVec<T, A>) -> Self::Output<'_> {
        self.get(vec).expect("range out of bounds")
    }

    #[inline]
    fn index_mut(self, vec: &mut SegmentedVec<T, A>) -> Self::OutputMut<'_> {
        self.get_mut(vec).expect("range out of bounds")
    }
}
