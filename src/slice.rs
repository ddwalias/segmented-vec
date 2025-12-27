//! Slice types for SegmentedVec.
//!
//! This module provides `SegmentedSlice` and `SegmentedSliceMut` types that
//! behave like `&[T]` and `&mut [T]` but work with non-contiguous memory.

use std::cmp::Ordering;
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::SegmentedVec;

/// An immutable slice view into a `SegmentedVec`.
///
/// This type behaves like `&[T]` but works with non-contiguous memory.
///
/// # Example
///
/// ```
/// use segmented_vec::SegmentedVec;
///
/// let mut vec: SegmentedVec<i32, 4> = SegmentedVec::new();
/// vec.extend(0..10);
///
/// let slice = vec.as_slice();
/// assert_eq!(slice.len(), 10);
/// assert_eq!(slice[0], 0);
/// assert_eq!(slice.first(), Some(&0));
/// ```
#[derive(Clone, Copy)]
pub struct SegmentedSlice<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
}

impl<'a, T, const PREALLOC: usize> SegmentedSlice<'a, T, PREALLOC> {
    /// Creates a new slice covering the entire vector.
    #[inline]
    pub(crate) fn new(vec: &'a SegmentedVec<T, PREALLOC>) -> Self {
        Self {
            vec,
            start: 0,
            end: vec.len(),
        }
    }

    /// Creates a new slice covering a range of the vector.
    #[inline]
    pub(crate) fn from_range(vec: &'a SegmentedVec<T, PREALLOC>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end && end <= vec.len());
        Self { vec, start, end }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if the slice is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns a reference to the element at the given index, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.vec.unchecked_at(self.start + index) })
        } else {
            None
        }
    }

    /// Returns a reference to the first element, or `None` if empty.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a reference to the last element, or `None` if empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Returns references to the first and rest of the elements, or `None` if empty.
    #[inline]
    pub fn split_first(&self) -> Option<(&T, SegmentedSlice<'a, T, PREALLOC>)> {
        if self.is_empty() {
            None
        } else {
            Some((
                unsafe { self.vec.unchecked_at(self.start) },
                SegmentedSlice::from_range(self.vec, self.start + 1, self.end),
            ))
        }
    }

    /// Returns references to the last and rest of the elements, or `None` if empty.
    #[inline]
    pub fn split_last(&self) -> Option<(&T, SegmentedSlice<'a, T, PREALLOC>)> {
        if self.is_empty() {
            None
        } else {
            Some((
                unsafe { self.vec.unchecked_at(self.end - 1) },
                SegmentedSlice::from_range(self.vec, self.start, self.end - 1),
            ))
        }
    }

    /// Divides the slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` and the second will contain
    /// all indices from `[mid, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    #[inline]
    pub fn split_at(&self, mid: usize) -> (SegmentedSlice<'a, T, PREALLOC>, SegmentedSlice<'a, T, PREALLOC>) {
        assert!(mid <= self.len());
        (
            SegmentedSlice::from_range(self.vec, self.start, self.start + mid),
            SegmentedSlice::from_range(self.vec, self.start + mid, self.end),
        )
    }

    /// Returns an iterator over the slice.
    #[inline]
    pub fn iter(&self) -> SliceIter<'a, T, PREALLOC> {
        SliceIter {
            vec: self.vec,
            start: self.start,
            end: self.end,
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
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        if needle.len() > self.len() {
            return false;
        }
        for (i, item) in needle.iter().enumerate() {
            if self.get(i) != Some(item) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        if needle.len() > self.len() {
            return false;
        }
        let start = self.len() - needle.len();
        for (i, item) in needle.iter().enumerate() {
            if self.get(start + i) != Some(item) {
                return false;
            }
        }
        true
    }

    /// Binary searches this slice for a given element.
    ///
    /// If the value is found, returns `Ok(index)`. If not found, returns
    /// `Err(index)` where `index` is the position where the element could be inserted.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|elem| elem.cmp(x))
    }

    /// Binary searches this slice with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let mut left = 0;
        let mut right = self.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let elem = unsafe { self.vec.unchecked_at(self.start + mid) };
            match f(elem) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    /// Binary searches this slice with a key extraction function.
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|elem| f(elem).cmp(b))
    }

    /// Returns a subslice with elements in the given range.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[inline]
    pub fn slice<R>(self, range: R) -> SegmentedSlice<'a, T, PREALLOC>
    where
        R: SliceIndex<'a, T, PREALLOC, Output = SegmentedSlice<'a, T, PREALLOC>>,
    {
        range.index(self)
    }

    /// Copies the elements into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }
}

impl<'a, T: std::fmt::Debug, const PREALLOC: usize> std::fmt::Debug for SegmentedSlice<'a, T, PREALLOC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: PartialEq, const PREALLOC: usize> PartialEq for SegmentedSlice<'a, T, PREALLOC> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<'a, T: PartialEq, const PREALLOC: usize> PartialEq<[T]> for SegmentedSlice<'a, T, PREALLOC> {
    fn eq(&self, other: &[T]) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<'a, T: PartialEq, const PREALLOC: usize> PartialEq<Vec<T>> for SegmentedSlice<'a, T, PREALLOC> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self == other.as_slice()
    }
}

impl<'a, T: Eq, const PREALLOC: usize> Eq for SegmentedSlice<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Index<usize> for SegmentedSlice<'a, T, PREALLOC> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, T, const PREALLOC: usize> IntoIterator for SegmentedSlice<'a, T, PREALLOC> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T, PREALLOC>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const PREALLOC: usize> IntoIterator for &SegmentedSlice<'a, T, PREALLOC> {
    type Item = &'a T;
    type IntoIter = SliceIter<'a, T, PREALLOC>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// --- Mutable Slice ---

/// A mutable slice view into a `SegmentedVec`.
///
/// This type behaves like `&mut [T]` but works with non-contiguous memory.
///
/// # Example
///
/// ```
/// use segmented_vec::SegmentedVec;
///
/// let mut vec: SegmentedVec<i32, 4> = SegmentedVec::new();
/// vec.extend(0..10);
///
/// let mut slice = vec.as_mut_slice();
/// slice[0] = 100;
/// assert_eq!(slice[0], 100);
/// ```
pub struct SegmentedSliceMut<'a, T, const PREALLOC: usize> {
    vec: &'a mut SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
}

impl<'a, T, const PREALLOC: usize> SegmentedSliceMut<'a, T, PREALLOC> {
    /// Creates a new mutable slice covering the entire vector.
    #[inline]
    pub(crate) fn new(vec: &'a mut SegmentedVec<T, PREALLOC>) -> Self {
        let end = vec.len();
        Self { vec, start: 0, end }
    }

    /// Creates a new mutable slice covering a range of the vector.
    #[inline]
    pub(crate) fn from_range(vec: &'a mut SegmentedVec<T, PREALLOC>, start: usize, end: usize) -> Self {
        debug_assert!(start <= end && end <= vec.len());
        Self { vec, start, end }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if the slice is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns a reference to the element at the given index, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.vec.unchecked_at(self.start + index) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index, or `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            Some(unsafe { self.vec.unchecked_at_mut(self.start + index) })
        } else {
            None
        }
    }

    /// Returns a reference to the first element, or `None` if empty.
    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns a mutable reference to the first element, or `None` if empty.
    #[inline]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    /// Returns a reference to the last element, or `None` if empty.
    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Returns a mutable reference to the last element, or `None` if empty.
    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            None
        } else {
            let idx = self.len() - 1;
            self.get_mut(idx)
        }
    }

    /// Swaps two elements in the slice.
    ///
    /// # Panics
    ///
    /// Panics if `a` or `b` are out of bounds.
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        assert!(a < self.len() && b < self.len());
        if a != b {
            unsafe {
                let ptr_a = self.vec.unchecked_at_mut(self.start + a) as *mut T;
                let ptr_b = self.vec.unchecked_at_mut(self.start + b) as *mut T;
                std::ptr::swap(ptr_a, ptr_b);
            }
        }
    }

    /// Reverses the order of elements in the slice.
    pub fn reverse(&mut self) {
        let len = self.len();
        for i in 0..len / 2 {
            self.swap(i, len - 1 - i);
        }
    }

    /// Returns an iterator over the slice.
    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T, PREALLOC> {
        SliceIter {
            vec: self.vec,
            start: self.start,
            end: self.end,
        }
    }

    /// Returns a mutable iterator over the slice.
    #[inline]
    pub fn iter_mut(&mut self) -> SliceIterMut<'_, T, PREALLOC> {
        SliceIterMut {
            vec: self.vec,
            end: self.end,
            index: self.start,
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

    /// Binary searches this slice for a given element.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search_by(|elem| elem.cmp(x))
    }

    /// Binary searches this slice with a comparator function.
    pub fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> Ordering,
    {
        let mut left = 0;
        let mut right = self.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let elem = unsafe { self.vec.unchecked_at(self.start + mid) };
            match f(elem) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    /// Binary searches this slice with a key extraction function.
    pub fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&T) -> B,
        B: Ord,
    {
        self.binary_search_by(|elem| f(elem).cmp(b))
    }

    /// Fills the slice with the given value.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for i in 0..self.len() {
            *unsafe { self.vec.unchecked_at_mut(self.start + i) } = value.clone();
        }
    }

    /// Fills the slice with values produced by a function.
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T,
    {
        for i in 0..self.len() {
            *unsafe { self.vec.unchecked_at_mut(self.start + i) } = f();
        }
    }

    /// Copies elements from `src` into the slice.
    ///
    /// The length of `src` must be the same as the slice.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    pub fn copy_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        assert_eq!(self.len(), src.len());
        for (i, val) in src.iter().enumerate() {
            *unsafe { self.vec.unchecked_at_mut(self.start + i) } = val.clone();
        }
    }

    /// Sorts the slice.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort_by(|a, b| a.cmp(b));
    }

    /// Sorts the slice with a comparator function.
    pub fn sort_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if self.len() <= 1 {
            return;
        }
        let mut is_less = |a: &T, b: &T| compare(a, b) == Ordering::Less;
        crate::sort::merge_sort(self.vec, self.start, self.end, &mut is_less);
    }

    /// Sorts the slice with a key extraction function.
    pub fn sort_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_by(|a, b| f(a).cmp(&f(b)));
    }

    /// Sorts the slice using an unstable algorithm.
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable_by(|a, b| a.cmp(b));
    }

    /// Sorts the slice with a comparator function using an unstable algorithm.
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        if self.len() <= 1 {
            return;
        }
        let mut is_less = |a: &T, b: &T| compare(a, b) == Ordering::Less;
        crate::sort::quicksort(self.vec, self.start, self.end, &mut is_less);
    }

    /// Sorts the slice with a key extraction function using an unstable algorithm.
    pub fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.sort_unstable_by(|a, b| f(a).cmp(&f(b)));
    }

    /// Copies the elements into a new `Vec`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Returns an immutable view of this slice.
    #[inline]
    pub fn as_slice(&self) -> SegmentedSlice<'_, T, PREALLOC> {
        SegmentedSlice {
            vec: self.vec,
            start: self.start,
            end: self.end,
        }
    }
}

impl<'a, T: std::fmt::Debug, const PREALLOC: usize> std::fmt::Debug for SegmentedSliceMut<'a, T, PREALLOC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T, const PREALLOC: usize> Index<usize> for SegmentedSliceMut<'a, T, PREALLOC> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<'a, T, const PREALLOC: usize> IndexMut<usize> for SegmentedSliceMut<'a, T, PREALLOC> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

// --- Iterators ---

/// An iterator over references to elements of a `SegmentedSlice`.
pub struct SliceIter<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
}

impl<'a, T, const PREALLOC: usize> Iterator for SliceIter<'a, T, PREALLOC> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let item = unsafe { self.vec.unchecked_at(self.start) };
            self.start += 1;
            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.end - self.start
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.end - self.start {
            self.start = self.end;
            None
        } else {
            self.start += n;
            self.next()
        }
    }
}

impl<'a, T, const PREALLOC: usize> DoubleEndedIterator for SliceIter<'a, T, PREALLOC> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            self.end -= 1;
            Some(unsafe { self.vec.unchecked_at(self.end) })
        }
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for SliceIter<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Clone for SliceIter<'a, T, PREALLOC> {
    fn clone(&self) -> Self {
        SliceIter {
            vec: self.vec,
            start: self.start,
            end: self.end,
        }
    }
}

/// A mutable iterator over elements of a `SegmentedSliceMut`.
pub struct SliceIterMut<'a, T, const PREALLOC: usize> {
    vec: &'a mut SegmentedVec<T, PREALLOC>,
    end: usize,
    index: usize,
}

impl<'a, T, const PREALLOC: usize> Iterator for SliceIterMut<'a, T, PREALLOC> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            None
        } else {
            let ptr = unsafe { self.vec.unchecked_at_mut(self.index) as *mut T };
            self.index += 1;
            // Safety: Each index is yielded only once, and we have exclusive access.
            Some(unsafe { &mut *ptr })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.index;
        (len, Some(len))
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for SliceIterMut<'a, T, PREALLOC> {}

// --- SliceIndex trait for range indexing ---

/// Helper trait for indexing operations on `SegmentedSlice`.
pub trait SliceIndex<'a, T, const PREALLOC: usize> {
    /// The output type returned by indexing.
    type Output;

    /// Returns the indexed slice.
    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> Self::Output;
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for Range<usize> {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        assert!(self.start <= self.end && self.end <= slice.len());
        SegmentedSlice::from_range(slice.vec, slice.start + self.start, slice.start + self.end)
    }
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for RangeFrom<usize> {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        assert!(self.start <= slice.len());
        SegmentedSlice::from_range(slice.vec, slice.start + self.start, slice.end)
    }
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for RangeTo<usize> {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        assert!(self.end <= slice.len());
        SegmentedSlice::from_range(slice.vec, slice.start, slice.start + self.end)
    }
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for RangeFull {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        slice
    }
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for RangeInclusive<usize> {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        let start = *self.start();
        let end = *self.end();
        assert!(start <= end && end < slice.len());
        SegmentedSlice::from_range(slice.vec, slice.start + start, slice.start + end + 1)
    }
}

impl<'a, T: 'a, const PREALLOC: usize> SliceIndex<'a, T, PREALLOC> for RangeToInclusive<usize> {
    type Output = SegmentedSlice<'a, T, PREALLOC>;

    fn index(self, slice: SegmentedSlice<'a, T, PREALLOC>) -> SegmentedSlice<'a, T, PREALLOC> {
        assert!(self.end < slice.len());
        SegmentedSlice::from_range(slice.vec, slice.start, slice.start + self.end + 1)
    }
}
