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

    /// Returns a reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len());
        unsafe { self.vec.unchecked_at(self.start + index) }
    }

    /// Checks if the elements of this slice are sorted.
    ///
    /// That is, for each element `a` and its following element `b`, `a <= b` must hold.
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.is_sorted_by(|a, b| a <= b)
    }

    /// Checks if the elements of this slice are sorted using the given comparator function.
    pub fn is_sorted_by<F>(&self, mut compare: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = self.len();
        if len <= 1 {
            return true;
        }
        for i in 0..len - 1 {
            let a = unsafe { self.vec.unchecked_at(self.start + i) };
            let b = unsafe { self.vec.unchecked_at(self.start + i + 1) };
            if !compare(a, b) {
                return false;
            }
        }
        true
    }

    /// Checks if the elements of this slice are sorted using the given key extraction function.
    pub fn is_sorted_by_key<K, F>(&self, mut f: F) -> bool
    where
        F: FnMut(&T) -> K,
        K: PartialOrd,
    {
        self.is_sorted_by(|a, b| f(a) <= f(b))
    }

    /// Returns the index of the partition point according to the given predicate.
    ///
    /// The slice is assumed to be partitioned according to the given predicate.
    /// This returns the first index where the predicate returns `false`.
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        let mut left = 0;
        let mut right = self.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let elem = unsafe { self.vec.unchecked_at(self.start + mid) };
            if pred(elem) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    /// Returns an iterator over all contiguous windows of length `size`.
    ///
    /// The windows overlap. If the slice is shorter than `size`, the iterator returns no values.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn windows(&self, size: usize) -> Windows<'a, T, PREALLOC> {
        assert!(size != 0);
        Windows {
            vec: self.vec,
            start: self.start,
            end: self.end,
            size,
        }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length
    /// of the slice, then the last chunk will not have length `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'a, T, PREALLOC> {
        assert!(chunk_size != 0);
        Chunks {
            vec: self.vec,
            start: self.start,
            end: self.end,
            chunk_size,
        }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time, starting at the end.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not divide the length
    /// of the slice, then the last chunk will not have length `chunk_size`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'a, T, PREALLOC> {
        assert!(chunk_size != 0);
        RChunks {
            vec: self.vec,
            start: self.start,
            end: self.end,
            chunk_size,
        }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    ///
    /// If `chunk_size` does not divide the length, the last up to `chunk_size-1`
    /// elements will be omitted and can be retrieved from the `remainder` function
    /// of the iterator.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'a, T, PREALLOC> {
        assert!(chunk_size != 0);
        let remainder_start = self.start + (self.len() / chunk_size) * chunk_size;
        ChunksExact {
            vec: self.vec,
            start: self.start,
            end: remainder_start,
            remainder_end: self.end,
            chunk_size,
        }
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

    /// Returns a reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len());
        unsafe { self.vec.unchecked_at(self.start + index) }
    }

    /// Returns a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len());
        unsafe { self.vec.unchecked_at_mut(self.start + index) }
    }

    /// Returns `true` if `needle` is a prefix of the slice.
    pub fn starts_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().starts_with(needle)
    }

    /// Returns `true` if `needle` is a suffix of the slice.
    pub fn ends_with(&self, needle: &[T]) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().ends_with(needle)
    }

    /// Checks if the elements of this slice are sorted.
    pub fn is_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.as_slice().is_sorted()
    }

    /// Checks if the elements of this slice are sorted using the given comparator function.
    pub fn is_sorted_by<F>(&self, compare: F) -> bool
    where
        F: FnMut(&T, &T) -> bool,
    {
        self.as_slice().is_sorted_by(compare)
    }

    /// Checks if the elements of this slice are sorted using the given key extraction function.
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

    /// Rotates the slice in-place such that the first `mid` elements move to the end.
    ///
    /// After calling `rotate_left`, the element previously at index `mid` is now at index `0`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    pub fn rotate_left(&mut self, mid: usize) {
        assert!(mid <= self.len());
        if mid == 0 || mid == self.len() {
            return;
        }
        // Use the reversal algorithm: reverse first part, reverse second part, reverse all
        self.reverse_range(0, mid);
        self.reverse_range(mid, self.len());
        self.reverse();
    }

    /// Rotates the slice in-place such that the last `k` elements move to the front.
    ///
    /// After calling `rotate_right`, the element previously at index `len - k` is now at index `0`.
    ///
    /// # Panics
    ///
    /// Panics if `k > len`.
    pub fn rotate_right(&mut self, k: usize) {
        assert!(k <= self.len());
        if k == 0 || k == self.len() {
            return;
        }
        self.rotate_left(self.len() - k);
    }

    /// Helper to reverse a range within the slice.
    fn reverse_range(&mut self, start: usize, end: usize) {
        let mut left = start;
        let mut right = end;
        while left < right {
            right -= 1;
            self.swap(left, right);
            left += 1;
        }
    }

    /// Divides one mutable slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` and the second will contain
    /// all indices from `[mid, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > len`.
    ///
    /// # Note
    ///
    /// Due to borrowing rules, this method consumes self and returns two new slices.
    pub fn split_at_mut(self, mid: usize) -> (SegmentedSliceMut<'a, T, PREALLOC>, SegmentedSliceMut<'a, T, PREALLOC>) {
        assert!(mid <= self.len());
        let start = self.start;
        let end = self.end;
        // Safety: We consume self and split the range, so no aliasing occurs.
        // We need to use raw pointers to create two mutable references.
        let vec_ptr = self.vec as *mut SegmentedVec<T, PREALLOC>;
        // Note: SegmentedSliceMut doesn't implement Drop, so we don't need to call forget
        unsafe {
            (
                SegmentedSliceMut {
                    vec: &mut *vec_ptr,
                    start,
                    end: start + mid,
                },
                SegmentedSliceMut {
                    vec: &mut *vec_ptr,
                    start: start + mid,
                    end,
                },
            )
        }
    }

    /// Returns an iterator over `chunk_size` elements of the slice at a time.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T, PREALLOC> {
        self.as_slice().chunks(chunk_size)
    }

    /// Returns an iterator over all contiguous windows of length `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn windows(&self, size: usize) -> Windows<'_, T, PREALLOC> {
        self.as_slice().windows(size)
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

// --- Additional Iterators ---

/// An iterator over overlapping windows of elements.
pub struct Windows<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
    size: usize,
}

impl<'a, T, const PREALLOC: usize> Iterator for Windows<'a, T, PREALLOC> {
    type Item = SegmentedSlice<'a, T, PREALLOC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start + self.size > self.end {
            None
        } else {
            let slice = SegmentedSlice::from_range(self.vec, self.start, self.start + self.size);
            self.start += 1;
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end.saturating_sub(self.start).saturating_sub(self.size - 1);
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.start = self.start.saturating_add(n);
        self.next()
    }
}

impl<'a, T, const PREALLOC: usize> DoubleEndedIterator for Windows<'a, T, PREALLOC> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start + self.size > self.end {
            None
        } else {
            self.end -= 1;
            Some(SegmentedSlice::from_range(self.vec, self.end - self.size + 1, self.end + 1))
        }
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for Windows<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Clone for Windows<'a, T, PREALLOC> {
    fn clone(&self) -> Self {
        Windows {
            vec: self.vec,
            start: self.start,
            end: self.end,
            size: self.size,
        }
    }
}

/// An iterator over non-overlapping chunks of elements.
pub struct Chunks<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
    chunk_size: usize,
}

impl<'a, T, const PREALLOC: usize> Iterator for Chunks<'a, T, PREALLOC> {
    type Item = SegmentedSlice<'a, T, PREALLOC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_end = std::cmp::min(self.start + self.chunk_size, self.end);
            let slice = SegmentedSlice::from_range(self.vec, self.start, chunk_end);
            self.start = chunk_end;
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start >= self.end {
            (0, Some(0))
        } else {
            let remaining = self.end - self.start;
            let len = remaining.div_ceil(self.chunk_size);
            (len, Some(len))
        }
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size);
        self.start = self.start.saturating_add(skip);
        self.next()
    }
}

impl<'a, T, const PREALLOC: usize> DoubleEndedIterator for Chunks<'a, T, PREALLOC> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let remaining = self.end - self.start;
            let last_chunk_size = if remaining.is_multiple_of(self.chunk_size) {
                self.chunk_size
            } else {
                remaining % self.chunk_size
            };
            let chunk_start = self.end - last_chunk_size;
            let slice = SegmentedSlice::from_range(self.vec, chunk_start, self.end);
            self.end = chunk_start;
            Some(slice)
        }
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for Chunks<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Clone for Chunks<'a, T, PREALLOC> {
    fn clone(&self) -> Self {
        Chunks {
            vec: self.vec,
            start: self.start,
            end: self.end,
            chunk_size: self.chunk_size,
        }
    }
}

/// An iterator over non-overlapping chunks of elements, starting from the end.
pub struct RChunks<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
    chunk_size: usize,
}

impl<'a, T, const PREALLOC: usize> Iterator for RChunks<'a, T, PREALLOC> {
    type Item = SegmentedSlice<'a, T, PREALLOC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let remaining = self.end - self.start;
            let chunk_size = std::cmp::min(self.chunk_size, remaining);
            let chunk_start = self.end - chunk_size;
            let slice = SegmentedSlice::from_range(self.vec, chunk_start, self.end);
            self.end = chunk_start;
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start >= self.end {
            (0, Some(0))
        } else {
            let remaining = self.end - self.start;
            let len = remaining.div_ceil(self.chunk_size);
            (len, Some(len))
        }
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'a, T, const PREALLOC: usize> DoubleEndedIterator for RChunks<'a, T, PREALLOC> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let chunk_end = std::cmp::min(self.start + self.chunk_size, self.end);
            let slice = SegmentedSlice::from_range(self.vec, self.start, chunk_end);
            self.start = chunk_end;
            Some(slice)
        }
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for RChunks<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Clone for RChunks<'a, T, PREALLOC> {
    fn clone(&self) -> Self {
        RChunks {
            vec: self.vec,
            start: self.start,
            end: self.end,
            chunk_size: self.chunk_size,
        }
    }
}

/// An iterator over exact-size chunks of elements.
pub struct ChunksExact<'a, T, const PREALLOC: usize> {
    vec: &'a SegmentedVec<T, PREALLOC>,
    start: usize,
    end: usize,
    remainder_end: usize,
    chunk_size: usize,
}

impl<'a, T, const PREALLOC: usize> ChunksExact<'a, T, PREALLOC> {
    /// Returns the remainder of the original slice that was not consumed.
    pub fn remainder(&self) -> SegmentedSlice<'a, T, PREALLOC> {
        SegmentedSlice::from_range(self.vec, self.end, self.remainder_end)
    }
}

impl<'a, T, const PREALLOC: usize> Iterator for ChunksExact<'a, T, PREALLOC> {
    type Item = SegmentedSlice<'a, T, PREALLOC>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start + self.chunk_size > self.end {
            None
        } else {
            let slice = SegmentedSlice::from_range(self.vec, self.start, self.start + self.chunk_size);
            self.start += self.chunk_size;
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.start);
        let len = remaining / self.chunk_size;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size);
        self.start = self.start.saturating_add(skip);
        self.next()
    }
}

impl<'a, T, const PREALLOC: usize> DoubleEndedIterator for ChunksExact<'a, T, PREALLOC> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start + self.chunk_size > self.end {
            None
        } else {
            self.end -= self.chunk_size;
            Some(SegmentedSlice::from_range(self.vec, self.end, self.end + self.chunk_size))
        }
    }
}

impl<'a, T, const PREALLOC: usize> ExactSizeIterator for ChunksExact<'a, T, PREALLOC> {}

impl<'a, T, const PREALLOC: usize> Clone for ChunksExact<'a, T, PREALLOC> {
    fn clone(&self) -> Self {
        ChunksExact {
            vec: self.vec,
            start: self.start,
            end: self.end,
            remainder_end: self.remainder_end,
            chunk_size: self.chunk_size,
        }
    }
}
