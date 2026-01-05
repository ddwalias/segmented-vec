//! Iterator implementations for `SegmentedVec`.

use crate::raw_vec::RawSegmentedVec;
use crate::SegmentedVec;

/// An iterator over references to elements of a `SegmentedVec`.
pub struct Iter<'a, T> {
    pub(crate) vec: &'a SegmentedVec<T>,
    /// Current pointer within segment
    pub(crate) ptr: *const T,
    /// End of current segment (min of segment capacity and vec.len)
    pub(crate) segment_end: *const T,
    /// Current logical index
    pub(crate) index: usize,
    /// Current segment index
    pub(crate) segment_index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // For ZSTs, just check index vs len
        if std::mem::size_of::<T>() == 0 {
            if self.index < self.vec.len() {
                self.index += 1;
                return Some(unsafe { &*std::ptr::NonNull::dangling().as_ptr() });
            }
            return None;
        }

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
        let remaining = self.vec.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T> Iter<'a, T> {
    #[cold]
    fn next_segment(&mut self) -> Option<&'a T> {
        if self.index >= self.vec.len() {
            return None;
        }

        let segment_size = RawSegmentedVec::<T>::segment_capacity(self.segment_index);
        let ptr = unsafe { self.vec.buf.segment_ptr(self.segment_index) };
        let segment_len = segment_size.min(self.vec.len() - self.index);
        self.ptr = ptr;
        self.segment_end = unsafe { ptr.add(segment_len) };
        self.segment_index += 1;

        let result = unsafe { &*self.ptr };
        self.ptr = unsafe { self.ptr.add(1) };
        self.index += 1;
        Some(result)
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> std::iter::FusedIterator for Iter<'_, T> {}

// Safety: Iter only yields shared references
unsafe impl<T: Sync> Sync for Iter<'_, T> {}
unsafe impl<T: Sync> Send for Iter<'_, T> {}

/// An iterator over mutable references to elements of a `SegmentedVec`.
pub struct IterMut<'a, T> {
    pub(crate) vec: &'a mut SegmentedVec<T>,
    /// Current pointer within segment
    pub(crate) ptr: *mut T,
    /// End of current segment (min of segment capacity and vec.len)
    pub(crate) segment_end: *mut T,
    /// Current logical index
    pub(crate) index: usize,
    /// Current segment index
    pub(crate) segment_index: usize,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // For ZSTs, just check index vs len
        if std::mem::size_of::<T>() == 0 {
            if self.index < self.vec.len() {
                self.index += 1;
                return Some(unsafe { &mut *std::ptr::NonNull::dangling().as_ptr() });
            }
            return None;
        }

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
        let remaining = self.vec.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a, T> IterMut<'a, T> {
    #[cold]
    fn next_segment(&mut self) -> Option<&'a mut T> {
        if self.index >= self.vec.len() {
            return None;
        }

        let segment_size = RawSegmentedVec::<T>::segment_capacity(self.segment_index);
        let ptr = unsafe { self.vec.buf.segment_ptr(self.segment_index) };
        let segment_len = segment_size.min(self.vec.len() - self.index);
        self.ptr = ptr;
        self.segment_end = unsafe { ptr.add(segment_len) };
        self.segment_index += 1;

        let result = self.ptr;
        self.ptr = unsafe { self.ptr.add(1) };
        self.index += 1;
        Some(unsafe { &mut *result })
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> std::iter::FusedIterator for IterMut<'_, T> {}

// Safety: IterMut yields exclusive references
unsafe impl<T: Send> Send for IterMut<'_, T> {}
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}
