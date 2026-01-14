//! Owning iterator for `SegmentedVec`.

use crate::SegmentedVec;

/// An owning iterator over elements of a `SegmentedVec`.
///
/// This struct is created by the `into_iter` method on `SegmentedVec`
/// (provided by the [`IntoIterator`] trait).
pub struct IntoIter<T> {
    pub(crate) vec: SegmentedVec<T>,
    pub(crate) index: usize,
}

impl<T> IntoIter<T> {
    /// Creates a new owning iterator from a `SegmentedVec`.
    #[inline]
    pub fn new(vec: SegmentedVec<T>) -> Self {
        Self { vec, index: 0 }
    }

    /// Returns the remaining items as a slice.
    ///
    /// Note: For segmented storage, this returns an empty slice since
    /// elements are not stored contiguously. Use iteration instead.
    pub fn as_slice(&self) -> &[T] {
        // We can't return a contiguous slice for segmented storage
        &[]
    }

    /// Returns the remaining items as a mutable slice.
    ///
    /// Note: For segmented storage, this returns an empty slice since
    /// elements are not stored contiguously. Use iteration instead.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // We can't return a contiguous slice for segmented storage
        &mut []
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.len() {
            return None;
        }
        // Safety: index < len, so the element exists and is initialized
        let value = unsafe { self.vec.unchecked_read(self.index) };
        self.index += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len() - self.index;
        (remaining, Some(remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.vec.len() - self.index
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.len() {
            return None;
        }
        let new_len = self.vec.len() - 1;
        // Safety: new_len < old len, so this index is valid
        let value = unsafe { self.vec.unchecked_read(new_len) };
        // Safety: new_len is less than capacity and we just read (not dropped)
        // the element at new_len, so it's safe to shrink
        unsafe { self.vec.set_len_internal(new_len) };
        Some(value)
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> std::iter::FusedIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // Drop remaining elements that weren't consumed
        if std::mem::needs_drop::<T>() {
            for i in self.index..self.vec.len() {
                unsafe {
                    std::ptr::drop_in_place(self.vec.unchecked_at_mut(i));
                }
            }
        }
        // Safety: All remaining elements have been dropped, set len to 0
        // to prevent the Vec from dropping them again
        unsafe { self.vec.set_len_internal(0) };
    }
}

// Safety: IntoIter owns the data
unsafe impl<T: Send> Send for IntoIter<T> {}
unsafe impl<T: Sync> Sync for IntoIter<T> {}

impl<T: Clone> Clone for IntoIter<T> {
    fn clone(&self) -> Self {
        // Create a new vec with the remaining elements
        let mut new_vec = SegmentedVec::new();
        for i in self.index..self.vec.len() {
            new_vec.push(unsafe { self.vec.unchecked_at(i).clone() });
        }
        IntoIter {
            vec: new_vec,
            index: 0,
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntoIter")
            .field("remaining", &(self.vec.len() - self.index))
            .finish()
    }
}
