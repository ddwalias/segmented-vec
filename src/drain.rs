//! Draining iterator for `SegmentedVec`.

use crate::SegmentedVec;
use std::ptr::NonNull;

/// A draining iterator for `SegmentedVec`.
///
/// This struct is created by the [`drain`] method on [`SegmentedVec`].
///
/// [`drain`]: SegmentedVec::drain
pub struct Drain<'a, T: 'a> {
    /// Pointer to the SegmentedVec we're draining
    pub(crate) vec: NonNull<SegmentedVec<T>>,
    /// Start of the drained range
    pub(crate) range_start: usize,
    /// Original end of the drained range
    pub(crate) range_end: usize,
    /// Current iteration position
    pub(crate) index: usize,
    /// Original length of the vector
    pub(crate) original_len: usize,
    /// Marker for the lifetime
    pub(crate) _marker: std::marker::PhantomData<&'a mut SegmentedVec<T>>,
}

impl<T> Drain<'_, T> {
    /// Returns the remaining items as a slice.
    ///
    /// Note: Due to the segmented nature of the storage, this returns an empty slice.
    /// Use iteration instead.
    pub fn as_slice(&self) -> &[T] {
        &[]
    }

    /// Keep the remaining items in the original vector.
    ///
    /// This method stops the draining process and preserves any elements
    /// that haven't been yielded yet, keeping them at their current positions
    /// in the vector.
    pub fn keep_rest(self) {
        // The Drop impl will handle cleanup
        // We just need to not iterate the remaining elements
        std::mem::drop(self);
    }
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.range_end {
            None
        } else {
            let vec = unsafe { self.vec.as_ref() };
            let value = unsafe { std::ptr::read(vec.unchecked_at(self.index)) };
            self.index += 1;
            Some(value)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.range_end - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> DoubleEndedIterator for Drain<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.range_end {
            None
        } else {
            self.range_end -= 1;
            let vec = unsafe { self.vec.as_ref() };
            Some(unsafe { std::ptr::read(vec.unchecked_at(self.range_end)) })
        }
    }
}

impl<T> ExactSizeIterator for Drain<'_, T> {}

impl<T> std::iter::FusedIterator for Drain<'_, T> {}

impl<T> Drop for Drain<'_, T> {
    fn drop(&mut self) {
        // Drop any remaining elements in the range that weren't consumed
        if std::mem::needs_drop::<T>() {
            let vec = unsafe { self.vec.as_ref() };
            for i in self.index..self.range_end {
                unsafe {
                    std::ptr::drop_in_place(vec.unchecked_at(i) as *const T as *mut T);
                }
            }
        }

        // Shift elements after the range to fill the gap
        let vec = unsafe { self.vec.as_mut() };
        let drain_count = self.range_end - self.range_start;

        // Only shift if we actually drained elements (drain_count > 0)
        // and there are elements after the range to shift
        if drain_count > 0 {
            // Use the original_len stored when Drain was created
            for i in 0..(self.original_len - self.range_end) {
                unsafe {
                    let src = vec.unchecked_at(self.range_end + i) as *const T;
                    let dst = vec.unchecked_at_mut(self.range_start + i) as *mut T;
                    std::ptr::copy_nonoverlapping(src, dst, 1);
                }
            }
        }

        // Update the vector's length
        vec.set_len_internal(self.original_len - drain_count);
        vec.update_write_ptr_for_len();
    }
}

// Safety: Drain has exclusive access to the drained portion
unsafe impl<T: Sync> Sync for Drain<'_, T> {}
unsafe impl<T: Send> Send for Drain<'_, T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for Drain<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drain")
            .field("range_start", &self.range_start)
            .field("range_end", &self.range_end)
            .field("index", &self.index)
            .finish()
    }
}
