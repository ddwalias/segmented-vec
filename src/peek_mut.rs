//! PeekMut implementation for SegmentedVec.

use core::ops::{Deref, DerefMut};

use allocator_api2::alloc::{Allocator, Global};

use crate::SegmentedVec;

/// Structure wrapping a mutable reference to the last item in a
/// `SegmentedVec`.
///
/// This `struct` is created by the [`peek_mut`] method on [`SegmentedVec`]. See
/// its documentation for more.
///
/// [`peek_mut`]: SegmentedVec::peek_mut
pub struct PeekMut<'a, T, A: Allocator = Global> {
    vec: &'a mut SegmentedVec<T, A>,
}

impl<T: std::fmt::Debug, A: Allocator> std::fmt::Debug for PeekMut<'_, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PeekMut").field(&**self).finish()
    }
}

impl<'a, T, A: Allocator> PeekMut<'a, T, A> {
    /// Creates a new `PeekMut` if the vector is non-empty.
    pub(crate) fn new(vec: &'a mut SegmentedVec<T, A>) -> Option<Self> {
        if vec.is_empty() {
            None
        } else {
            Some(Self { vec })
        }
    }

    /// Removes the peeked value from the vector and returns it.
    pub fn pop(this: Self) -> T {
        // Safety: PeekMut is only constructed if the vec is non-empty
        this.vec.pop().unwrap()
    }
}

impl<T, A: Allocator> Deref for PeekMut<'_, T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: PeekMut is only constructed if the vec is non-empty
        self.vec.last().unwrap()
    }
}

impl<T, A: Allocator> DerefMut for PeekMut<'_, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: PeekMut is only constructed if the vec is non-empty
        self.vec.last_mut().unwrap()
    }
}
