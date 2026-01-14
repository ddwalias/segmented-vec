use crate::raw_vec::RawSegmentedVec;
use crate::SegmentedVec;
use allocator_api2::alloc::Global;
use core::marker::PhantomData;

/// Helper trait for FromIterator implementation
/// Uses size_hint to optimize when exact size is known
pub(crate) trait SpecFromIter<T, I> {
    fn from_iter(iter: I) -> SegmentedVec<T>;
}

impl<T, I> SpecFromIter<T, I> for SegmentedVec<T>
where
    I: Iterator<Item = T>,
{
    #[inline]
    fn from_iter(iter: I) -> SegmentedVec<T> {
        let (lower, upper) = iter.size_hint();

        // Check if we have an exact size (lower == upper)
        // This is true for ExactSizeIterator and many common iterators
        if let Some(upper) = upper {
            if lower == upper {
                // Fast path: we know the exact size, use incremental allocation
                return unsafe { from_iter_exact_size(iter, lower) };
            }
        }

        // Fallback path: use extend
        from_iter_fallback(iter, lower)
    }
}

/// Fast path for iterators with known exact size.
/// Uses incremental allocation (grow_one) for optimal performance at small sizes.
#[inline]
unsafe fn from_iter_exact_size<T, I: Iterator<Item = T>>(
    mut iter: I,
    length: usize,
) -> SegmentedVec<T> {
    if length == 0 {
        return SegmentedVec::new();
    }

    // Allocate incrementally using grow_one
    let mut buf: RawSegmentedVec<T, Global> = RawSegmentedVec::new_in(Global);

    let mut i = 0;
    let mut write_ptr: *mut T = core::ptr::null_mut();
    let mut segment_end: *mut T = core::ptr::null_mut();

    while i < length {
        // grow_one returns (segment_ptr, segment_capacity)
        let (seg_ptr, seg_cap) = buf.grow_one();

        let remaining = length - i;
        let count = if remaining < seg_cap {
            remaining
        } else {
            seg_cap
        };

        for offset in 0..count {
            // SAFETY: We verified exact size from size_hint, so iter.next()
            // is guaranteed to return Some for `length` iterations
            let element = unsafe { iter.next().unwrap_unchecked() };
            core::ptr::write(seg_ptr.add(offset), element);
            i += 1;
        }

        write_ptr = seg_ptr.add(count);
        segment_end = seg_ptr.add(seg_cap);
    }

    let active_segment_index = buf.segment_count().saturating_sub(1);

    SegmentedVec {
        buf,
        len: length,
        write_ptr,
        segment_end,
        active_segment_index,
        _marker: PhantomData,
    }
}

/// Fallback path for iterators without known exact size
#[inline(never)]
#[cold]
fn from_iter_fallback<T, I: Iterator<Item = T>>(iter: I, lower: usize) -> SegmentedVec<T> {
    let mut vec = SegmentedVec::with_capacity(lower);
    vec.extend(iter);
    vec
}
