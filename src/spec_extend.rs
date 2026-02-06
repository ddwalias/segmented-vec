//! Specialization for extend operations using runtime size_hint checking.
//!
//! Uses size_hint() to detect when exact size is known (lower == upper),
//! and applies incremental allocation strategy similar to from_fn_optimized.

use crate::SegmentedVec;
use allocator_api2::alloc::Allocator;
use std::ptr::NonNull;

/// Extend using runtime size_hint check.
/// If size_hint provides an exact count (lower == upper), use incremental allocation.
#[inline]
pub(crate) fn extend_with_size_hint_check<T, A: Allocator, I: Iterator<Item = T>>(
    vec: &mut SegmentedVec<T, A>,
    iter: I,
) {
    let (lower, upper) = iter.size_hint();

    // If we have an exact size, use the optimized path
    if let Some(upper) = upper {
        if lower == upper && lower > 0 {
            extend_with_exact_count(vec, iter, lower);
            return;
        }
    }

    // Fallback to desugared extend
    vec.extend_desugared(iter);
}

/// Extend with a known exact count using incremental allocation.
/// Uses grow_one() to allocate segments incrementally like from_fn_optimized.
#[inline]
fn extend_with_exact_count<T, A: Allocator, I: Iterator<Item = T>>(
    vec: &mut SegmentedVec<T, A>,
    mut iter: I,
    count: usize,
) {
    if count == 0 {
        return;
    }

    if std::mem::size_of::<T>() == 0 {
        // For ZST, we just need to consume the iterator and update len
        for _ in 0..count {
            if iter.next().is_none() {
                break;
            }
            vec.len += 1;
        }
        return;
    }

    let mut remaining = count;

    unsafe {
        while remaining > 0 {
            // Check if current segment has space
            if vec.write_ptr >= vec.segment_end {
                // Check if we have a next segment available
                // Note: active_segment_index is initialized to usize::MAX when empty,
                // so we use wrapping_add to check for segment 0.
                let next_seg = vec.active_segment_index.wrapping_add(1);
                if next_seg < vec.buf.segment_count() {
                    // Reuse existing next segment
                    vec.active_segment_index = next_seg;
                    let ptr = vec.buf.segment_ptr(next_seg);
                    vec.write_ptr = NonNull::new_unchecked(ptr);
                    let cap = crate::RawSegmentedVec::<T, A>::segment_capacity(next_seg);
                    vec.segment_end = NonNull::new_unchecked(ptr.add(cap));
                } else {
                    // Need a new segment
                    // grow_one returns (segment_ptr, segment_capacity)
                    let (seg_ptr, seg_cap) = vec.buf.grow_one();
                    let seg_idx = vec.buf.segment_count() - 1;
                    vec.write_ptr = NonNull::new_unchecked(seg_ptr);
                    vec.segment_end = NonNull::new_unchecked(seg_ptr.add(seg_cap));
                    vec.active_segment_index = seg_idx;
                }
            }

            // Calculate how many elements fit in current segment
            // Use as_ptr() for pointer arithmetic
            let space = vec.segment_end.as_ptr().offset_from(vec.write_ptr.as_ptr()) as usize;
            let batch = std::cmp::min(remaining, space);

            // Fill current segment
            for _ in 0..batch {
                // SAFETY: We verified exact size from size_hint
                let element = iter.next().unwrap_unchecked();
                std::ptr::write(vec.write_ptr.as_ptr(), element);
                vec.write_ptr = NonNull::new_unchecked(vec.write_ptr.as_ptr().add(1));
            }

            vec.len += batch;
            remaining -= batch;
        }
    }
}
