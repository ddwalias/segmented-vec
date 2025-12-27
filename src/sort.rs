//! Sorting algorithms for SegmentedVec.
//!
//! This module contains implementations of sorting algorithms adapted from the
//! Rust standard library to work with non-contiguous memory.

use std::mem::MaybeUninit;
use std::ptr;

/// Threshold for switching to insertion sort.
const INSERTION_SORT_THRESHOLD: usize = 20;

/// Sorts `v[start..end]` using insertion sort.
///
/// Adapted from the Rust standard library's `insertion_sort_shift_left`.
#[inline]
pub fn insertion_sort<T, F>(
    v: &mut impl IndexedAccess<T>,
    start: usize,
    end: usize,
    is_less: &mut F,
)
where
    F: FnMut(&T, &T) -> bool,
{
    for i in (start + 1)..end {
        // Insert v[i] into the sorted sequence v[start..i].
        let mut j = i;
        while j > start && is_less(v.get_ref(j), v.get_ref(j - 1)) {
            v.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Sorts using heapsort. Guarantees O(n log n) worst-case.
///
/// Adapted from the Rust standard library's heapsort.
#[inline(never)]
pub fn heapsort<T, F>(v: &mut impl IndexedAccess<T>, start: usize, end: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = end - start;
    if len < 2 {
        return;
    }

    // Build the heap in-place.
    for i in (0..len / 2).rev() {
        sift_down(v, start, i, len, is_less);
    }

    // Pop elements from the heap one by one.
    for i in (1..len).rev() {
        v.swap(start, start + i);
        sift_down(v, start, 0, i, is_less);
    }
}

/// Sift down element at `node` in heap rooted at `start` with size `heap_size`.
#[inline]
fn sift_down<T, F>(
    v: &mut impl IndexedAccess<T>,
    start: usize,
    mut node: usize,
    heap_size: usize,
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    loop {
        let mut child = 2 * node + 1;
        if child >= heap_size {
            break;
        }

        // Choose the greater child.
        if child + 1 < heap_size && is_less(v.get_ref(start + child), v.get_ref(start + child + 1))
        {
            child += 1;
        }

        // Stop if the invariant holds.
        if !is_less(v.get_ref(start + node), v.get_ref(start + child)) {
            break;
        }

        v.swap(start + node, start + child);
        node = child;
    }
}

/// Sorts using quicksort with heapsort fallback.
///
/// Adapted from the Rust standard library's quicksort implementation.
pub fn quicksort<T, F>(v: &mut impl IndexedAccess<T>, start: usize, end: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = end - start;
    if len < 2 {
        return;
    }

    // Use insertion sort for small arrays.
    if len <= INSERTION_SORT_THRESHOLD {
        insertion_sort(v, start, end, is_less);
        return;
    }

    // Limit recursion depth to 2 * log2(len) to guarantee O(n log n) worst-case.
    let limit = 2 * (usize::BITS - len.leading_zeros());
    quicksort_recursive(v, start, end, is_less, limit);
}

/// Recursive quicksort with recursion limit.
fn quicksort_recursive<T, F>(
    v: &mut impl IndexedAccess<T>,
    start: usize,
    end: usize,
    is_less: &mut F,
    mut limit: u32,
) where
    F: FnMut(&T, &T) -> bool,
{
    let mut start = start;
    let mut end = end;

    loop {
        let len = end - start;

        if len <= INSERTION_SORT_THRESHOLD {
            insertion_sort(v, start, end, is_less);
            return;
        }

        // If we've hit the recursion limit, fall back to heapsort.
        if limit == 0 {
            heapsort(v, start, end, is_less);
            return;
        }
        limit -= 1;

        // Choose pivot using median-of-three.
        let mid = start + len / 2;
        let pivot_idx = choose_pivot(v, start, mid, end - 1, is_less);

        // Move pivot to the start.
        v.swap(start, pivot_idx);

        // Partition around the pivot.
        let pivot_final = partition(v, start, end, is_less);

        // Recurse on the smaller partition first to limit stack depth.
        let left_len = pivot_final - start;
        let right_len = end - pivot_final - 1;

        if left_len < right_len {
            quicksort_recursive(v, start, pivot_final, is_less, limit);
            start = pivot_final + 1;
        } else {
            quicksort_recursive(v, pivot_final + 1, end, is_less, limit);
            end = pivot_final;
        }
    }
}

/// Choose pivot using median-of-three.
#[inline]
fn choose_pivot<T, F>(
    v: &impl IndexedAccess<T>,
    a: usize,
    b: usize,
    c: usize,
    is_less: &mut F,
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // Return the index of the median of the three elements.
    if is_less(v.get_ref(a), v.get_ref(b)) {
        if is_less(v.get_ref(b), v.get_ref(c)) {
            b
        } else if is_less(v.get_ref(a), v.get_ref(c)) {
            c
        } else {
            a
        }
    } else if is_less(v.get_ref(a), v.get_ref(c)) {
        a
    } else if is_less(v.get_ref(b), v.get_ref(c)) {
        c
    } else {
        b
    }
}

/// Hoare partition scheme.
///
/// Partitions `v[start..end]` around the pivot at `v[start]`.
/// Returns the final position of the pivot.
fn partition<T, F>(v: &mut impl IndexedAccess<T>, start: usize, end: usize, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let mut left = start + 1;
    let mut right = end - 1;

    loop {
        // Move left pointer right while elements are less than pivot.
        while left <= right && is_less(v.get_ref(left), v.get_ref(start)) {
            left += 1;
        }

        // Move right pointer left while elements are greater than or equal to pivot.
        while left <= right && !is_less(v.get_ref(right), v.get_ref(start)) {
            right -= 1;
        }

        if left > right {
            break;
        }

        v.swap(left, right);
        left += 1;
        right -= 1;
    }

    // Move pivot to its final position.
    v.swap(start, right);
    right
}

/// Stable merge sort implementation.
///
/// Adapted from the Rust standard library's merge sort.
pub fn merge_sort<T, F>(v: &mut impl IndexedAccess<T>, start: usize, end: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = end - start;
    if len < 2 {
        return;
    }

    // Use insertion sort for small arrays.
    if len <= INSERTION_SORT_THRESHOLD {
        insertion_sort(v, start, end, is_less);
        return;
    }

    // Allocate scratch space for merging.
    let scratch_len = len / 2 + 1;
    let mut scratch: Vec<MaybeUninit<T>> = Vec::with_capacity(scratch_len);
    // SAFETY: We're only using this as uninitialized storage.
    unsafe {
        scratch.set_len(scratch_len);
    }

    merge_sort_with_scratch(v, start, end, &mut scratch, is_less);
}

/// Merge sort with provided scratch space.
fn merge_sort_with_scratch<T, F>(
    v: &mut impl IndexedAccess<T>,
    start: usize,
    end: usize,
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    let len = end - start;
    if len < 2 {
        return;
    }

    if len <= INSERTION_SORT_THRESHOLD {
        insertion_sort(v, start, end, is_less);
        return;
    }

    let mid = start + len / 2;

    // Recursively sort both halves.
    merge_sort_with_scratch(v, start, mid, scratch, is_less);
    merge_sort_with_scratch(v, mid, end, scratch, is_less);

    // If already sorted, we're done.
    if !is_less(v.get_ref(mid), v.get_ref(mid - 1)) {
        return;
    }

    // Merge the two sorted halves.
    merge(v, start, mid, end, scratch, is_less);
}

/// Merges two sorted runs: `v[start..mid]` and `v[mid..end]`.
#[allow(clippy::needless_range_loop)]
fn merge<T, F>(
    v: &mut impl IndexedAccess<T>,
    start: usize,
    mid: usize,
    end: usize,
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    let left_len = mid - start;
    let right_len = end - mid;

    // Copy the shorter half to scratch.
    if left_len <= right_len {
        // Copy left half to scratch.
        for i in 0..left_len {
            // SAFETY: We're moving the value to scratch.
            unsafe {
                let val = ptr::read(v.get_ptr(start + i));
                scratch[i].write(val);
            }
        }

        // Merge back.
        let mut s = 0; // scratch index
        let mut r = mid; // right index
        let mut w = start; // write index

        while s < left_len && r < end {
            // SAFETY: scratch[s] is initialized, v.get_ptr(r) is valid.
            let take_left = unsafe { !is_less(v.get_ref(r), scratch[s].assume_init_ref()) };

            if take_left {
                // SAFETY: Moving from scratch.
                unsafe {
                    ptr::write(v.get_ptr_mut(w), scratch[s].assume_init_read());
                }
                s += 1;
            } else {
                // SAFETY: Moving within v.
                unsafe {
                    let val = ptr::read(v.get_ptr(r));
                    ptr::write(v.get_ptr_mut(w), val);
                }
                r += 1;
            }
            w += 1;
        }

        // Copy remaining elements from scratch.
        while s < left_len {
            unsafe {
                ptr::write(v.get_ptr_mut(w), scratch[s].assume_init_read());
            }
            s += 1;
            w += 1;
        }
        // Remaining elements from right are already in place.
    } else {
        // Copy right half to scratch.
        for i in 0..right_len {
            unsafe {
                let val = ptr::read(v.get_ptr(mid + i));
                scratch[i].write(val);
            }
        }

        // Merge back from the end.
        let mut s = right_len; // scratch index (exclusive, counting down)
        let mut l = mid; // left index (exclusive, counting down)
        let mut w = end; // write index (exclusive, counting down)

        while s > 0 && l > start {
            // SAFETY: scratch[s-1] is initialized, v.get_ptr(l-1) is valid.
            let take_right =
                unsafe { is_less(scratch[s - 1].assume_init_ref(), v.get_ref(l - 1)) };

            w -= 1;
            if take_right {
                // SAFETY: Moving from scratch.
                s -= 1;
                unsafe {
                    ptr::write(v.get_ptr_mut(w), scratch[s].assume_init_read());
                }
            } else {
                // SAFETY: Moving within v.
                l -= 1;
                unsafe {
                    let val = ptr::read(v.get_ptr(l));
                    ptr::write(v.get_ptr_mut(w), val);
                }
            }
        }

        // Copy remaining elements from scratch.
        while s > 0 {
            s -= 1;
            w -= 1;
            unsafe {
                ptr::write(v.get_ptr_mut(w), scratch[s].assume_init_read());
            }
        }
        // Remaining elements from left are already in place.
    }
}

/// Trait for indexed access to a collection.
///
/// This abstraction allows the sorting algorithms to work with
/// non-contiguous memory layouts like SegmentedVec.
pub trait IndexedAccess<T> {
    /// Get a reference to the element at index.
    fn get_ref(&self, index: usize) -> &T;

    /// Get a raw pointer to the element at index.
    fn get_ptr(&self, index: usize) -> *const T;

    /// Get a mutable raw pointer to the element at index.
    fn get_ptr_mut(&mut self, index: usize) -> *mut T;

    /// Swap elements at two indices.
    fn swap(&mut self, a: usize, b: usize);
}
