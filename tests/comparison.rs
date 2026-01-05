//! Comparison tests between SegmentedVec and std::Vec
//!
//! This module provides property-based testing that compares the behavior of
//! SegmentedVec with std::Vec to automatically catch behavioral discrepancies.

use proptest::prelude::*;
use segmented_vec::SegmentedVec;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ============================================================================
// COMPARISON TESTING INFRASTRUCTURE
// ============================================================================

/// A trait that abstracts common operations for comparison testing.
/// Both Vec<T> and SegmentedVec<T> implement these operations.
#[allow(dead_code)]
trait VecLike<T> {
    fn new_vec() -> Self;
    fn push_val(&mut self, value: T);
    fn pop_val(&mut self) -> Option<T>;
    fn len_val(&self) -> usize;
    fn is_empty_val(&self) -> bool;
    fn get_val(&self, index: usize) -> Option<&T>;
    fn get_mut_val(&mut self, index: usize) -> Option<&mut T>;
    fn first_val(&self) -> Option<&T>;
    fn last_val(&self) -> Option<&T>;
    fn clear_val(&mut self);
    fn truncate_val(&mut self, len: usize);
    fn insert_val(&mut self, index: usize, value: T);
    fn remove_val(&mut self, index: usize) -> T;
    fn swap_remove_val(&mut self, index: usize) -> T;
    fn swap_vals(&mut self, a: usize, b: usize);
    fn reverse_val(&mut self);
    fn extend_val<I: IntoIterator<Item = T>>(&mut self, iter: I);
    fn to_vec_val(&self) -> Vec<T>
    where
        T: Clone;
}

impl<T> VecLike<T> for Vec<T> {
    fn new_vec() -> Self {
        Vec::new()
    }
    fn push_val(&mut self, value: T) {
        self.push(value);
    }
    fn pop_val(&mut self) -> Option<T> {
        self.pop()
    }
    fn len_val(&self) -> usize {
        self.len()
    }
    fn is_empty_val(&self) -> bool {
        self.is_empty()
    }
    fn get_val(&self, index: usize) -> Option<&T> {
        self.get(index)
    }
    fn get_mut_val(&mut self, index: usize) -> Option<&mut T> {
        self.get_mut(index)
    }
    fn first_val(&self) -> Option<&T> {
        self.first()
    }
    fn last_val(&self) -> Option<&T> {
        self.last()
    }
    fn clear_val(&mut self) {
        self.clear();
    }
    fn truncate_val(&mut self, len: usize) {
        self.truncate(len);
    }
    fn insert_val(&mut self, index: usize, value: T) {
        self.insert(index, value);
    }
    fn remove_val(&mut self, index: usize) -> T {
        self.remove(index)
    }
    fn swap_remove_val(&mut self, index: usize) -> T {
        self.swap_remove(index)
    }
    fn swap_vals(&mut self, a: usize, b: usize) {
        self.swap(a, b);
    }
    fn reverse_val(&mut self) {
        self.reverse();
    }
    fn extend_val<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.extend(iter);
    }
    fn to_vec_val(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.clone()
    }
}

impl<T> VecLike<T> for SegmentedVec<T> {
    fn new_vec() -> Self {
        SegmentedVec::new()
    }
    fn push_val(&mut self, value: T) {
        self.push(value);
    }
    fn pop_val(&mut self) -> Option<T> {
        self.pop()
    }
    fn len_val(&self) -> usize {
        self.len()
    }
    fn is_empty_val(&self) -> bool {
        self.is_empty()
    }
    fn get_val(&self, index: usize) -> Option<&T> {
        self.get(index)
    }
    fn get_mut_val(&mut self, index: usize) -> Option<&mut T> {
        self.get_mut(index)
    }
    fn first_val(&self) -> Option<&T> {
        self.first()
    }
    fn last_val(&self) -> Option<&T> {
        self.last()
    }
    fn clear_val(&mut self) {
        self.clear();
    }
    fn truncate_val(&mut self, len: usize) {
        self.truncate(len);
    }
    fn insert_val(&mut self, index: usize, value: T) {
        self.insert(index, value);
    }
    fn remove_val(&mut self, index: usize) -> T {
        self.remove(index)
    }
    fn swap_remove_val(&mut self, index: usize) -> T {
        self.swap_remove(index)
    }
    fn swap_vals(&mut self, a: usize, b: usize) {
        self.swap(a, b);
    }
    fn reverse_val(&mut self) {
        self.reverse();
    }
    fn extend_val<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.extend(iter);
    }
    fn to_vec_val(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_vec()
    }
}

/// Operations that can be applied to a vector for comparison testing.
#[derive(Debug, Clone)]
enum VecOp<T> {
    Push(T),
    Pop,
    Clear,
    Truncate(usize),
    Insert(usize, T),
    Remove(usize),
    SwapRemove(usize),
    Swap(usize, usize),
    Reverse,
    Extend(Vec<T>),
}

/// Apply an operation to both vectors and compare results.
fn apply_op<T: Clone + PartialEq + std::fmt::Debug>(
    std_vec: &mut Vec<T>,
    seg_vec: &mut SegmentedVec<T>,
    op: &VecOp<T>,
) {
    match op {
        VecOp::Push(v) => {
            std_vec.push_val(v.clone());
            seg_vec.push_val(v.clone());
        }
        VecOp::Pop => {
            let std_result = std_vec.pop_val();
            let seg_result = seg_vec.pop_val();
            assert_eq!(std_result, seg_result, "pop() mismatch");
        }
        VecOp::Clear => {
            std_vec.clear_val();
            seg_vec.clear_val();
        }
        VecOp::Truncate(len) => {
            std_vec.truncate_val(*len);
            seg_vec.truncate_val(*len);
        }
        VecOp::Insert(idx, v) => {
            if *idx <= std_vec.len() {
                std_vec.insert_val(*idx, v.clone());
                seg_vec.insert_val(*idx, v.clone());
            }
        }
        VecOp::Remove(idx) => {
            if *idx < std_vec.len() && !std_vec.is_empty() {
                let std_result = std_vec.remove_val(*idx);
                let seg_result = seg_vec.remove_val(*idx);
                assert_eq!(std_result, seg_result, "remove() mismatch");
            }
        }
        VecOp::SwapRemove(idx) => {
            if *idx < std_vec.len() && !std_vec.is_empty() {
                let std_result = std_vec.swap_remove_val(*idx);
                let seg_result = seg_vec.swap_remove_val(*idx);
                assert_eq!(std_result, seg_result, "swap_remove() mismatch");
            }
        }
        VecOp::Swap(a, b) => {
            if *a < std_vec.len() && *b < std_vec.len() {
                std_vec.swap_vals(*a, *b);
                seg_vec.swap_vals(*a, *b);
            }
        }
        VecOp::Reverse => {
            std_vec.reverse_val();
            seg_vec.reverse_val();
        }
        VecOp::Extend(vals) => {
            std_vec.extend_val(vals.clone());
            seg_vec.extend_val(vals.clone());
        }
    }
}

/// Verify that both vectors have the same content.
fn assert_vecs_equal<T: Clone + PartialEq + std::fmt::Debug>(
    std_vec: &[T],
    seg_vec: &SegmentedVec<T>,
) {
    assert_eq!(std_vec.len(), seg_vec.len(), "length mismatch");
    assert_eq!(std_vec.is_empty(), seg_vec.is_empty(), "is_empty mismatch");

    // Compare element by element
    for (i, (std_elem, seg_elem)) in std_vec.iter().zip(seg_vec.iter()).enumerate() {
        assert_eq!(std_elem, seg_elem, "element mismatch at index {}", i);
    }

    // Compare first/last
    assert_eq!(std_vec.first(), seg_vec.first(), "first() mismatch");
    assert_eq!(std_vec.last(), seg_vec.last(), "last() mismatch");

    // Compare get() for all indices
    for i in 0..std_vec.len() {
        assert_eq!(std_vec.get(i), seg_vec.get(i), "get({}) mismatch", i);
    }

    // Out of bounds should return None
    assert_eq!(std_vec.get(std_vec.len()), seg_vec.get(seg_vec.len()));
    assert_eq!(std_vec.get(usize::MAX), seg_vec.get(usize::MAX));
}

// ============================================================================
// PROPTEST STRATEGIES
// ============================================================================

/// Strategy for generating a single vector operation.
fn vec_op_strategy() -> impl Strategy<Value = VecOp<i32>> {
    prop_oneof![
        // Push with various values
        any::<i32>().prop_map(VecOp::Push),
        // Pop
        Just(VecOp::Pop),
        // Clear
        Just(VecOp::Clear),
        // Truncate to random length
        (0usize..1000).prop_map(VecOp::Truncate),
        // Insert at random position
        (0usize..100, any::<i32>()).prop_map(|(idx, v)| VecOp::Insert(idx, v)),
        // Remove at random position
        (0usize..100).prop_map(VecOp::Remove),
        // SwapRemove at random position
        (0usize..100).prop_map(VecOp::SwapRemove),
        // Swap two positions
        (0usize..100, 0usize..100).prop_map(|(a, b)| VecOp::Swap(a, b)),
        // Reverse
        Just(VecOp::Reverse),
        // Extend with random values
        prop::collection::vec(any::<i32>(), 0..50).prop_map(VecOp::Extend),
    ]
}

/// Strategy for generating a sequence of operations.
fn ops_sequence_strategy() -> impl Strategy<Value = Vec<VecOp<i32>>> {
    prop::collection::vec(vec_op_strategy(), 0..200)
}

// ============================================================================
// PROPTEST TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Test that a random sequence of operations produces identical results.
    #[test]
    fn proptest_operations_match(ops in ops_sequence_strategy()) {
        let mut std_vec: Vec<i32> = Vec::new();
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();

        for op in &ops {
            apply_op(&mut std_vec, &mut seg_vec, op);
            assert_vecs_equal(&std_vec, &seg_vec);
        }
    }

    /// Test push followed by iteration.
    #[test]
    fn proptest_push_and_iter(values in prop::collection::vec(any::<i32>(), 0..500)) {
        let mut std_vec: Vec<i32> = Vec::new();
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();

        for v in &values {
            std_vec.push(*v);
            seg_vec.push(*v);
        }

        // Compare using iterator
        let std_collected: Vec<_> = std_vec.to_vec();
        let seg_collected: Vec<_> = seg_vec.iter().copied().collect();
        prop_assert_eq!(std_collected, seg_collected);

        // Compare using into_iter
        let std_into: Vec<_> = std_vec.clone().into_iter().collect();
        let seg_into: Vec<_> = seg_vec.clone().into_iter().collect();
        prop_assert_eq!(std_into, seg_into);
    }

    /// Test that sort produces the same result.
    #[test]
    fn proptest_sort(values in prop::collection::vec(any::<i32>(), 0..200)) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.sort();
        seg_vec.sort();

        let std_sorted: Vec<_> = std_vec.to_vec();
        let seg_sorted: Vec<_> = seg_vec.iter().copied().collect();
        prop_assert_eq!(std_sorted, seg_sorted);
    }

    /// Test that sort_unstable produces a correctly sorted result (order may differ for equal elements).
    #[test]
    fn proptest_sort_unstable(values in prop::collection::vec(any::<i32>(), 0..200)) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.sort_unstable();
        seg_vec.sort_unstable();

        // Both should be sorted, but order of equal elements may differ
        let std_sorted: Vec<_> = std_vec.to_vec();
        let seg_sorted: Vec<_> = seg_vec.iter().copied().collect();

        // Both should be sorted
        prop_assert!(std_sorted.windows(2).all(|w| w[0] <= w[1]));
        prop_assert!(seg_sorted.windows(2).all(|w| w[0] <= w[1]));

        // Both should have the same elements (when sorted)
        prop_assert_eq!(std_sorted, seg_sorted);
    }

    /// Test is_sorted consistency.
    #[test]
    fn proptest_is_sorted(values in prop::collection::vec(any::<i32>(), 0..200)) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        prop_assert_eq!(std_vec.is_sorted(), seg_vec.is_sorted());
    }

    /// Test binary_search consistency.
    #[test]
    fn proptest_binary_search(
        values in prop::collection::vec(any::<i32>(), 0..100),
        search_val in any::<i32>()
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.sort();
        seg_vec.sort();

        let std_result = std_vec.binary_search(&search_val);
        let seg_result = seg_vec.binary_search(&search_val);

        // Results should match exactly
        prop_assert_eq!(std_result, seg_result);
    }

    /// Test contains consistency.
    #[test]
    fn proptest_contains(
        values in prop::collection::vec(any::<i32>(), 0..100),
        search_val in any::<i32>()
    ) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        prop_assert_eq!(std_vec.contains(&search_val), seg_vec.contains(&search_val));
    }

    /// Test starts_with and ends_with consistency.
    #[test]
    fn proptest_starts_ends_with(
        values in prop::collection::vec(any::<i32>(), 0..100),
        needle_len in 0usize..20
    ) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        if !std_vec.is_empty() {
            let actual_len = needle_len.min(std_vec.len());
            let prefix: Vec<i32> = std_vec[..actual_len].to_vec();
            let suffix: Vec<i32> = std_vec[std_vec.len() - actual_len..].to_vec();

            prop_assert_eq!(
                std_vec.starts_with(&prefix),
                seg_vec.starts_with(&prefix)
            );
            prop_assert_eq!(
                std_vec.ends_with(&suffix),
                seg_vec.ends_with(&suffix)
            );
        }
    }

    /// Test drain consistency.
    #[test]
    fn proptest_drain(
        values in prop::collection::vec(any::<i32>(), 1..100),
        start in 0usize..50,
        len in 0usize..50
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        if !std_vec.is_empty() {
            let actual_start = start.min(std_vec.len() - 1);
            let actual_end = (actual_start + len).min(std_vec.len());

            let std_drained: Vec<_> = std_vec.drain(actual_start..actual_end).collect();
            let seg_drained: Vec<_> = seg_vec.drain(actual_start..actual_end).collect();

            prop_assert_eq!(std_drained, seg_drained, "drained elements mismatch");
            assert_vecs_equal(&std_vec, &seg_vec);
        }
    }

    /// Test split_off consistency.
    #[test]
    fn proptest_split_off(
        values in prop::collection::vec(any::<i32>(), 0..100),
        split_at in 0usize..100
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        let actual_split = split_at.min(std_vec.len());

        let std_split = std_vec.split_off(actual_split);
        let seg_split = seg_vec.split_off(actual_split);

        assert_vecs_equal(&std_vec, &seg_vec);
        assert_vecs_equal(&std_split, &seg_split);
    }

    /// Test resize consistency.
    #[test]
    fn proptest_resize(
        values in prop::collection::vec(any::<i32>(), 0..100),
        new_len in 0usize..200,
        fill_value in any::<i32>()
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.resize(new_len, fill_value);
        seg_vec.resize(new_len, fill_value);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test retain consistency.
    #[test]
    fn proptest_retain(
        values in prop::collection::vec(any::<i32>(), 0..100),
        threshold in any::<i32>()
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.retain(|x| *x > threshold);
        seg_vec.retain(|x| *x > threshold);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test dedup consistency.
    #[test]
    fn proptest_dedup(values in prop::collection::vec(0i32..10, 0..100)) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.dedup();
        seg_vec.dedup();

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test rotate_left consistency.
    #[test]
    fn proptest_rotate_left(
        values in prop::collection::vec(any::<i32>(), 1..100),
        mid in 0usize..100
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        let actual_mid = mid % std_vec.len().max(1);

        std_vec.rotate_left(actual_mid);
        seg_vec.rotate_left(actual_mid);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test rotate_right consistency.
    #[test]
    fn proptest_rotate_right(
        values in prop::collection::vec(any::<i32>(), 1..100),
        k in 0usize..100
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        let actual_k = k % std_vec.len().max(1);

        std_vec.rotate_right(actual_k);
        seg_vec.rotate_right(actual_k);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test fill consistency.
    #[test]
    fn proptest_fill(
        values in prop::collection::vec(any::<i32>(), 0..100),
        fill_value in any::<i32>()
    ) {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.fill(fill_value);
        seg_vec.fill(fill_value);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test extend_from_slice consistency.
    #[test]
    fn proptest_extend_from_slice(
        initial in prop::collection::vec(any::<i32>(), 0..50),
        extension in prop::collection::vec(any::<i32>(), 0..50)
    ) {
        let mut std_vec: Vec<i32> = initial.clone();
        let mut seg_vec: SegmentedVec<i32> = initial.into_iter().collect();

        std_vec.extend_from_slice(&extension);
        seg_vec.extend_from_slice(&extension);

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    /// Test append consistency.
    #[test]
    fn proptest_append(
        values1 in prop::collection::vec(any::<i32>(), 0..50),
        values2 in prop::collection::vec(any::<i32>(), 0..50)
    ) {
        let mut std_vec1: Vec<i32> = values1.clone();
        let mut std_vec2: Vec<i32> = values2.clone();
        let mut seg_vec1: SegmentedVec<i32> = values1.into_iter().collect();
        let mut seg_vec2: SegmentedVec<i32> = values2.into_iter().collect();

        std_vec1.append(&mut std_vec2);
        seg_vec1.append(&mut seg_vec2);

        assert_vecs_equal(&std_vec1, &seg_vec1);
        assert_vecs_equal(&std_vec2, &seg_vec2);
    }

    /// Test clone equality.
    #[test]
    fn proptest_clone(values in prop::collection::vec(any::<i32>(), 0..100)) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        let seg_cloned = seg_vec.clone();
        assert_vecs_equal(&std_vec, &seg_cloned);
    }

    /// Test equality comparison.
    #[test]
    fn proptest_equality(
        values1 in prop::collection::vec(any::<i32>(), 0..50),
        values2 in prop::collection::vec(any::<i32>(), 0..50)
    ) {
        let seg_vec1: SegmentedVec<i32> = values1.clone().into_iter().collect();
        let seg_vec2: SegmentedVec<i32> = values2.clone().into_iter().collect();

        let should_be_equal = values1 == values2;
        prop_assert_eq!(seg_vec1 == seg_vec2, should_be_equal);
    }

    /// Test ordering comparison.
    #[test]
    fn proptest_ordering(
        values1 in prop::collection::vec(any::<i32>(), 0..50),
        values2 in prop::collection::vec(any::<i32>(), 0..50)
    ) {
        let seg_vec1: SegmentedVec<i32> = values1.clone().into_iter().collect();
        let seg_vec2: SegmentedVec<i32> = values2.clone().into_iter().collect();

        prop_assert_eq!(seg_vec1.cmp(&seg_vec2), values1.cmp(&values2));
        prop_assert_eq!(seg_vec1.partial_cmp(&seg_vec2), values1.partial_cmp(&values2));
    }

    /// Test hash consistency.
    #[test]
    fn proptest_hash(values in prop::collection::vec(any::<i32>(), 0..100)) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        fn hash_val<T: Hash>(val: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            val.hash(&mut hasher);
            hasher.finish()
        }

        // Convert both to same representation for hashing comparison
        let std_hash = hash_val(&std_vec);
        let seg_as_vec = seg_vec.to_vec();
        let seg_hash = hash_val(&seg_as_vec);

        prop_assert_eq!(std_hash, seg_hash);
    }

    /// Test chunks iterator consistency.
    #[test]
    fn proptest_chunks(
        values in prop::collection::vec(any::<i32>(), 0..100),
        chunk_size in 1usize..20
    ) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        let std_chunks: Vec<Vec<i32>> = std_vec.chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();
        let seg_chunks: Vec<Vec<i32>> = seg_vec.chunks(chunk_size)
            .map(|c| c.iter().copied().collect())
            .collect();

        prop_assert_eq!(std_chunks, seg_chunks);
    }

    /// Test windows iterator consistency.
    #[test]
    fn proptest_windows(
        values in prop::collection::vec(any::<i32>(), 1..100),
        window_size in 1usize..20
    ) {
        let std_vec: Vec<i32> = values.clone();
        let seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        if window_size <= std_vec.len() {
            let std_windows: Vec<Vec<i32>> = std_vec.windows(window_size)
                .map(|w| w.to_vec())
                .collect();
            let seg_windows: Vec<Vec<i32>> = seg_vec.windows(window_size)
                .map(|w| w.iter().copied().collect())
                .collect();

            prop_assert_eq!(std_windows, seg_windows);
        }
    }
}

// ============================================================================
// QUICKCHECK TESTS
// ============================================================================

#[cfg(test)]
mod quickcheck_tests {
    use super::*;
    #[allow(unused_imports)]
    use quickcheck::QuickCheck;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn qc_push_pop_symmetry(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        for v in &values {
            seg_vec.push(*v);
        }

        let mut popped: Vec<i32> = Vec::new();
        while let Some(v) = seg_vec.pop() {
            popped.push(v);
        }

        popped.reverse();
        popped == values
    }

    #[quickcheck]
    fn qc_len_after_push(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        for v in &values {
            seg_vec.push(*v);
        }
        seg_vec.len() == values.len()
    }

    #[quickcheck]
    fn qc_get_after_push(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        for v in &values {
            seg_vec.push(*v);
        }

        values
            .iter()
            .enumerate()
            .all(|(i, v)| seg_vec.get(i) == Some(v))
    }

    #[quickcheck]
    fn qc_iter_matches_values(values: Vec<i32>) -> bool {
        let seg_vec: SegmentedVec<i32> = values.iter().copied().collect();
        let collected: Vec<i32> = seg_vec.iter().copied().collect();
        collected == values
    }

    #[quickcheck]
    fn qc_from_iter_round_trip(values: Vec<i32>) -> bool {
        let seg_vec: SegmentedVec<i32> = values.iter().copied().collect();
        let back: Vec<i32> = seg_vec.into_iter().collect();
        back == values
    }

    #[quickcheck]
    fn qc_clear_empties(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();
        seg_vec.clear();
        seg_vec.is_empty() && seg_vec.is_empty()
    }

    #[quickcheck]
    fn qc_truncate_limits_len(values: Vec<i32>, new_len: usize) -> bool {
        let mut std_vec: Vec<i32> = values.clone();
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();

        std_vec.truncate(new_len);
        seg_vec.truncate(new_len);

        std_vec == seg_vec.iter().copied().collect::<Vec<_>>()
    }

    #[quickcheck]
    fn qc_reverse_twice_is_identity(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = values.clone().into_iter().collect();
        seg_vec.reverse();
        seg_vec.reverse();

        values == seg_vec.iter().copied().collect::<Vec<_>>()
    }

    #[quickcheck]
    fn qc_sort_produces_sorted(values: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = values.into_iter().collect();
        seg_vec.sort();

        seg_vec.is_sorted()
    }

    #[quickcheck]
    fn qc_extend_adds_all(initial: Vec<i32>, extension: Vec<i32>) -> bool {
        let mut seg_vec: SegmentedVec<i32> = initial.clone().into_iter().collect();
        seg_vec.extend(extension.iter().copied());

        let expected_len = initial.len() + extension.len();
        seg_vec.len() == expected_len
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_operations() {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        assert_eq!(seg_vec.pop(), std_vec.pop());
        assert_eq!(seg_vec.first(), std_vec.first());
        assert_eq!(seg_vec.last(), std_vec.last());
        assert_eq!(seg_vec.is_empty(), std_vec.is_empty());
        assert_eq!(seg_vec.len(), std_vec.len());

        seg_vec.clear();
        std_vec.clear();
        assert_vecs_equal(&std_vec, &seg_vec);

        seg_vec.reverse();
        std_vec.reverse();
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_single_element() {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        seg_vec.push(42);
        std_vec.push(42);

        assert_vecs_equal(&std_vec, &seg_vec);

        assert_eq!(seg_vec.pop(), std_vec.pop());
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_boundary_segment_sizes() {
        // Test at segment boundaries (4, 8, 16, 32, 64, ...)
        let boundaries = [4, 8, 16, 32, 64, 128];

        for &boundary in &boundaries {
            let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
            let mut std_vec: Vec<i32> = Vec::new();

            // Fill to just before boundary
            for i in 0..(boundary - 1) {
                seg_vec.push(i);
                std_vec.push(i);
            }
            assert_vecs_equal(&std_vec, &seg_vec);

            // At boundary
            seg_vec.push(boundary - 1);
            std_vec.push(boundary - 1);
            assert_vecs_equal(&std_vec, &seg_vec);

            // Just after boundary
            seg_vec.push(boundary);
            std_vec.push(boundary);
            assert_vecs_equal(&std_vec, &seg_vec);
        }
    }

    #[test]
    fn test_large_vector() {
        let size = 10_000;
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for i in 0..size {
            seg_vec.push(i);
            std_vec.push(i);
        }

        assert_vecs_equal(&std_vec, &seg_vec);

        // Test random access
        for i in (0..size).step_by(100) {
            assert_eq!(std_vec.get(i as usize), seg_vec.get(i as usize));
        }

        // Test iteration
        let seg_sum: i32 = seg_vec.iter().sum();
        let std_sum: i32 = std_vec.iter().sum();
        assert_eq!(seg_sum, std_sum);
    }

    #[test]
    fn test_push_pop_interleaved() {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for i in 0..100 {
            seg_vec.push(i);
            std_vec.push(i);

            if i % 3 == 0 {
                assert_eq!(seg_vec.pop(), std_vec.pop());
            }
        }

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_insert_at_all_positions() {
        for size in [0, 1, 5, 10] {
            for insert_pos in 0..=size {
                let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                let mut std_vec: Vec<i32> = (0..size as i32).collect();

                seg_vec.insert(insert_pos, 999);
                std_vec.insert(insert_pos, 999);

                assert_vecs_equal(&std_vec, &seg_vec);
            }
        }
    }

    #[test]
    fn test_remove_at_all_positions() {
        for size in [1, 5, 10] {
            for remove_pos in 0..size {
                let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                let mut std_vec: Vec<i32> = (0..size as i32).collect();

                let seg_removed = seg_vec.remove(remove_pos);
                let std_removed = std_vec.remove(remove_pos);

                assert_eq!(seg_removed, std_removed);
                assert_vecs_equal(&std_vec, &seg_vec);
            }
        }
    }

    #[test]
    fn test_swap_remove_at_all_positions() {
        for size in [1, 5, 10] {
            for remove_pos in 0..size {
                let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                let mut std_vec: Vec<i32> = (0..size as i32).collect();

                let seg_removed = seg_vec.swap_remove(remove_pos);
                let std_removed = std_vec.swap_remove(remove_pos);

                assert_eq!(seg_removed, std_removed);
                assert_vecs_equal(&std_vec, &seg_vec);
            }
        }
    }

    #[test]
    fn test_drain_all_ranges() {
        for size in [0, 1, 5, 10] {
            for start in 0..=size {
                for end in start..=size {
                    let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                    let mut std_vec: Vec<i32> = (0..size as i32).collect();

                    let seg_drained: Vec<_> = seg_vec.drain(start..end).collect();
                    let std_drained: Vec<_> = std_vec.drain(start..end).collect();

                    assert_eq!(seg_drained, std_drained);
                    assert_vecs_equal(&std_vec, &seg_vec);
                }
            }
        }
    }

    #[test]
    fn test_split_off_all_positions() {
        for size in [0, 1, 5, 10] {
            for split_at in 0..=size {
                let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                let mut std_vec: Vec<i32> = (0..size as i32).collect();

                let seg_split = seg_vec.split_off(split_at);
                let std_split = std_vec.split_off(split_at);

                assert_vecs_equal(&std_vec, &seg_vec);
                assert_vecs_equal(&std_split, &seg_split);
            }
        }
    }

    #[test]
    fn test_truncate_all_lengths() {
        for size in [0, 1, 5, 10] {
            for trunc_len in 0..=size + 5 {
                let mut seg_vec: SegmentedVec<i32> = (0..size as i32).collect();
                let mut std_vec: Vec<i32> = (0..size as i32).collect();

                seg_vec.truncate(trunc_len);
                std_vec.truncate(trunc_len);

                assert_vecs_equal(&std_vec, &seg_vec);
            }
        }
    }

    #[test]
    fn test_resize_grow_and_shrink() {
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();

        // Grow
        seg_vec.resize(20, 99);
        std_vec.resize(20, 99);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Shrink
        seg_vec.resize(5, 99);
        std_vec.resize(5, 99);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Same size
        seg_vec.resize(5, 99);
        std_vec.resize(5, 99);
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_rotate_edge_cases() {
        // Empty vector
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();
        seg_vec.rotate_left(0);
        std_vec.rotate_left(0);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Single element
        let mut seg_vec: SegmentedVec<i32> = vec![1].into_iter().collect();
        let mut std_vec: Vec<i32> = vec![1];
        seg_vec.rotate_left(0);
        std_vec.rotate_left(0);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Full rotation
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();
        seg_vec.rotate_left(10);
        std_vec.rotate_left(10);
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_dedup_edge_cases() {
        // All same
        let mut seg_vec: SegmentedVec<i32> = vec![1, 1, 1, 1, 1].into_iter().collect();
        let mut std_vec: Vec<i32> = vec![1, 1, 1, 1, 1];
        seg_vec.dedup();
        std_vec.dedup();
        assert_vecs_equal(&std_vec, &seg_vec);

        // All different
        let mut seg_vec: SegmentedVec<i32> = vec![1, 2, 3, 4, 5].into_iter().collect();
        let mut std_vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        seg_vec.dedup();
        std_vec.dedup();
        assert_vecs_equal(&std_vec, &seg_vec);

        // Alternating
        let mut seg_vec: SegmentedVec<i32> = vec![1, 2, 1, 2, 1].into_iter().collect();
        let mut std_vec: Vec<i32> = vec![1, 2, 1, 2, 1];
        seg_vec.dedup();
        std_vec.dedup();
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_binary_search_edge_cases() {
        // Empty
        let seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let std_vec: Vec<i32> = Vec::new();
        assert_eq!(seg_vec.binary_search(&0), std_vec.binary_search(&0));

        // Single element
        let seg_vec: SegmentedVec<i32> = vec![5].into_iter().collect();
        let std_vec: Vec<i32> = vec![5];
        assert_eq!(seg_vec.binary_search(&5), std_vec.binary_search(&5));
        assert_eq!(seg_vec.binary_search(&0), std_vec.binary_search(&0));
        assert_eq!(seg_vec.binary_search(&10), std_vec.binary_search(&10));
    }

    #[test]
    fn test_retain_all_none_some() {
        // Retain all
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();
        seg_vec.retain(|_| true);
        std_vec.retain(|_| true);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Retain none
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();
        seg_vec.retain(|_| false);
        std_vec.retain(|_| false);
        assert_vecs_equal(&std_vec, &seg_vec);

        // Retain even
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();
        seg_vec.retain(|x| x % 2 == 0);
        std_vec.retain(|x| x % 2 == 0);
        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_iter_mut_modifications() {
        let mut seg_vec: SegmentedVec<i32> = (0..10).collect();
        let mut std_vec: Vec<i32> = (0..10).collect();

        for x in seg_vec.iter_mut() {
            *x *= 2;
        }
        for x in std_vec.iter_mut() {
            *x *= 2;
        }

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn test_double_ended_iterator() {
        let seg_vec: SegmentedVec<i32> = (0..10).collect();
        let std_vec: Vec<i32> = (0..10).collect();

        let seg_rev: Vec<_> = seg_vec.into_iter().rev().collect();
        let std_rev: Vec<_> = std_vec.into_iter().rev().collect();

        assert_eq!(seg_rev, std_rev);
    }
}

// ============================================================================
// DROP COUNTING TESTS
// ============================================================================

#[cfg(test)]
mod drop_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct DropCounter {
        count: Rc<RefCell<usize>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            *self.count.borrow_mut() += 1;
        }
    }

    impl Clone for DropCounter {
        fn clone(&self) -> Self {
            DropCounter {
                count: self.count.clone(),
            }
        }
    }

    #[test]
    fn test_drop_on_clear() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        assert_eq!(*count.borrow(), 0);
        seg_vec.clear();
        assert_eq!(*count.borrow(), 10);
    }

    #[test]
    fn test_drop_on_truncate() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        seg_vec.truncate(5);
        assert_eq!(*count.borrow(), 5);
    }

    #[test]
    fn test_drop_on_pop() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        for i in 0..5 {
            seg_vec.pop();
            assert_eq!(*count.borrow(), i + 1);
        }
    }

    #[test]
    fn test_drop_on_remove() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        seg_vec.remove(5);
        assert_eq!(*count.borrow(), 1);
    }

    #[test]
    fn test_drop_on_swap_remove() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        seg_vec.swap_remove(5);
        assert_eq!(*count.borrow(), 1);
    }

    #[test]
    fn test_drop_on_drain() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        // Drain but don't consume - should still drop
        drop(seg_vec.drain(3..7));
        assert_eq!(*count.borrow(), 4);
    }

    #[test]
    fn test_drop_on_vec_drop() {
        let count = Rc::new(RefCell::new(0));

        {
            let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();
            for _ in 0..10 {
                seg_vec.push(DropCounter {
                    count: count.clone(),
                });
            }
            assert_eq!(*count.borrow(), 0);
        }

        assert_eq!(*count.borrow(), 10);
    }

    #[test]
    fn test_drop_on_into_iter_partial() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        // Consume only half
        let mut iter = seg_vec.into_iter();
        for _ in 0..5 {
            iter.next();
        }
        drop(iter);

        // All 10 should be dropped (5 consumed + 5 remaining)
        assert_eq!(*count.borrow(), 10);
    }

    #[test]
    fn test_drop_on_retain() {
        let count = Rc::new(RefCell::new(0));
        let mut seg_vec: SegmentedVec<DropCounter> = SegmentedVec::new();

        for _ in 0..10 {
            seg_vec.push(DropCounter {
                count: count.clone(),
            });
        }

        // Keep every other element
        let mut i = 0;
        seg_vec.retain(|_| {
            let keep = i % 2 == 0;
            i += 1;
            keep
        });

        assert_eq!(*count.borrow(), 5);
        assert_eq!(seg_vec.len(), 5);
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn stress_many_pushes() {
        let count = 100_000;
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for i in 0..count {
            seg_vec.push(i);
            std_vec.push(i);
        }

        assert_eq!(seg_vec.len(), std_vec.len());
        assert_eq!(seg_vec.first(), std_vec.first());
        assert_eq!(seg_vec.last(), std_vec.last());

        // Spot check
        for i in (0..count).step_by(1000) {
            assert_eq!(seg_vec.get(i as usize), std_vec.get(i as usize));
        }
    }

    #[test]
    fn stress_many_push_pops() {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for i in 0..10_000 {
            seg_vec.push(i);
            std_vec.push(i);

            if i % 3 == 0 {
                assert_eq!(seg_vec.pop(), std_vec.pop());
            }
        }

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn stress_sort_large() {
        use rand::Rng;
        let mut rng = rand::rng();

        let count = 10_000;
        let values: Vec<i32> = (0..count).map(|_| rng.random()).collect();

        let mut seg_vec: SegmentedVec<i32> = values.clone().into_iter().collect();
        let mut std_vec: Vec<i32> = values;

        seg_vec.sort();
        std_vec.sort();

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn stress_drain_repeatedly() {
        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for _ in 0..100 {
            // Add elements
            for i in 0..100 {
                seg_vec.push(i);
                std_vec.push(i);
            }

            // Drain middle
            let seg_drained: Vec<_> = seg_vec.drain(25..75).collect();
            let std_drained: Vec<_> = std_vec.drain(25..75).collect();
            assert_eq!(seg_drained, std_drained);

            // Clear
            seg_vec.clear();
            std_vec.clear();
        }

        assert_vecs_equal(&std_vec, &seg_vec);
    }

    #[test]
    fn stress_random_operations() {
        use rand::Rng;
        let mut rng = rand::rng();

        let mut seg_vec: SegmentedVec<i32> = SegmentedVec::new();
        let mut std_vec: Vec<i32> = Vec::new();

        for _ in 0..10_000 {
            let op: u8 = rng.random_range(0..10);

            match op {
                0..=4 => {
                    // Push (more likely)
                    let val: i32 = rng.random();
                    seg_vec.push(val);
                    std_vec.push(val);
                }
                5 => {
                    // Pop
                    assert_eq!(seg_vec.pop(), std_vec.pop());
                }
                6 => {
                    // Insert
                    if !std_vec.is_empty() {
                        let idx = rng.random_range(0..=std_vec.len());
                        let val: i32 = rng.random();
                        seg_vec.insert(idx, val);
                        std_vec.insert(idx, val);
                    }
                }
                7 => {
                    // Remove
                    if !std_vec.is_empty() {
                        let idx = rng.random_range(0..std_vec.len());
                        assert_eq!(seg_vec.remove(idx), std_vec.remove(idx));
                    }
                }
                8 => {
                    // Swap remove
                    if !std_vec.is_empty() {
                        let idx = rng.random_range(0..std_vec.len());
                        assert_eq!(seg_vec.swap_remove(idx), std_vec.swap_remove(idx));
                    }
                }
                9 => {
                    // Truncate
                    if !std_vec.is_empty() {
                        let len = rng.random_range(0..=std_vec.len());
                        seg_vec.truncate(len);
                        std_vec.truncate(len);
                    }
                }
                _ => unreachable!(),
            }
        }

        assert_vecs_equal(&std_vec, &seg_vec);
    }
}

// ============================================================================
// ZERO-SIZED TYPE (ZST) TESTS
// ============================================================================

#[cfg(test)]
mod zst_tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    struct ZST;

    #[test]
    fn test_zst_push_pop() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..1000 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        assert_eq!(seg_vec.len(), std_vec.len());
        assert_eq!(seg_vec.len(), 1000);

        for _ in 0..500 {
            assert_eq!(seg_vec.pop(), std_vec.pop());
        }

        assert_eq!(seg_vec.len(), 500);
    }

    #[test]
    fn test_zst_get() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        for i in 0..100 {
            assert_eq!(seg_vec.get(i), Some(&ZST));
        }
        assert_eq!(seg_vec.get(100), None);
    }

    #[test]
    fn test_zst_iter() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        assert_eq!(seg_vec.iter().count(), std_vec.iter().count());
        assert!(seg_vec.iter().all(|x| *x == ZST));
    }

    #[test]
    fn test_zst_into_iter() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        let collected: Vec<ZST> = seg_vec.into_iter().collect();
        assert_eq!(collected.len(), 100);
    }

    #[test]
    fn test_zst_clear() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        seg_vec.clear();
        assert!(seg_vec.is_empty());
        assert_eq!(seg_vec.len(), 0);
    }

    #[test]
    fn test_zst_truncate() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        seg_vec.truncate(50);
        std_vec.truncate(50);

        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_insert_remove() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..10 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        seg_vec.insert(5, ZST);
        std_vec.insert(5, ZST);
        assert_eq!(seg_vec.len(), std_vec.len());

        seg_vec.remove(5);
        std_vec.remove(5);
        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_swap_remove() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..10 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        seg_vec.swap_remove(5);
        std_vec.swap_remove(5);
        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_drain() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..10 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        let seg_drained: Vec<_> = seg_vec.drain(3..7).collect();
        let std_drained: Vec<_> = std_vec.drain(3..7).collect();

        assert_eq!(seg_drained.len(), std_drained.len());
        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_extend() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        seg_vec.extend(std::iter::repeat(ZST).take(100));
        std_vec.extend(std::iter::repeat(ZST).take(100));

        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_clone() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        let cloned = seg_vec.clone();
        assert_eq!(cloned.len(), seg_vec.len());
    }

    #[test]
    fn test_zst_reverse() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        seg_vec.reverse();
        assert_eq!(seg_vec.len(), 100);
    }

    #[test]
    fn test_zst_sort() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
        }

        seg_vec.sort();
        assert_eq!(seg_vec.len(), 100);
        assert!(seg_vec.is_sorted());
    }

    #[test]
    fn test_zst_large_count() {
        // Test with a large number of ZSTs
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let count = 100_000;

        for _ in 0..count {
            seg_vec.push(ZST);
        }

        assert_eq!(seg_vec.len(), count);

        // Iterate all
        assert_eq!(seg_vec.iter().count(), count);

        // Pop all
        for _ in 0..count {
            assert_eq!(seg_vec.pop(), Some(ZST));
        }

        assert!(seg_vec.is_empty());
    }

    #[test]
    fn test_zst_unit_type() {
        // Test with () which is the canonical ZST
        let mut seg_vec: SegmentedVec<()> = SegmentedVec::new();
        let mut std_vec: Vec<()> = Vec::new();

        for _ in 0..100 {
            seg_vec.push(());
            std_vec.push(());
        }

        assert_eq!(seg_vec.len(), std_vec.len());
        assert_eq!(seg_vec.pop(), std_vec.pop());
    }

    #[test]
    fn test_zst_with_drop() {
        use std::cell::RefCell;
        use std::rc::Rc;

        #[derive(Clone)]
        struct ZstWithDrop {
            count: Rc<RefCell<usize>>,
        }

        impl Drop for ZstWithDrop {
            fn drop(&mut self) {
                *self.count.borrow_mut() += 1;
            }
        }

        let count = Rc::new(RefCell::new(0));
        {
            let mut seg_vec: SegmentedVec<ZstWithDrop> = SegmentedVec::new();

            for _ in 0..100 {
                seg_vec.push(ZstWithDrop {
                    count: count.clone(),
                });
            }

            assert_eq!(*count.borrow(), 0);
        }

        // All 100 should be dropped
        assert_eq!(*count.borrow(), 100);
    }

    #[test]
    fn test_zst_split_off() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        let seg_split = seg_vec.split_off(50);
        let std_split = std_vec.split_off(50);

        assert_eq!(seg_vec.len(), std_vec.len());
        assert_eq!(seg_split.len(), std_split.len());
    }

    #[test]
    fn test_zst_retain() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();
        let mut std_vec: Vec<ZST> = Vec::new();

        for _ in 0..100 {
            seg_vec.push(ZST);
            std_vec.push(ZST);
        }

        let mut i = 0;
        seg_vec.retain(|_| {
            i += 1;
            i % 2 == 0
        });

        let mut j = 0;
        std_vec.retain(|_| {
            j += 1;
            j % 2 == 0
        });

        assert_eq!(seg_vec.len(), std_vec.len());
    }

    #[test]
    fn test_zst_first_last() {
        let mut seg_vec: SegmentedVec<ZST> = SegmentedVec::new();

        assert_eq!(seg_vec.first(), None);
        assert_eq!(seg_vec.last(), None);

        seg_vec.push(ZST);

        assert_eq!(seg_vec.first(), Some(&ZST));
        assert_eq!(seg_vec.last(), Some(&ZST));

        for _ in 0..99 {
            seg_vec.push(ZST);
        }

        assert_eq!(seg_vec.first(), Some(&ZST));
        assert_eq!(seg_vec.last(), Some(&ZST));
    }
}
