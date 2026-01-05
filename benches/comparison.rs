//! Benchmarks comparing SegmentedVec with std::Vec using divan.
//!
//! Run with: `cargo bench`

use segmented_vec::SegmentedVec;

fn main() {
    divan::main();
}

// Trait to abstract over Vec and SegmentedVec for generic benchmarks
#[allow(dead_code)]
trait VecLike<T>: Default {
    fn with_capacity(cap: usize) -> Self;
    fn push(&mut self, val: T);
    fn pop(&mut self) -> Option<T>;
    fn get(&self, idx: usize) -> Option<&T>;
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;
    fn sort(&mut self)
    where
        T: Ord;
    fn binary_search(&self, val: &T) -> Result<usize, usize>
    where
        T: Ord;
    fn contains(&self, val: &T) -> bool
    where
        T: PartialEq;
    fn reverse(&mut self);
    fn insert(&mut self, idx: usize, val: T);
    fn remove(&mut self, idx: usize) -> T;
    fn swap_remove(&mut self, idx: usize) -> T;
}

impl<T> VecLike<T> for Vec<T> {
    fn with_capacity(cap: usize) -> Self {
        Vec::with_capacity(cap)
    }
    fn push(&mut self, val: T) {
        self.push(val);
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn get(&self, idx: usize) -> Option<&T> {
        <[T]>::get(self, idx)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        <[T]>::iter(self)
    }
    fn sort(&mut self)
    where
        T: Ord,
    {
        <[T]>::sort(self);
    }
    fn binary_search(&self, val: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        <[T]>::binary_search(self, val)
    }
    fn contains(&self, val: &T) -> bool
    where
        T: PartialEq,
    {
        <[T]>::contains(self, val)
    }
    fn reverse(&mut self) {
        <[T]>::reverse(self);
    }
    fn insert(&mut self, idx: usize, val: T) {
        self.insert(idx, val);
    }
    fn remove(&mut self, idx: usize) -> T {
        self.remove(idx)
    }
    fn swap_remove(&mut self, idx: usize) -> T {
        self.swap_remove(idx)
    }
}

impl<T> VecLike<T> for SegmentedVec<T> {
    fn with_capacity(cap: usize) -> Self {
        SegmentedVec::with_capacity(cap)
    }
    fn push(&mut self, val: T) {
        self.push(val);
    }
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn get(&self, idx: usize) -> Option<&T> {
        self.get(idx)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        SegmentedVec::iter(self)
    }
    fn sort(&mut self)
    where
        T: Ord,
    {
        self.sort();
    }
    fn binary_search(&self, val: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.binary_search(val)
    }
    fn contains(&self, val: &T) -> bool
    where
        T: PartialEq,
    {
        self.contains(val)
    }
    fn reverse(&mut self) {
        self.reverse();
    }
    fn insert(&mut self, idx: usize, val: T) {
        self.insert(idx, val);
    }
    fn remove(&mut self, idx: usize) -> T {
        self.remove(idx)
    }
    fn swap_remove(&mut self, idx: usize) -> T {
        self.swap_remove(idx)
    }
}

// ============================================================================
// Push Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn push<V: VecLike<i32>, const N: usize>() -> V {
    let mut v = V::default();
    for i in 0..N as i32 {
        v.push(i);
    }
    v
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn push_with_capacity<V: VecLike<i32>, const N: usize>() -> V {
    let mut v = V::with_capacity(N);
    for i in 0..N as i32 {
        v.push(i);
    }
    v
}

// ============================================================================
// Pop Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn pop<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_values(|mut v| {
            while v.pop().is_some() {}
            v
        });
}

// ============================================================================
// Access Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn sequential_read<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| {
            let mut sum = 0i32;
            for i in 0..N {
                sum = sum.wrapping_add(*v.get(i).unwrap());
            }
            sum
        });
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn random_read<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    let indices: Vec<usize> = (0..N).map(|_| rng.random_range(0..N)).collect();

    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| {
            let mut sum = 0i32;
            for &i in &indices {
                sum = sum.wrapping_add(*v.get(i).unwrap());
            }
            sum
        });
}

// ============================================================================
// Iteration Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn iterate<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| {
            let mut sum = 0i32;
            for &x in v.iter() {
                sum = sum.wrapping_add(x);
            }
            sum
        });
}

// ============================================================================
// Sort Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn sort<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    let data: Vec<i32> = (0..N).map(|_| rng.random()).collect();

    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for &x in &data {
                v.push(x);
            }
            v
        })
        .bench_local_values(|mut v| {
            v.sort();
            v
        });
}

// ============================================================================
// Search Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn binary_search<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    let targets: Vec<i32> = (0..100).map(|_| rng.random_range(0..N as i32)).collect();

    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| {
            let mut found = 0usize;
            for &t in &targets {
                if v.binary_search(&t).is_ok() {
                    found += 1;
                }
            }
            found
        });
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn contains<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    let targets: Vec<i32> = (0..100)
        .map(|_| rng.random_range(0..N as i32 * 2))
        .collect();

    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| {
            let mut found = 0usize;
            for &t in &targets {
                if v.contains(&t) {
                    found += 1;
                }
            }
            found
        });
}

// ============================================================================
// Mutation Benchmarks
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn reverse<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_values(|mut v| {
            v.reverse();
            v
        });
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000])]
fn insert_front<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher.bench_local(|| {
        let mut v = V::default();
        for i in 0..N as i32 {
            v.insert(0, i);
        }
        v
    });
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000])]
fn remove_front<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_values(|mut v| {
            while v.len() > 0 {
                v.remove(0);
            }
            v
        });
}

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn swap_remove<V: VecLike<i32>, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_values(|mut v| {
            while v.len() > 0 {
                v.swap_remove(0);
            }
            v
        });
}

// ============================================================================
// Clone Benchmark
// ============================================================================

#[divan::bench(types = [Vec<i32>, SegmentedVec<i32>], consts = [100, 1000, 10000])]
fn clone<V: VecLike<i32> + Clone, const N: usize>(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut v = V::default();
            for i in 0..N as i32 {
                v.push(i);
            }
            v
        })
        .bench_local_refs(|v| v.clone());
}

// ============================================================================
// Pointer Stability (SegmentedVec's key advantage)
// ============================================================================

#[divan::bench(consts = [100, 1000, 10000])]
fn pointer_stability_segmented_vec<const N: usize>() {
    let mut v: SegmentedVec<i32> = SegmentedVec::new();
    let mut ptrs = Vec::with_capacity(N);

    for i in 0..N as i32 {
        v.push(i);
        ptrs.push(v.get(v.len() - 1).unwrap() as *const i32);
    }

    // Verify all pointers are still valid
    for (i, &ptr) in ptrs.iter().enumerate() {
        assert_eq!(unsafe { *ptr }, i as i32);
    }
}

#[divan::bench(consts = [100, 1000, 10000])]
fn pointer_stability_vec_requires_realloc<const N: usize>() {
    // Vec cannot guarantee pointer stability - this shows the workaround cost
    // where you'd need to Box each element or use indices
    let mut v: Vec<Box<i32>> = Vec::new();
    let mut ptrs = Vec::with_capacity(N);

    for i in 0..N as i32 {
        v.push(Box::new(i));
        ptrs.push(v.last().unwrap().as_ref() as *const i32);
    }

    // Verify all pointers are still valid
    for (i, &ptr) in ptrs.iter().enumerate() {
        assert_eq!(unsafe { *ptr }, i as i32);
    }
}
