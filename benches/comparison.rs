//! Benchmarks comparing SegmentedVec/SegmentedSlice with std Vec/slice.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use segmented_vec::SegmentedVec;
use std::hint::black_box;

// Test sizes for benchmarks
#[cfg(feature = "ci")]
const SIZES: &[usize] = &[100, 1_000];

#[cfg(not(feature = "ci"))]
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];

// ============================================================================
// Push benchmarks
// ============================================================================

fn bench_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("push");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter(|| {
                let mut v: Vec<i32> = Vec::new();
                for i in 0..size {
                    v.push(i as i32);
                }
                black_box(v)
            });
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter(|| {
                let mut v: SegmentedVec<i32> = SegmentedVec::new();
                for i in 0..size {
                    v.push(i as i32);
                }
                black_box(v)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Push with pre-allocated capacity
// ============================================================================

fn bench_push_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_with_capacity");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter(|| {
                let mut v: Vec<i32> = Vec::with_capacity(size);
                for i in 0..size {
                    v.push(i as i32);
                }
                black_box(v)
            });
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter(|| {
                let mut v: SegmentedVec<i32> = SegmentedVec::with_capacity(size);
                for i in 0..size {
                    v.push(i as i32);
                }
                black_box(v)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Pop benchmarks
// ============================================================================

fn bench_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<Vec<_>>(),
                |mut v| {
                    while let Some(val) = v.pop() {
                        black_box(val);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<SegmentedVec<_>>(),
                |mut v| {
                    while let Some(val) = v.pop() {
                        black_box(val);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Sequential access (indexing)
// ============================================================================

fn bench_sequential_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_access");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &vec_data, |b, v| {
            b.iter(|| {
                let mut sum = 0i64;
                for num in v {
                    sum += *num as i64;
                }
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &seg_data, |b, v| {
            b.iter(|| {
                let mut sum = 0i64;
                for i in 0..v.len() {
                    sum += v[i] as i64;
                }
                black_box(sum)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Random access
// ============================================================================

fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        // Pre-generate random indices
        let mut rng = StdRng::seed_from_u64(42);
        let indices: Vec<usize> = (0..size).map(|_| rng.random_range(0..size)).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Vec", size),
            &(&vec_data, &indices),
            |b, (v, idx)| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for &i in idx.iter() {
                        sum += v[i] as i64;
                    }
                    black_box(sum)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SegmentedVec", size),
            &(&seg_data, &indices),
            |b, (v, idx)| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for &i in idx.iter() {
                        sum += v[i] as i64;
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Iterator benchmarks
// ============================================================================

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iter");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &vec_data, |b, v| {
            b.iter(|| {
                let sum: i64 = v.iter().map(|&x| x as i64).sum();
                black_box(sum)
            });
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &seg_data, |b, v| {
            b.iter(|| {
                let sum: i64 = v.iter().map(|&x| x as i64).sum();
                black_box(sum)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Sort benchmarks
// ============================================================================

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort");

    for &size in SIZES {
        // Generate random data
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<i32> = (0..size).map(|_| rng.random()).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut v| {
                    v.sort();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &data, |b, data| {
            b.iter_batched(
                || data.iter().copied().collect::<SegmentedVec<_>>(),
                |mut v| {
                    v.sort();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Unstable sort benchmarks
// ============================================================================

fn bench_sort_unstable(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_unstable");

    for &size in SIZES {
        // Generate random data
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<i32> = (0..size).map(|_| rng.random()).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &data, |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut v| {
                    v.sort_unstable();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &data, |b, data| {
            b.iter_batched(
                || data.iter().copied().collect::<SegmentedVec<_>>(),
                |mut v| {
                    v.sort_unstable();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Binary search benchmarks
// ============================================================================

fn bench_binary_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_search");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        // Pre-generate search targets
        let mut rng = StdRng::seed_from_u64(42);
        let targets: Vec<i32> = (0..1000)
            .map(|_| rng.random_range(0..size as i32))
            .collect();

        group.throughput(Throughput::Elements(1000));

        group.bench_with_input(
            BenchmarkId::new("slice", size),
            &(&vec_data, &targets),
            |b, (v, targets)| {
                b.iter(|| {
                    let mut found = 0;
                    for &t in targets.iter() {
                        if v.binary_search(&t).is_ok() {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SegmentedSlice", size),
            &(&seg_data, &targets),
            |b, (v, targets)| {
                b.iter(|| {
                    let mut found = 0;
                    for &t in targets.iter() {
                        if v.binary_search(&t).is_ok() {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Contains benchmarks
// ============================================================================

fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");

    // Smaller sizes for linear search
    let sizes = &[100, 1_000, 10_000];

    for &size in sizes {
        let vec_data: Vec<i32> = (0..size).collect();
        let seg_data: SegmentedVec<i32> = (0..size).collect();

        // Search for element in the middle
        let target = size / 2;

        group.bench_with_input(
            BenchmarkId::new("slice", size),
            &(&vec_data, target),
            |b, (v, t)| {
                b.iter(|| black_box(v.contains(t)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SegmentedSlice", size),
            &(&seg_data, target),
            |b, (v, t)| {
                b.iter(|| black_box(v.contains(t)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Chunk iteration benchmarks
// ============================================================================

fn bench_chunks(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunks");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("slice", size), &vec_data, |b, v| {
            b.iter(|| {
                let mut sum = 0i64;
                for chunk in v.chunks(64) {
                    for &x in chunk {
                        sum += x as i64;
                    }
                }
                black_box(sum)
            });
        });

        group.bench_with_input(
            BenchmarkId::new("SegmentedSlice", size),
            &seg_data,
            |b, v| {
                b.iter(|| {
                    let mut sum = 0i64;
                    for chunk in v.chunks(64) {
                        for &x in chunk.iter() {
                            sum += x as i64;
                        }
                    }
                    black_box(sum)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Insert benchmarks (insert at beginning - worst case)
// ============================================================================

fn bench_insert_front(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_front");

    // Smaller sizes since insert is O(n)
    let sizes = &[100, 1_000, 10_000];

    for &size in sizes {
        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || Vec::with_capacity(size + 100),
                |mut v| {
                    for i in 0..100 {
                        v.insert(0, i);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || SegmentedVec::<i32>::with_capacity(size + 100),
                |mut v| {
                    for i in 0..100 {
                        v.insert(0, i);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Remove benchmarks
// ============================================================================

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    // Smaller sizes since remove is O(n)
    let sizes = &[100, 1_000, 10_000];

    for &size in sizes {
        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size).collect::<Vec<_>>(),
                |mut v| {
                    // Remove from the middle
                    for _ in 0..100.min(size) {
                        if !v.is_empty() {
                            v.remove(v.len() / 2);
                        }
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size).collect::<SegmentedVec<_>>(),
                |mut v| {
                    // Remove from the middle
                    for _ in 0..100.min(size) {
                        if !v.is_empty() {
                            v.remove(v.len() / 2);
                        }
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Swap remove benchmarks (O(1) removal)
// ============================================================================

fn bench_swap_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("swap_remove");

    for &size in SIZES {
        let remove_count = 100.min(size);

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let v: Vec<i32> = (0..size as i32).collect();
                    let mut rng = StdRng::seed_from_u64(42);
                    let indices: Vec<usize> = (0..remove_count)
                        .map(|i| rng.random_range(0..(size - i)))
                        .collect();
                    (v, indices)
                },
                |(mut v, indices)| {
                    for idx in indices {
                        v.swap_remove(idx);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let v: SegmentedVec<i32> = (0..size as i32).collect();
                    let mut rng = StdRng::seed_from_u64(42);
                    let indices: Vec<usize> = (0..remove_count)
                        .map(|i| rng.random_range(0..(size - i)))
                        .collect();
                    (v, indices)
                },
                |(mut v, indices)| {
                    for idx in indices {
                        v.swap_remove(idx);
                    }
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Pointer stability test (unique to SegmentedVec)
// ============================================================================

fn bench_pointer_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("pointer_stability");

    // This demonstrates the advantage of SegmentedVec:
    // pushing doesn't invalidate existing pointers

    let size = 10_000usize;

    group.bench_function("Vec_push_many", |b| {
        b.iter(|| {
            let mut v: Vec<i32> = (0..100).collect();
            // Vec may reallocate during these pushes
            for i in 100..size as i32 {
                v.push(i);
            }
            black_box(v.len())
        });
    });

    group.bench_function("SegmentedVec_push_many_with_stable_ptr", |b| {
        b.iter(|| {
            let mut v: SegmentedVec<i32> = (0..100).collect();
            // Get a pointer - this remains valid after push!
            let ptr = &v[0] as *const i32;
            for i in 100..size as i32 {
                v.push(i);
            }
            // Pointer is still valid - this is the key advantage!
            let val = unsafe { *ptr };
            black_box((v.len(), val))
        });
    });

    group.finish();
}

// ============================================================================
// Clone benchmarks
// ============================================================================

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    for &size in SIZES {
        let vec_data: Vec<i32> = (0..size as i32).collect();
        let seg_data: SegmentedVec<i32> = (0..size as i32).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &vec_data, |b, v| {
            b.iter(|| black_box(v.clone()));
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &seg_data, |b, v| {
            b.iter(|| black_box(v.clone()));
        });
    }

    group.finish();
}

// ============================================================================
// Reverse benchmarks
// ============================================================================

fn bench_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<Vec<_>>(),
                |mut v| {
                    v.reverse();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<SegmentedVec<_>>(),
                |mut v| {
                    v.reverse();
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================================
// Rotate benchmarks
// ============================================================================

fn bench_rotate(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("Vec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<Vec<_>>(),
                |mut v| {
                    v.rotate_left(size / 3);
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("SegmentedVec", size), &size, |b, &size| {
            b.iter_batched(
                || (0..size as i32).collect::<SegmentedVec<_>>(),
                |mut v| {
                    v.rotate_left(size / 3);
                    black_box(v)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_push,
    bench_push_with_capacity,
    bench_pop,
    bench_sequential_access,
    bench_random_access,
    bench_iter,
    bench_sort,
    bench_sort_unstable,
    bench_binary_search,
    bench_contains,
    bench_chunks,
    bench_insert_front,
    bench_remove,
    bench_swap_remove,
    bench_pointer_stability,
    bench_clone,
    bench_reverse,
    bench_rotate,
);

criterion_main!(benches);
