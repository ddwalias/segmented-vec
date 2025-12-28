# segmented-vec

A high-performance vector implementation with **stable element addresses**. Unlike `Vec<T>`, elements never move once inserted, making it safe to hold pointers to elements while the collection grows.

## Features

- **Stable Addresses**: Elements never relocate after insertion. Pointers to elements remain valid even after `push()`.
- **O(1) Indexing**: Constant-time index-to-segment mapping using bit manipulation (no linear search).
- **Optional Inline Storage**: Configurable `PREALLOC` const generic for stack-allocated elements before heap allocation.
- **Cache-Friendly**: Exponentially growing segments (sizes 4, 8, 16, 32...) balance memory locality with stable addressing.
- **Full `Vec`-like API**: `push`, `pop`, `get`, `iter`, `sort`, `extend`, slicing, and more.

## How It Works

`SegmentedVec` uses a clever allocation strategy where elements are stored in exponentially-sized segments:

```
Segment 0: [_, _, _, _]        (4 elements)
Segment 1: [_, _, _, _, _, _, _, _]  (8 elements)
Segment 2: [_, _, _, _, ...]   (16 elements)
...
```

Given an index, the correct segment and offset are computed in O(1) using bit operations:
- Segment = `msb(index + bias) - offset`
- Position within segment = clear the most significant bit

This avoids the pointer-chasing of linked lists while guaranteeing address stability.

## Usage

```rust
use segmented_vec::SegmentedVec;

// Create with default settings (no inline preallocation)
let mut vec: SegmentedVec<i32> = SegmentedVec::new();

// Or with inline storage for the first N elements
let mut vec: SegmentedVec<i32, 8> = SegmentedVec::new();

// Use like a normal Vec
vec.push(1);
vec.push(2);
vec.push(3);

// Get a pointer to an element
let ptr = &vec[0] as *const i32;

// Push more elements - the pointer remains valid!
for i in 4..1000 {
    vec.push(i);
}

// ptr still points to the first element
assert_eq!(unsafe { *ptr }, 1);
```

## When to Use

`SegmentedVec` is ideal when you need:

- **Stable references**: Self-referential structures, arena allocators, or caches where elements shouldn't move.
- **Incremental growth**: Building a collection where you can't predict the final size.
- **Pointer validity**: Handing out `&T` references that must remain valid across insertions.

Consider `Vec<T>` instead if:

- You need contiguous memory layout
- You frequently iterate over all elements (cache locality matters more than stability)
- You need `as_slice()` to return a single contiguous `&[T]`

## Performance

| Operation | `SegmentedVec` | `Vec` |
|-----------|----------------|-------|
| `push` | O(1) amortized | O(1) amortized |
| `pop` | O(1) | O(1) |
| `get(i)` | O(1) | O(1) |
| Address stability | Yes | No |
| Memory layout | Segmented | Contiguous |

The index calculation uses only bit operations (`bsr`, `btc`, shifts) - typically 3-5 instructions.

## PREALLOC

The `PREALLOC` const generic controls inline storage:

```rust
// No inline storage - first push allocates
let vec: SegmentedVec<i32, 0> = SegmentedVec::new();

// 8 elements stored inline before heap allocation
let vec: SegmentedVec<i32, 8> = SegmentedVec::new();
```

`PREALLOC` must be 0 or a power of 2.

## Minimum Segment Size

To avoid tiny allocations, the first dynamic segment has a minimum size based on element size:
- 1-byte elements: minimum 8 elements per segment
- Elements <= 1KB: minimum 4 elements per segment
- Larger elements: minimum 1 element per segment

This matches the optimization strategy used by `Vec` in the standard library.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
