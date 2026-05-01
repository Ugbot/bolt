# Layer 1.4 — `bolt::doc::jsonb` design log

Sorted-keys binary JSON container, modelled on Postgres JSONB. Lands as
a header-only primitive at `include/bolt/doc/bolt_jsonb.h`. Stand-alone:
no parser dependency; values come in via the builder API.

## What we kept from Postgres JSONB

- **Sorted object keys.** Canonical order is length ascending, then
  lexicographic ascending. Lookup is `O(log K)` per object level via
  binary search rather than `O(K)` scan. Encoder enforces the order at
  finalise via insertion sort over the per-frame child run.
- **4-byte Jentry header** per child: 28 bits length-or-offset, 1 bit
  is-offset flag, 3 bits type tag (`String`, `Int64`, `Float64`,
  `BoolTrue`, `BoolFalse`, `Null`, `Object`, `Array`).
- **Stride-based offsets**: every 32nd Jentry carries an absolute
  offset into the payload region; the rest carry lengths. Random access
  at index `i` is `offset[i / 32]` plus the sum of intervening lengths
  — bounded by stride.
- **Zero-copy strings.** `String` values surface as a `(const uint8_t*,
  int32_t)` pair pointing into the encoded buffer. No allocation, no
  copy.
- **Per-row immutability.** Mutations re-encode; in-place mutation is
  explicitly out of scope (see "rejected" below).

## What we rejected from Postgres JSONB

- **Arbitrary-precision `numeric`.** We do not need 131,072-digit
  numbers. Numbers are fixed-width: `Int64` (8 bytes LE) and `Float64`
  (8 bytes LE). The 3-bit tag has slots reserved for either; rejecting
  numeric also lets us round-trip without any `decimal` library.
- **In-place mutation / TOAST.** JSONB rows in Postgres can be edited
  in place by `jsonb_set` to avoid full re-encode. This is the source
  of WAL-amplification cliffs in the Heap.io postmortem (~80% reduced
  WAL after promoting hot fields to typed columns). Our position: hot
  fields go to typed columns at table-create (Layer 3); JSONB is the
  cold/dynamic tail and is treated as immutable per row.
- **GIN pending list.** Postgres's "fast-update" pending list is the
  source of unpredictable query latency and operational pain. We will
  do batch-merge on flush with a fixed-size threshold instead (when GIN
  ships in a later layer).

## What we deviate on

- **Container header.** Postgres JSONB uses one 4-byte container
  header (count + flags). We extend to **12 bytes**: header (4) +
  `keys_bytes` (4) + `values_bytes` (4). The two extra words are the
  total payload bytes for keys and values respectively. They are needed
  to compute the length of any slot whose Jentry is an offset slot —
  in particular the offset slot at `count - kStride` — without an
  external bound. Postgres avoids this by using a different stride
  encoding (length OR offset; offset is "absolute end-of-slot rather
  than start-of-slot" in their scheme). Our simpler "offset slot is
  start-of-slot, derive length from next-stride-or-payload-end" rule
  needs the explicit payload size, which the two extra 4-byte words
  give us cheaply and unambiguously.
- **Object layout.** Objects store **all keys' Jentries first, then
  all values' Jentries**, then the payload (keys first, then values).
  Postgres stores keys-Jentry+value-Jentry interleaved per pair. Our
  split lets the binary search read only key Jentries (better cache
  behaviour for lookup) at the cost of slightly more arithmetic when
  resolving the matched value's offset.
- **Top-level scalar wrap.** A scalar root is encoded as a 1-element
  array container with the `is_scalar_wrapper` flag set. Same shape as
  Postgres; uniform format across the whole reader.

## Critical constraints honoured

- No `std::string`, `std::vector`, `std::map`, `std::deque`, smart
  pointers anywhere in the header. Builder state uses arena-resident
  buffers grown in place via `Arena::allocate`.
- Every public function is `noexcept`; ≥ 2 assertions; ≤ 70 lines.
- POD layout for `Jentry` and `ContainerHeader` with
  `static_assert(sizeof == 4)`.
- Path lookup never asserts on user input; malformed buffers return
  `false` and never read out of bounds.
- Depth capped at `kMaxDepth = 64`; total encoded payload capped at
  `kMaxBytes = 0x0FFFFFFF` (the 28-bit Jentry length bound).

## Open questions

- **Stride choice.** kStride = 32 is the Postgres default and a
  reasonable cache-line / forward-scan compromise. For 1024-key objects
  this gives 32 offset slots and 992 length slots; binary search hits
  ≤ 10 levels and each `read_key` walks at most 31 length slots inside
  a stride. Worth re-measuring once Layer 3 is wired and we have real
  document workloads — wider strides shrink the table at the cost of
  longer forward scans on offset-slot length derivation.
- **PFOR-Delta / SIMD on Jentry tables.** The 4-byte Jentry layout is
  already a hot column; a future phase could PFOR-pack lengths inside
  a stride. Defer until profiling shows the table footprint matters.
- **Bytewise key compression.** Object keys are stored verbatim. A
  later layer (when Layer 1.5 FST lands) can intern keys into a
  per-segment dictionary and store dictionary IDs in the Jentry —
  amortising key bytes across rows. The `Field::Json` shred already
  does most of this at the column level; tail-JSONB key interning is a
  follow-up.
- **Reusing Jentry slot 0 as length OR offset.** Slot 0's offset is
  always implicitly 0; we always emit a length there. A future
  micro-optimisation could tag slot 0 as "always length" implicitly —
  saving 4 bytes when count == 1 (the scalar-wrapper case). Defer.

## Files

- `external/bolt/include/bolt/doc/bolt_jsonb.h` — header-only
  encoder + reader.
- `external/bolt/tests/test_bolt_jsonb.cpp` — gtest coverage (9
  tests; round-trip scalars, flat object, sorted-key invariant,
  3-level nested object, array, missing-key returns Null, malformed
  buffer rejected, stride-offsets at 100 keys, binary search on
  {1, 2, 32, 1024}-key objects).
- `external/bolt/tests/CMakeLists.txt` — wires `test_bolt_jsonb`.

All 48 Bolt unit tests pass after the change.
