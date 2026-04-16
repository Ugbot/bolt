# Hash function choice for Bolt's open-addressing tables

Picking a hash function for `SwissTable` (and downstream `GroupByTable`,
`HashJoinBuild`/`Probe`) is a per-cycle trade-off between mixing quality
and inner-loop cost. We considered four candidates while looking at
1BRC perf and stayed with the existing Murmur3 finalizer; this note
records *why*, so future tuning passes don't relitigate from scratch.

## The candidates

### 1. `swiss_mix` — Murmur3 finalizer (status quo)

```cpp
BOLT_FORCE_INLINE uint64_t swiss_mix(uint64_t k) noexcept {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDULL;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ULL;
    k ^= k >> 33;
    return k;
}
```

- **Cost**: 6 ops (3 multiplies, 3 xor-shifts). ~5-7 cycle dependency
  chain on modern x86. ~3-4 cycles on AArch64 with multiply forwarding.
- **Quality**: passes SMHasher avalanche tests. Every output bit depends
  on every input bit. The low bits we use for the table index
  `idx = h & (cap-1)` and the high bits we use for the SwissTable tag
  `tag = (h >> 57) & 0x7F` are statistically independent.
- **Robustness**: works on any input distribution — random keys, dense
  integers, sequential timestamps, sparse hashes — all produce uniform
  output.

### 2. Fibonacci hash (Knuth multiplicative)

```cpp
BOLT_FORCE_INLINE uint64_t fib_hash(uint64_t x) noexcept {
    return x * 0x9E3779B97F4A7C15ULL;  // 2^64 / golden ratio
}
```

- **Cost**: 1 multiply. ~3 cycle latency. Roughly half of `swiss_mix`.
- **Quality**: spreads input bits into the **high** bits of the output;
  the **low** bits stay highly correlated with the low bits of the
  input. Dense input keys 0..N-1 produce dense low bits in the output.
- **Use it correctly**: indexing must use **high bits**, e.g.
  `idx = (h * cap) >> 64` (Lemire's fastrange) — *not* `idx = h & mask`.
- **Failure mode**: paired with low-bit masking and dense integer
  inputs, every key collides into the same probe chain. 10-100× probe
  inflation on workloads we don't control.

### 3. wyhash-style 3-op mix

```cpp
BOLT_FORCE_INLINE uint64_t wyhash_mix(uint64_t k) noexcept {
    k ^= k >> 32;
    k *= 0xE7037ED1A0B428DBULL;  // any well-chosen odd 64-bit constant
    k ^= k >> 32;
    return k;
}
```

- **Cost**: 3 ops, ~3-4 cycles. Roughly half `swiss_mix`'s dep chain.
- **Quality**: not as strong as full Murmur3 (one less round) but still
  passes basic avalanche; high and low bits are independent enough for
  open-addressing.
- **Robustness**: safe on any input distribution. Lower margin than
  `swiss_mix` but no known catastrophic failure mode for our use.

### 4. Perfect hashing (PtrHash, BBHash, …)

- **Cost**: 1 multiply + shift at probe time, zero collisions, no
  empty-slot checks.
- **Quality**: optimal — the function is built per key set so
  collisions are mathematically impossible.
- **Constraint**: requires the key set to be known at build time (or
  inside a tight time budget per refresh). Not applicable to general
  hash join / streaming groupby. Researched; deferred to a dedicated
  module per `docs/research/1brc.md`.

## SwissTable's design coupling

Bolt's SwissTable splits the hash output into two roles:

```cpp
uint8_t  tag = swiss_tag(h);   // (h >> 57) & 0x7F   — high 7 bits
uint32_t idx = h & mask;       // (cap - 1)           — low log2(cap) bits
```

The tag is the per-slot fast-reject filter (16-byte SIMD scan via
`bmm_cmpeq_i8`); the index is the probe-chain head. **Critical
property: tag and idx must be statistically independent**, otherwise
matching tags within a group all share the same idx and the SIMD scan
loses its selectivity. `swiss_mix`'s avalanche guarantees this.

Fibonacci hash with high-bit indexing breaks the property: both tag
and idx now derive from the same high bits, so a tag match implies an
idx match — the tag stops being useful as a filter. Inner loop
regresses.

## Why we kept `swiss_mix`

The 5-10% saved on the inner loop by switching to fib hash:

- evaporates if any caller passes dense-integer keys (loop becomes
  10-100× slower);
- forces a SwissTable index-scheme change (`h & mask` → `(h * cap) >> 64`)
  that propagates to every caller;
- couples tag and idx, costing the SIMD-scan selectivity that's the
  whole point of SwissTable.

Bolt is a library. We don't control caller key shapes. Trading
correctness margin for a 5-10% inner-loop saving is a poor deal.

## Two viable paths if we want the win later

These are NOT done; recording them so the next perf wave can pick up
without re-deriving:

1. **Switch SwissTable internal mix to wyhash-style 3-op (#3 above).**
   Cuts the mix cost roughly in half, keeps full input-distribution
   robustness, no API change, no tag/idx coupling. Single edit to
   `swiss_mix`. Safe to ship.

2. **Add a `pre_hashed=true` SwissTable contract.** Callers that already
   produce uniform keys (e.g. our 1BRC bench passes `swiss_mix(station_id)`
   as the key — the table re-mixes wastedly) opt in to skip the
   internal mix entirely. Zero-op fast path. Documents an explicit
   caller responsibility — misuse is a probe-chain explosion, but the
   contract makes it the caller's bug, not the library's. Implement as
   a template parameter or a second `SwissTable_PreHashed` type to keep
   the default path safe.

## Out of scope (deferred)

- **PtrHash / minimal perfect hashing** for fixed-cardinality
  dictionaries. See [`1brc.md`](1brc.md) followups list. Belongs next
  to dictionary-encoding work.
- **Per-type specialised hashes** (e.g. Identity for `int32_t` keys
  small enough to fit the table directly). Tempting; same fragility
  as plain fib hash. Won't ship without a `pre_hashed` opt-in.

## References

- Knuth, *TAOCP* Vol. 3 §6.4 — multiplicative hashing.
- Lemire, "Fast random integer generation in an interval"
  https://arxiv.org/abs/1805.10941 — high-bit indexing.
- Wang Yi, "wyhash" https://github.com/wangyi-fudan/wyhash —
  3-op mix in current production use (Rust hashbrown via foldhash, etc.).
- Groot Koerkamp, "PtrHash"
  https://curiouscoding.nl/posts/ptrhash/ — perfect hashing for
  fixed key sets.
- Abseil SwissTable design doc
  https://abseil.io/about/design/swisstables — original tag/idx split.
