# Bolt: Zero-Dependency Columnar Execution

## Build Independence from Arrow

**Goal:** `bolt/` compiles with only a C++20 compiler. No Arrow headers, no gRPC,
no protobuf, no vcpkg. Arrow interop is an optional integration layer that
links against Arrow when available, but the core execution engine, column
format, kernels, channels, and arena are fully self-contained.

---

## 1. What Arrow Actually Gives Us (and What We Replace)

### Things We Currently Use from Arrow

| Arrow Component | What It Does | Bolt Replacement | Status |
|----------------|-------------|------------------|--------|
| `arrow::Buffer` | Refcounted memory buffer | `Arena` bump allocation | **Done** |
| `arrow::Array` | Typed columnar array | `BoltColumn` (multi-format) | **Header done** |
| `arrow::RecordBatch` | Collection of columns + schema | `BoltBatch` (double-buffered COW) | **Header done** |
| `arrow::Schema` | Column names + types | `BoltSchema` (lightweight) | **TODO** |
| `arrow::compute::*` | Filter, Take, Sort, Hash, etc. | `bolt::kernels::*` (X-macro dispatch) | **Partial** |
| `arrow::acero::HashJoinImpl` | Swiss table hash join | `bolt::SwissJoin` (own impl) | **TODO** |
| `arrow::acero::ExecBatch` | Execution batch format | `BoltBatch` directly | **Done** |
| `arrow::Status` / `Result` | Error handling | `assert` + error codes | **Done** |
| `std::shared_ptr<Array>` | Lifetime management | Arena + epoch ownership | **Done** |
| Arrow Flight (gRPC) | Distributed RPC | FasterAPI + custom wire format | **TODO** |
| Arrow IPC | Serialization format | Custom IPC (Arrow-compatible layout) | **TODO** |
| Arrow Parquet reader | File I/O | Own Parquet reader or thin wrapper | **TODO** |
| Arrow CSV reader | File I/O | Own or vendored (trivial) | **TODO** |

### What We Keep from Arrow (Optional Link)

Arrow remains available as an **optional integration layer**. When the
`BOLT_ENABLE_ARROW_INTEROP` CMake flag is ON and Arrow is found, we compile
thin adapter code that provides `to_arrow()` / `from_arrow()` conversions.
When it's OFF, Bolt compiles standalone.

```cmake
option(BOLT_ENABLE_ARROW_INTEROP "Enable Arrow conversion layer" OFF)

if(BOLT_ENABLE_ARROW_INTEROP)
    find_package(Arrow QUIET)
    if(Arrow_FOUND)
        target_compile_definitions(bolt PUBLIC BOLT_HAS_ARROW=1)
        target_sources(bolt PRIVATE src/bolt/arrow_interop.cpp)
        target_link_libraries(bolt PUBLIC Arrow::arrow_shared)
    endif()
endif()
```

---

## 2. Component Replacement Map

### 2.1 Type System (replaces `arrow::DataType`)

Bolt uses a flat enum + size table. No inheritance, no shared_ptr, no virtual.

```cpp
enum class BoltType : uint8_t {
    // Fixed-width numeric
    Bool = 0, Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,

    // Temporal
    Date32, Date64,
    Timestamp_s, Timestamp_ms, Timestamp_us, Timestamp_ns,
    Duration_s, Duration_ms, Duration_us, Duration_ns,
    Interval_months, Interval_daytime,

    // Variable-width
    Utf8,           // German-style string view (16 bytes inline)
    Binary,         // Variable-length binary
    LargeUtf8,      // For strings > 4GB offset range
    LargeBinary,

    // Nested (Phase 3+)
    List, LargeList, Struct, Map,
    DenseUnion, SparseUnion,

    // Special
    Null,
    Dictionary,     // Keys + values encoding
    FixedSizeBinary,
    Decimal128, Decimal256,

    // Extensions
    UUID,           // 128-bit (like QuestDB)
    IPv4,           // 32-bit (like QuestDB)
    Embedding,      // Fixed-size float array for vector search

    NUM_TYPES
};

// Compile-time size table — no heap, no virtual, one array lookup
constexpr uint8_t kTypeSizes[] = {
    1,  // Bool (bit-packed validity, byte-packed values)
    1, 2, 4, 8,     // Int8..Int64
    1, 2, 4, 8,     // UInt8..UInt64
    2, 4, 8,         // Float16..Float64
    4, 8,            // Date32, Date64
    8, 8, 8, 8,     // Timestamps
    8, 8, 8, 8,     // Durations
    4, 8,            // Intervals
    16,              // Utf8 (German-style view)
    0,               // Binary (variable)
    // ... etc
};
```

### 2.2 Schema (replaces `arrow::Schema`)

```cpp
struct BoltField {
    std::string name;
    BoltType type;
    bool nullable = true;
    // For Dictionary: key type and value type
    BoltType dict_key_type = BoltType::UInt8;
    BoltType dict_value_type = BoltType::Utf8;
    // For FixedSizeBinary / Embedding: element count
    uint32_t fixed_size = 0;
};

struct BoltSchema {
    std::vector<BoltField> fields;
    int find_field(std::string_view name) const;
    // Arrow interop (only compiled with BOLT_HAS_ARROW)
    #ifdef BOLT_HAS_ARROW
    static BoltSchema from_arrow(const std::shared_ptr<arrow::Schema>&);
    std::shared_ptr<arrow::Schema> to_arrow() const;
    #endif
};
```

### 2.3 Compute Kernels (replaces `arrow::compute::*`)

Bolt kernels operate on `BoltColumn` directly. No `Datum`, no `ExecContext`,
no function registry lookup. Pure functions with X-macro type dispatch.

```
Arrow kernel call path:
  CallFunction("greater", {array, scalar})
    → FunctionRegistry::GetFunction("greater")
    → Function::DispatchExact(types)
    → KernelType::Exec(ExecContext, ExecBatch, ...)
    → Type switch inside kernel
    → Actual comparison loop

Bolt kernel call path:
  bolt::kernels::filter_gt(column, scalar, output_indices)
    → Type switch (one time, branch-predicted)
    → filter_scalar<int64_t>(data, n, value, output)
    → SIMD-vectorized comparison loop
```

**Kernel inventory — what we need to implement:**

| Category | Arrow Kernels | Bolt Equivalent | Priority |
|----------|--------------|-----------------|----------|
| Comparison | equal, not_equal, less, less_equal, greater, greater_equal | `filter_scalar<T>()` dispatch | P0 |
| Arithmetic | add, sub, mul, div, negate, abs | `arith_*<T>()` dispatch | P1 |
| Aggregation | sum, min, max, mean, count, any, all | `agg_*<T>()` with constant shortcut | P0 |
| Hash | hash_64 (xxhash) | `hash_column<T>()` (murmur3) | P0 |
| Filter | filter (bool mask → materialized) | Selection vector (deferred) | P0 — **Done** |
| Take | take (indices → materialized) | `gather<T>()` with prefetch | P0 — **Done** |
| Sort | sort_indices, array_sort_indices | `sort_permutation<T>()` (pdqsort) | P1 |
| Unique | unique, value_counts, dictionary_encode | `dict_encode<T>()` | P1 |
| String | utf8_upper, utf8_lower, starts_with, contains | Native on German-style views | P2 |
| Cast | cast (type conversion) | `cast<From, To>()` dispatch | P1 |
| If-else | if_else (conditional) | `select<T>()` with mask | P2 |
| Temporal | year, month, day, hour, minute, second | Inline arithmetic on epoch values | P2 |
| List | list_flatten, list_parent_indices | Phase 3 | P3 |
| Struct | struct_field | Phase 3 | P3 |

### 2.4 Swiss Table Hash Join (replaces `arrow::acero::HashJoinImpl`)

The Swiss table is the single most complex piece we use from Acero. Options:

**Option A: Vendor Abseil's Swiss table + write our own join logic.**
Abseil's `flat_hash_map` / `raw_hash_set` is ~3 files, MIT-licensed, no deps.
We already have `SIMDHashTable` in Chukonu with linear probing + SIMD. Extend
it with partitioned build/probe for parallel joins.

**Option B: Write a partitioned hash join from first principles.**
The actual join algorithm is well-documented (Balkesen et al., Blanas et al.).
Partitioned probe with per-partition hash tables. Our `SIMDHashTable` already
does the SIMD-accelerated probe. What's missing is the parallel build phase
(partition → build per partition → probe per partition).

**Recommendation: Option B.** The `SIMDHashTable` is already 80% there. Add
partitioned build and the join output logic. The Acero Swiss join is ~5000 lines
of code with deep Acero coupling (ExecBatch, QueryContext, TaskScheduler).
Writing 500 lines of partitioned hash join against BoltBatch is cleaner.

### 2.5 RPC / Flight Replacement (replaces Arrow Flight + gRPC)

Arrow Flight is built on gRPC + protobuf. On Windows this is a 200GB+ install
chain. FasterAPI already provides everything we need:

| Arrow Flight | FasterAPI Equivalent |
|-------------|---------------------|
| gRPC transport | `fasterapi::net::TcpListener` + `TlsSocket` (kqueue/epoll/IOCP) |
| Protobuf serialization | Bolt IPC (flat binary, no codegen) |
| Flight RPC (GetFlightInfo, DoGet, DoPut) | Custom RPC on FasterAPI (HTTP/2 or raw TCP) |
| FlightData (IPC record batches) | BoltBatch serialized as flat buffers |

**Bolt Wire Format (replaces Arrow IPC + Flight):**

```
BoltFrame (on the wire):
  [4 bytes] total_length
  [1 byte]  message_type (Schema=1, Batch=2, EOS=3, Error=4)
  [1 byte]  flags (compressed=0x01, has_stats=0x02)
  [2 bytes] num_columns
  [8 bytes] num_rows
  [schema]  only for message_type=Schema
  [column_headers] per column: type(1) + null_count(4) + data_length(4)
  [data_buffers]   contiguous column data (Arrow-compatible layout)
  [validity_buffers] packed bitmaps
  [stats_block]    if has_stats: ColumnStats × num_columns
```

This is intentionally Arrow IPC-layout-compatible for the data buffers.
A BoltFrame can be zero-copy converted to Arrow RecordBatch if needed.
But the framing, headers, and transport are ours — no protobuf, no gRPC.

### 2.6 Parquet Reading (replaces Arrow Parquet reader)

Options:

**Option A: Vendor `parquet-cpp` standalone.** The Parquet C++ library *can*
be built independently of Arrow, but it's painful and drags in thrift.

**Option B: Use DuckDB's Parquet reader.** Already vendored in `vendor/duckdb`.
DuckDB's Parquet reader is self-contained and fast. We'd read into BoltBatch
directly.

**Option C: Write a minimal Parquet reader.** Parquet's on-disk format is
documented. For our needs (read row groups, decode pages, apply predicate
pushdown), a focused implementation of ~2000 lines is feasible. This is
what QuestDB did — they wrote their own Parquet reader/writer rather than
depending on Arrow's.

**Recommendation: Start with Option A (thin wrapper around Arrow Parquet
when available), fall back to Option C for standalone builds.** Parquet
reading is the one place where the Arrow dependency is hardest to avoid
because the format is complex (nested types, page encoding, statistics).

---

## 3. Interop Matrix

What we're compatible with, and how:

| External System | Interop Method | Bolt Side | Dependency |
|----------------|---------------|-----------|------------|
| Arrow (in-process) | `to_arrow()` / `from_arrow()` | `arrow_interop.cpp` | Optional Arrow link |
| Arrow Flight (network) | Bolt Wire → Arrow IPC adapter | Receive BoltFrame, convert | Optional |
| Parquet files | Read/write via Parquet library | Arrow Parquet or own reader | Optional |
| Arrow IPC files (.arrow) | Read: parse IPC format into BoltBatch | Own IPC reader (flat binary) | None |
| DuckDB | Share via Arrow C Data Interface | PyCapsule / C Data export | None (ABI-level) |
| Polars | Arrow PyCapsule Interface | Same as DuckDB | None |
| Pandas (PyArrow) | Arrow PyCapsule Interface | Same as DuckDB | None |
| PostgreSQL (CDC) | Vendored libpq, own type mapping | Already exists in Chukonu | None |
| Redis | cyredis (vendored) | Already exists | None |
| MarbleDB | Native BoltBatch storage | Direct integration | None |
| FasterAPI | RPC transport | Bolt Wire Format over TCP | Vendored |

**Key insight:** The Arrow C Data Interface and PyCapsule Interface provide
zero-copy interop at the ABI level. We don't need to link Arrow to exchange
data with Arrow-based tools. We just need to expose our column buffers through
the C Data Interface structs (`ArrowSchema`, `ArrowArray`). This is ~100 lines
of code with zero dependencies.

---

## 4. Build Configuration

```cmake
# Bolt core — zero external deps, compiles in seconds
add_library(bolt_core STATIC
    src/bolt/arena.cpp           # Arena allocator (header-only, .cpp for explicit instantiation)
    src/bolt/column.cpp          # BoltColumn stats computation, format promotion
    src/bolt/batch.cpp           # BoltBatch COW, epoch management
    src/bolt/schema.cpp          # BoltSchema field lookup
    src/bolt/kernels/filter.cpp  # Filter kernels (X-macro generated)
    src/bolt/kernels/hash.cpp    # Hash kernels
    src/bolt/kernels/agg.cpp     # Aggregation kernels
    src/bolt/kernels/gather.cpp  # Gather / scatter
    src/bolt/kernels/sort.cpp    # Sort (pdqsort)
    src/bolt/kernels/cast.cpp    # Type casting
    src/bolt/kernels/string.cpp  # German-style string ops
    src/bolt/bitmap_index.cpp    # QuestDB-style bitmap index
    src/bolt/bloom_filter.cpp    # Bloom filter for semi-join
    src/bolt/swiss_join.cpp      # Partitioned hash join
    src/bolt/ipc.cpp             # Bolt wire format (read/write)
)
target_include_directories(bolt_core PUBLIC include/)
target_compile_features(bolt_core PUBLIC cxx_std_20)
# No external libraries. Zero. None.

# Optional Arrow interop
option(BOLT_ENABLE_ARROW_INTEROP "Arrow conversion layer" OFF)
if(BOLT_ENABLE_ARROW_INTEROP)
    find_package(Arrow QUIET)
    if(Arrow_FOUND)
        target_sources(bolt_core PRIVATE src/bolt/arrow_interop.cpp)
        target_link_libraries(bolt_core PUBLIC Arrow::arrow_shared)
        target_compile_definitions(bolt_core PUBLIC BOLT_HAS_ARROW=1)
    endif()
endif()

# Optional Arrow C Data Interface (no Arrow link needed!)
# Just implements the ABI-level structs for zero-copy exchange
option(BOLT_ENABLE_C_DATA_INTERFACE "Arrow C Data Interface export" ON)
if(BOLT_ENABLE_C_DATA_INTERFACE)
    target_sources(bolt_core PRIVATE src/bolt/c_data_interface.cpp)
    target_compile_definitions(bolt_core PUBLIC BOLT_HAS_C_DATA=1)
endif()

# Optional Parquet (needs Arrow or own reader)
option(BOLT_ENABLE_PARQUET "Parquet file read/write" OFF)

# Optional network transport (uses FasterAPI)
option(BOLT_ENABLE_TRANSPORT "Bolt Wire Protocol transport" OFF)
if(BOLT_ENABLE_TRANSPORT)
    target_sources(bolt_core PRIVATE
        src/bolt/transport/wire_format.cpp
        src/bolt/transport/bolt_server.cpp
        src/bolt/transport/bolt_client.cpp
    )
    # FasterAPI is vendored, no external dep
    target_link_libraries(bolt_core PRIVATE fasterapi)
endif()
```

Expected compile times:
- `bolt_core` (no options): **< 5 seconds** on a modern machine
- `bolt_core` + Arrow interop: **+ 2 seconds** (one translation unit)
- Full Chukonu with Arrow: same as today (Arrow is the bottleneck)
- Full Chukonu without Arrow: **massive improvement** — all the Arrow
  compile time eliminated, replaced by bolt_core's 5-second build

---

## 5. Migration Path (Chukonu → Bolt)

### Phase 1: Parallel existence (current)
- Bolt headers in `chukonu/include/chukonu/bolt/`
- Arena and Channel used directly (no Arrow dependency)
- Existing Arrow-based operators untouched

### Phase 2: Dual-path operators
- New operators written against BoltBatch
- `#ifdef BOLT_HAS_ARROW` guards on Arrow-specific code
- Morsel can hold either BoltBatch or shared_ptr<RecordBatch>

### Phase 3: Arrow at boundaries only
- All internal operators use BoltBatch
- Arrow conversion only in: Parquet reader, Flight client/server, Python bindings
- `bolt_core` is the primary library, Arrow is optional

### Phase 4: Full independence
- Own Parquet reader (or vendored minimal implementation)
- Bolt Wire Protocol replaces Arrow Flight
- Arrow C Data Interface for tool interop (Polars, DuckDB, Pandas)
- Arrow is fully optional — not needed for any core functionality

---

## 6. What We Borrow from Each Source

| Source | What We Take | How |
|--------|-------------|-----|
| **Venus ECS** | Tick-tock COW, arena-per-frame, BH-tree sidecar, X-macro dispatch, deferred ops ring buffer, job system grain sizes | Direct architectural patterns |
| **FasterAPI** | kqueue/epoll/IOCP event loop, TCP/TLS sockets, HTTP/2, connection pooling, PostgreSQL wire protocol | Vendored library for transport |
| **DuckDB** | Vector format concepts (Constant/Dict/Sequence), zone maps, push-based execution, 2048 vector size, FSST strings | Reimplemented (ideas, not code) |
| **QuestDB** | Symbol type (dict + bitmap as separate concerns), mmap column files, adaptive dictionary caching, bitmap index block design | Reimplemented |
| **Polars/arrow2** | German-style string views, Rust-inspired ownership (no refcount), ChunkedArray as Vec<Array> | Reimplemented |
| **Chronicle Queue** | Lock-free SPSC/MPSC channels, cache-line padding, pre-allocated ring buffers, sequence-based publish | Reimplemented (done) |
| **Arrow (spec only)** | Column memory layout (contiguous buffers + validity bitmap), type system, IPC wire layout for data buffers | Spec compliance, not code dependency |
| **Abseil** | Swiss table hash map (for join probe) | Potentially vendored (~3 files) |
