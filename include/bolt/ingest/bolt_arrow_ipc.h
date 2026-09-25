// bolt_arrow_ipc.h — Arrow IPC STREAM writer (G2ARROW-10, widened G2ARROW-20,
// nested List/Struct added G2ARROW-21).
//
// Writes the Arrow IPC streaming format — an encapsulated Schema message
// followed by RecordBatch messages and the end-of-stream marker — such
// that `pyarrow.ipc.open_stream()` reads the bytes and every value
// matches. The flatbuffer message headers are encoded FROM THE SPEC (
// https://flatbuffers.dev/internals/ + arrow/format/{Message,Schema}.fbs,
// fetched live and re-verified for this pass) by a minimal bottom-up
// builder in bolt_arrow_ipc.cpp (the same from-spec discipline as the
// thrift-compact parquet page-index decoder): no flatbuffers library
// dependency, no code generation.
//
// TYPE MAPPING — deliberately the SAME per-value resolution the Arrow C
// Data export (bolt/bolt_arrow.h) uses: fixed-width columns hand their
// Flat payload through verbatim (per-type width via bolt::type_size —
// Int64/Float64/Date32/Decimal128 differ only in byte width), Utf8/Binary
// both resolve every row through bolt::arrow::detail::var_at (inline
// StringView / spilled str_overflow_base / VarBinary offsets all covered
// — var_at does not care whether the logical type is Utf8 or Binary, only
// about the physical layout), Bool is bit-packed from bolt's byte-packed
// storage the same way bolt_arrow.h's pack_bool does, Decimal128's 16-byte
// payload is ALREADY the little-endian two's-complement layout Arrow
// wants (bolt stores it that way natively), and the validity bitmap is
// bolt's Arrow-shape LSB-first bitmap. One mapping, two transports — or
// they drift. (List/Struct have no C-Data-export counterpart to share
// with yet — bolt_arrow.h does not export nested columns today — so their
// mapping below is new to this writer, derived directly from
// bolt_column.h's own `ColumnFormat::Nested` contract, which already
// documents the Arrow-shaped layout it was designed to mirror.)
//
// NESTED TYPES (G2ARROW-21). A List/Struct column's child field type(s)
// must be known when the Schema message is written — at open(), before
// any batch or child column has been seen — so a flat `BoltType[]` per
// top-level column (what G2ARROW-10/20 used) cannot describe them. Two
// entry points:
//   - `arrow_ipc_open()` — unchanged flat signature; every column is a
//     leaf (Int64/Float64/Utf8/Bool/Date32/Binary/Decimal128). Thin
//     wrapper over arrow_ipc_open_nested() below.
//   - `arrow_ipc_open_nested()` — each top-level column is a recursive
//     `FieldSpec`: a leaf as above, or `type == List` (exactly one child
//     FieldSpec — the element type) or `type == Struct` (one or more
//     child FieldSpecs — the fields, in schema order). Nesting is bounded
//     (kIpcMaxNestDepth) and the total flattened field count (top-level +
//     every descendant) is bounded (kIpcMaxFields); either bound crossed
//     fails closed at open(), nothing written.
//   Runtime batches carry List/Struct columns via bolt_column.h's
//   `ColumnFormat::Nested` (`BoltColumn::make_list`/`make_struct`,
//   `child_at()`/`list_offsets()`/`list_element()`) — arrow_ipc_write_batch
//   walks that tree directly; see bolt_column.h's own contract comment for
//   the null-vs-empty-list distinction (an empty list is a VALID row with
//   `offsets[i] == offsets[i+1]`; a null list differs only in the
//   validity bit, not the offset span).
//   Wire layout per field, depth-first pre-order (Message.fbs:
//   "Nodes/Buffers correspond to the pre-ordered flattened logical
//   schema/buffer tree" — a struct field's own children follow it
//   immediately, then the next sibling): List contributes
//   [validity, int32 offsets] then its one child's own node+buffers;
//   Struct contributes [validity] (no data buffer of its own) then each
//   field's own node+buffers in order.
//   Map is NOT attempted here (ticket scope is List/Struct); a Map
//   FieldSpec fails closed at open() the same as any other unsupported
//   type.
//
// HONEST SCOPE (fail closed at open, never a misencoded stream):
//   Leaf column types: Int64, Float64, Utf8, Bool, Date32, Timestamp
//   (timestamp[us], no zone), Binary, Decimal128. Nested: List (1 child), Struct (N children), to any depth
//   up to kIpcMaxNestDepth.
//   Everything else — Decimal64 (bolt_arrow.h's own C-Data export has no
//   Arrow format string for it either; extending only this writer would
//   invent a mapping the C-Data export doesn't share, breaking the "one
//   mapping, two transports" rule above) and Map — is rejected by
//   arrow_ipc_open()/arrow_ipc_open_nested() with `false`.
//   Decimal128 additionally requires a scale at open() (via
//   `decimal_scales[c]` on the flat entry point, or `FieldSpec::
//   decimal_scale` at any depth on the nested one) — the schema's Decimal
//   type table carries (precision, scale) and both must be known before
//   any batch is written, so there is no later point to source it from;
//   open() fails closed on a Decimal128 field with no scale (flat entry
//   point only — the nested one always carries a scale field, defaulted
//   0) or scale > 38. Precision is always 38 (the Decimal128 maximum —
//   bolt does not track a narrower precision per column, matching
//   bolt_arrow.h's C-Data export).
//   Column formats: Flat/View for fixed-width; Flat/View/VarBinary for
//   Utf8/Binary (exactly what var_at can resolve); Nested for List/Struct
//   (exactly what bolt_column.h's ColumnFormat::Nested produces).
//   Dictionary/RLE/Constant must be materialized by the caller first.
//
// Tiger Style: caller-owned writer struct (heap/arena it — a stack
// BoltBatch-sized local is this repo's documented stack-overflow trap),
// bounded columns/fields/depth, no allocation, no exceptions; errors are
// `false` returns and the writer latches failed state.

#pragma once

#include <cstdint>
#include <cstdio>

#include "bolt/bolt_types.h"

namespace bolt { struct BoltBatch; struct BoltColumn; }

namespace bolt::ingest {

// Bounded surface. 64 columns matches the Arrow C-Data export cap.
inline constexpr std::uint16_t kIpcMaxCols   = 64;
inline constexpr std::uint32_t kIpcNameCap   = 64;   // per-field name bytes
inline constexpr std::uint32_t kIpcFbCap     = 1u << 15;  // flatbuffer scratch

// Total flattened field budget: kIpcMaxCols top-level columns plus every
// List/Struct descendant (a List's one element field; a Struct's field
// list, and THEIR descendants, recursively). Bounded so the writer's
// per-field bookkeeping (ArrowIpcWriter::desc, BatchLayout::nodes/buffers)
// stays a fixed-size array — no allocation, no unbounded recursion.
inline constexpr std::uint16_t kIpcMaxFields    = 256;
// Maximum List/Struct nesting depth (a bare leaf column is depth 0).
inline constexpr std::uint16_t kIpcMaxNestDepth = 8;

/// Recursive column/field descriptor for arrow_ipc_open_nested(). Every
/// FieldSpec describes exactly one Arrow field: a top-level column, a
/// List's element field, or one Struct field.
///
/// `children`/`n_children` matter only when `type == List` (exactly 1
/// child: the element type) or `type == Struct` (1..kIpcMaxCols children:
/// the fields, in schema order) — every other `type` must leave both
/// nullptr/0. arrow_ipc_open_nested() rejects the mismatched shape at
/// open() (fail closed) rather than silently ignoring extra/missing
/// children.
struct FieldSpec {
    BoltType          type;
    const char*       name;           // nullptr/"" -> a name is synthesized
                                       // ("cN" top-level, "item" a List's
                                       // element, "fK" a Struct's Kth field)
    std::uint8_t      decimal_scale;  // Decimal128 (any depth) only; ignored
                                       // otherwise, so a zeroed FieldSpec is
                                       // a valid non-decimal leaf
    std::uint16_t     n_children;     // 0 (leaf), 1 (List), or N (Struct)
    const FieldSpec*  children;       // borrowed; must outlive the open() call
};

/// One flattened field, as arrow_ipc_open()/arrow_ipc_open_nested() build
/// it into ArrowIpcWriter::desc[] — an internal representation; do not
/// construct this directly. desc[0..n_cols-1] are exactly the top-level
/// columns, in order. Any List/Struct descendant is appended afterward,
/// each node's own direct children as a CONTIGUOUS block reachable via
/// `first_child` (breadth-first append order — NOT the depth-first
/// pre-order the wire format itself uses for a given batch's nodes/
/// buffers; that numbering is produced separately, per batch, by
/// arrow_ipc_write_batch, since only breadth-first append keeps a node's
/// own children contiguous once a sibling's subtree can be an arbitrary
/// size).
struct IpcFieldDesc {
    std::uint16_t type;            // BoltType
    std::uint16_t first_child;     // index into ArrowIpcWriter::desc;
                                    // meaningful iff n_children > 0
    std::uint8_t  n_children;      // 0 (leaf), 1 (List), or N (Struct)
    std::uint8_t  decimal_scale;   // Decimal128 leaves only
};

struct ArrowIpcWriter {
    std::FILE*    f;                       // borrowed; caller closes
    std::uint16_t n_cols;                  // top-level column count
    std::uint16_t n_desc;                  // total flattened field count
    std::uint16_t open;                    // 1 between open() and close()
    std::uint16_t failed;                  // latched on any write error
    IpcFieldDesc  desc[kIpcMaxFields];
    std::uint8_t  fb[kIpcFbCap];           // flatbuffer build scratch
};

/// Begin a stream: validate the schema (fail closed on any unsupported
/// type) and write the Schema message to `f`. `types` are BoltType
/// values; `names[i]` may be nullptr (a "cN" name is synthesized —
/// pyarrow requires non-null field names). `decimal_scales[i]` (0..38)
/// supplies the Decimal128 scale for any column whose type is
/// BoltType::Decimal128; it may be nullptr only when no column is
/// Decimal128 (arrow_ipc_open fails closed otherwise, since the schema's
/// Decimal type table must carry a scale and open() is the only point
/// that knows it before any batch is written). Every column is a leaf —
/// use arrow_ipc_open_nested() for a List/Struct column.
bool arrow_ipc_open(ArrowIpcWriter* w, std::FILE* f,
                    const BoltType* types, const char* const* names,
                    std::uint16_t n_cols,
                    const std::uint8_t* decimal_scales = nullptr) noexcept;

/// Begin a stream whose columns may be List/Struct: like arrow_ipc_open(),
/// but each of the `n_cols` top-level columns is a recursive `FieldSpec`
/// so a nested column's own child field type(s) are known before any
/// batch is written. `fields` and every FieldSpec it (transitively)
/// points to must be valid only for the duration of this call — nothing
/// is retained past it. Fails closed (returns false, writes nothing) on
/// an unsupported leaf type, a List/Struct child-count mismatch, an
/// out-of-range Decimal128 scale at any depth, nesting deeper than
/// kIpcMaxNestDepth, or more than kIpcMaxFields total fields.
bool arrow_ipc_open_nested(ArrowIpcWriter* w, std::FILE* f,
                           const FieldSpec* fields,
                           std::uint16_t n_cols) noexcept;

/// Append one RecordBatch message from the batch's read-epoch columns.
/// Column count and (recursively, for List/Struct) every field's type
/// must match the open schema; row values are resolved with the C-Data
/// mapping (see header comment) for leaves, and by walking
/// `ColumnFormat::Nested` (bolt_column.h) for List/Struct. Returns false
/// (and latches failure) on any mismatch or I/O error.
bool arrow_ipc_write_batch(ArrowIpcWriter* w,
                           const BoltBatch* batch) noexcept;

/// Write the end-of-stream marker and flush. The FILE* stays open
/// (borrowed). Returns false if the stream previously failed or the
/// final write fails; the writer is closed either way.
bool arrow_ipc_close(ArrowIpcWriter* w) noexcept;

}  // namespace bolt::ingest
