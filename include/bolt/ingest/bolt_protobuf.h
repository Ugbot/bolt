// bolt/ingest/bolt_protobuf.h — dynamic, descriptor-driven Protobuf decoder.
//
// Sibling of bolt_avro.h. Backs Kafka/Schema-Registry protobuf payloads and any
// other place a protobuf message must be turned into columns at RUNTIME.
//
// NO CODE GENERATION AND NO libprotobuf. Nothing here is generated from a
// .proto and there is no third-party dependency (CLAUDE.md bans protobuf as a
// dependency). The decoder is fully dynamic: it walks the wire format and
// matches fields against a descriptor table parsed at runtime.
//
// ---------------------------------------------------------------------------
// Three independent layers
// ---------------------------------------------------------------------------
//  1. WIRE READER (needs no schema at all). Every field on the wire is a
//     varint tag `(field_number << 3) | wire_type` followed by a payload whose
//     LENGTH is always derivable from the wire type alone:
//         0 varint | 1 fixed64 | 2 length-delimited | 5 fixed32
//         (3/4 = start/end group — deprecated, see "Not supported")
//     So any message can be WALKED and unknown fields SKIPPED without knowing
//     what they mean. That property is what makes protobuf forward-compatible,
//     and it is the foundation of layers 2 and 3.
//
//  2. DESCRIPTOR PARSING. A FileDescriptorSet is ITSELF a protobuf message, so
//     layer 1 parses it — bootstrapped with a small constant table of
//     descriptor.proto's own field numbers (see kFdp*/kDp*/kFd* below). That
//     bootstrap is a handful of integer constants, not generated code.
//
//  3. MESSAGE DECODE against a descriptor: match by FIELD NUMBER (fields may
//     arrive in any order and may repeat), convert to PbValue, skip unknown
//     field numbers, and handle packed repeated scalars.
//
// ---------------------------------------------------------------------------
// Supported
// ---------------------------------------------------------------------------
//   * All scalar types: double, float, int32/64, uint32/64, sint32/64 (zigzag),
//     fixed32/64, sfixed32/64, bool, string, bytes, enum.
//   * Fields in any order; repeated fields; packed repeated scalars (the
//     proto3 default) as well as the unpacked form.
//   * Unknown field numbers skipped losslessly (forward compatibility).
//   * Nested messages — reported as a raw byte slice plus the resolved
//     descriptor index, so the caller decodes recursively at its own depth.
//     The decoder itself never recurses, so a hostile deeply-nested message
//     cannot blow the stack.
//   * Malformed input fails closed: every read is bounds-checked, varints are
//     capped at 10 bytes, and a truncated payload returns false rather than
//     reading past the buffer.
//
// ---------------------------------------------------------------------------
// NOT supported (deliberate, documented gaps — a silent mis-decode would be
// worse than an honest refusal)
// ---------------------------------------------------------------------------
//   * GROUPS (wire types 3/4). Deprecated since proto2; pb_skip_field and the
//     decoders REJECT them rather than guess. Modern .proto files never use
//     them.
//   * maps — on the wire a map<K,V> is a repeated message of {key=1, value=2},
//     so it decodes as a repeated nested message today; there is no
//     map-flattening convenience layer yet.
//   * oneof — the members decode as ordinary optional fields (which is exactly
//     what they are on the wire); the "which one is set" discriminator is not
//     surfaced.
//   * Field defaults from the descriptor (proto2 `[default = x]`) — an absent
//     field reports is_null, not its declared default.
//   * extensions / Any / well-known-type unwrapping (Timestamp, Duration, …):
//     they decode as ordinary nested messages, uninterpreted.
//
// Tiger Style: PODs, Arena allocs, fixed caps, ≥2 asserts/fn, bounded loops,
// no exceptions, no std::string/vector/map.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bolt/bolt_arena.h"
#include "bolt/bolt_types.h"

namespace bolt {
namespace ingest {

// ---- bounds ---------------------------------------------------------------

static constexpr uint32_t kPbMaxName        = 64u;   // one identifier
static constexpr uint32_t kPbMaxFullName    = 192u;  // "pkg.Outer.Inner"
static constexpr uint32_t kPbMaxFields      = 64u;   // fields per message
static constexpr uint32_t kPbMaxMessages    = 32u;   // messages per set
static constexpr uint32_t kPbMaxNesting     = 8u;    // nested_type depth
static constexpr uint32_t kPbMaxMsgIndexes  = 16u;   // Confluent index array
static constexpr uint32_t kPbVarintMaxBytes = 10u;   // 64-bit varint ceiling

// ---- wire format ----------------------------------------------------------

enum class PbWireType : uint8_t {
    kVarint = 0, kFixed64 = 1, kLenDelim = 2,
    kStartGroup = 3, kEndGroup = 4,          // deprecated — rejected
    kFixed32 = 5,
};

// FieldDescriptorProto.Type — the numbering is descriptor.proto's own and is
// part of the wire-visible contract, so these constants are stable.
enum class PbType : uint8_t {
    kUnset = 0,
    kDouble = 1, kFloat = 2, kInt64 = 3, kUint64 = 4, kInt32 = 5,
    kFixed64 = 6, kFixed32 = 7, kBool = 8, kString = 9, kGroup = 10,
    kMessage = 11, kBytes = 12, kUint32 = 13, kEnum = 14,
    kSfixed32 = 15, kSfixed64 = 16, kSint32 = 17, kSint64 = 18,
};

// FieldDescriptorProto.Label.
enum class PbLabel : uint8_t { kUnset = 0, kOptional = 1, kRequired = 2, kRepeated = 3 };

// ---- decoded value --------------------------------------------------------

// One decoded field. Shape mirrors AvroValue so the two ingest paths feel like
// one family. bytes/msg slices point INTO the caller's source buffer (zero
// copy) — they are valid for as long as that buffer is.
//
// REPEATED SEMANTICS (chosen deliberately): pb_decode_message returns ONE
// PbValue per descriptor field, so it cannot carry a list. For a repeated
// field it reports rep_count = the number of occurrences and the value of the
// FIRST one. Use pb_decode_repeated_field() to collect every occurrence. For a
// NON-repeated field that appears more than once the protobuf spec says
// last-one-wins, so the LAST is kept (and rep_count records that it repeated).
struct PbValue {
    PbType   type;
    bool     is_null;          // true ⇒ field absent from this message
    uint8_t  _pad[2];
    uint32_t rep_count;        // occurrences on the wire (0 ⇒ absent)
    union {
        int64_t  i64;          // signed ints, enums, bool
        uint64_t u64;          // unsigned + fixed
        double   f64;          // float (widened) / double
    } num;
    const uint8_t* bytes;      // string / bytes / nested-message payload
    uint32_t       bytes_len;
    int32_t        msg_index;  // kMessage ⇒ index into PbDescriptorSet, else -1
};

// ---- descriptor -----------------------------------------------------------

struct PbFieldDesc {
    char     name[kPbMaxName];
    char     type_name[kPbMaxFullName];  // for kMessage/kEnum: ".pkg.Msg"
    int32_t  number;
    PbType   type;
    PbLabel  label;
    bool     repeated;                   // label == kRepeated
    bool     nullable;                   // !repeated && label != kRequired
    int32_t  msg_index;                  // resolved kMessage target, else -1
    uint32_t _pad;
};

struct PbMessageDesc {
    char        full_name[kPbMaxFullName];  // "pkg.Outer.Inner"
    PbFieldDesc field[kPbMaxFields];
    uint32_t    n_fields;
    uint32_t    _pad;
};

// A parsed FileDescriptorSet, flattened: nested types are hoisted to the top
// level with dotted full names, so lookup is one flat scan.
//
// !! SIZE WARNING !! This is ~half a megabyte (kPbMaxMessages fixed-capacity
// PbMessageDesc, each carrying kPbMaxFields PbFieldDesc). NEVER declare one as
// a stack local — that is a STATUS_STACK_BUFFER_OVERRUN waiting to happen, the
// same trap BoltBatch and MergeIter have. Allocate it from an Arena (or as a
// file-scope/heap object) and parse into it once; a descriptor is cold, long-
// lived data, so one allocation per schema is the intended shape.
struct PbDescriptorSet {
    PbMessageDesc msg[kPbMaxMessages];
    uint32_t      n_msgs;
    uint32_t      _pad;
};

// ---------------------------------------------------------------------------
// Layer 1 — wire reader (schema-free)
// ---------------------------------------------------------------------------

// Base-128 varint. Advances *off. False on truncation or > 10 bytes.
bool pb_read_varint(const uint8_t* src, uint64_t len, uint64_t* off,
                    uint64_t* out) noexcept;

// Read one field tag. Returns the field number + wire type.
bool pb_read_tag(const uint8_t* src, uint64_t len, uint64_t* off,
                 int32_t* out_field, PbWireType* out_wire) noexcept;

// Skip the payload of a field whose tag was just read. Rejects groups.
bool pb_skip_field(const uint8_t* src, uint64_t len, uint64_t* off,
                   PbWireType wire) noexcept;

// ZigZag decode (sint32/sint64).
inline int64_t pb_zigzag_decode(uint64_t u) noexcept {
    return static_cast<int64_t>((u >> 1) ^ (~(u & 1u) + 1u));
}

// ---------------------------------------------------------------------------
// Layer 2 — descriptor parsing (a FileDescriptorSet is itself protobuf)
// ---------------------------------------------------------------------------

// Parse a serialized FileDescriptorSet into a flat message table. Nested types
// are hoisted with dotted names; kMessage/kEnum type_names are resolved to
// msg_index where the target is present in the same set. Returns false on
// malformed input or when a bound (kPbMaxMessages/kPbMaxFields/kPbMaxNesting)
// would be exceeded — bounded, never truncating silently.
bool pb_parse_descriptor_set(const uint8_t* src, uint64_t len,
                             Arena* arena, PbDescriptorSet* out) noexcept;

// Find a message by full name ("pkg.Msg") or by a leading-dot type_name
// (".pkg.Msg"). Returns index, or -1.
int32_t pb_find_message(const PbDescriptorSet* set, const char* full_name) noexcept;

// ---------------------------------------------------------------------------
// Layer 3 — decode a message against a descriptor
// ---------------------------------------------------------------------------

// Decode `src` against `desc`, filling out_vals[0..desc->n_fields) — one entry
// per DESCRIPTOR field, in descriptor order (NOT wire order). Absent fields get
// is_null = true. Unknown field numbers on the wire are skipped. Returns false
// only on malformed input (truncation, bad varint, group).
//
// `set` may be null; when supplied, kMessage fields get msg_index resolved.
bool pb_decode_message(const uint8_t* src, uint64_t len,
                       const PbMessageDesc* desc, const PbDescriptorSet* set,
                       PbValue* out_vals, uint32_t out_cap) noexcept;

// Collect EVERY occurrence of one field number into out_vals (up to out_cap).
// Handles both the packed (wire type 2 carrying a run of scalars) and unpacked
// encodings. *out_n gets the count written; a count == out_cap means the caller
// should re-run with a larger buffer. Returns false on malformed input.
bool pb_decode_repeated_field(const uint8_t* src, uint64_t len,
                              int32_t field_number, PbType type,
                              PbValue* out_vals, uint32_t out_cap,
                              uint32_t* out_n) noexcept;

// ---------------------------------------------------------------------------
// Confluent Schema-Registry framing (Kafka)
// ---------------------------------------------------------------------------
//
//   [0x00 magic][4-byte BE schema id][message-index varint array][payload]
//
// The message-index array is protobuf-SPECIFIC (Avro has no such thing): it
// selects WHICH message inside the schema file the payload is, as a path of
// indexes through the file's message_type list. Confluent encodes the very
// common "first message" case as the single byte 0x00 rather than a [1, 0]
// array — both forms are accepted here.
//
// This helper is provided so callers don't re-derive the framing; it hands
// back the payload slice, which is what layer 3 wants.
bool pb_confluent_parse(const uint8_t* src, uint64_t len,
                        uint32_t* out_schema_id,
                        int32_t* out_msg_index, uint32_t idx_cap,
                        uint32_t* out_n_idx,
                        const uint8_t** out_payload,
                        uint64_t* out_payload_len) noexcept;

}  // namespace ingest
}  // namespace bolt
