// Bounded v2 replica coverage transport. A checksum is NOT authority: trusted
// assignment metadata binds source epoch (including physical storage/tier
// namespace), table, incarnation/schema, cuts and sender before admission.
// Digests/End/ACK do not establish history retention or a query read lease.
#pragma once
#include "bolt/net/replica_proof_frame_detail.h"
#include "bolt/net/wal_frame.h"
namespace bolt::net {
inline constexpr std::uint32_t kReplicaProofMagic = 0x32505247u;
inline constexpr std::uint16_t kReplicaProofVersion = 2;
inline constexpr std::uint32_t kReplicaProofHeaderBytes = 252;
inline constexpr std::uint32_t kReplicaProofMaxPayload = 4u * 1024u * 1024u - 512u;
inline constexpr std::uint64_t kReplicaProofMaxEntries = 65536;
static_assert(kReplicaProofHeaderBytes + kReplicaProofMaxPayload + kFrameHeaderBytes <= kWalMaxFrameBytes,
              "proof envelope and outer WAL wrapper must fit fragmentation cap");
// Outer WAL wrapper uses this non-catalog tid and a disjoint kind. Only a
// peer which explicitly negotiated v2 may receive these wrappers.
inline constexpr std::uint32_t kReplicaProofControlTable = UINT32_MAX;
inline constexpr std::uint8_t kReplicaProofOuterKind = 0xf0;
enum class ReplicaProofKind : std::uint8_t { Begin = 1, Entry, End, Decline, Ack };
enum class ReplicaProofReason : std::uint16_t {
    None = 0, UnknownTable, SourceMismatch, SchemaMismatch, IncarnationMismatch,
    HistoryUnavailable, Busy, Budget, IoFailure, ApplyFailure, Gap, UnboundBaseline
};
struct ReplicaProofFrame {
    ReplicaProofKind kind = ReplicaProofKind::Begin;
    std::uint8_t source_epoch[32]{};
    std::uint64_t assignment = 0, manifest = 0, replay = 0;
    std::uint32_t domain = 0; // 0 = global DDL; source data lane index + 1.
    std::uint32_t table = 0, incarnation = 0;
    ReplicaProofReason reason = ReplicaProofReason::None;
    std::uint64_t base_lsn = 0, cut_lsn = 0, synced_lsn = 0;
    std::uint64_t ordinal = 0, count = 0;
    // Contiguous replay prefixes, not maxima of out-of-order frames seen.
    std::uint64_t received_ordinal = 0, durable_ordinal = 0, applied_ordinal = 0;
    std::uint8_t schema[32]{}, chain[32]{};
    // Per-kind: Begin/Entry/Decline have zero count/progress/synced_lsn;
    // Begin/Decline ordinal is zero. End ordinal=count and synced_lsn covers cut.
    // Ack count/synced_lsn are zero, ordinal=received prefix; trusted state also
    // checks ACK progress against the sent prefix/final count. End alone is not readiness.
    wire::ByteView payload{}; // Entry: nested existing WAL frame; borrowed.
};
namespace replica_proof_detail {
inline bool valid(const ReplicaProofFrame& f) noexcept {
    if (f.assignment == 0 || f.manifest == 0 || f.replay == 0 ||
        !nonzero_digest(f.source_epoch) || f.base_lsn > f.cut_lsn ||
        f.count > kReplicaProofMaxEntries || f.ordinal > kReplicaProofMaxEntries ||
        f.received_ordinal > kReplicaProofMaxEntries ||
        f.applied_ordinal > f.durable_ordinal || f.durable_ordinal > f.received_ordinal ||
        f.payload.size > kReplicaProofMaxPayload ||
        (f.payload.data == nullptr && f.payload.size != 0)) return false;
    if (f.table != 0 && (f.domain == 0 || f.incarnation == 0 || !nonzero_digest(f.schema))) return false;
    if (f.table == 0 && (f.domain != 0 || f.incarnation != 0 || nonzero_digest(f.schema))) return false;
    const bool progress = f.received_ordinal != 0 || f.durable_ordinal != 0 || f.applied_ordinal != 0;
    if (f.kind == ReplicaProofKind::Decline)
        return f.reason > ReplicaProofReason::None && f.reason <= ReplicaProofReason::UnboundBaseline &&
               f.payload.size == 0 && f.ordinal == 0 && f.count == 0 && !progress && f.synced_lsn == 0;
    if (f.reason != ReplicaProofReason::None) return false;
    switch (f.kind) {
        case ReplicaProofKind::Begin: return f.ordinal == 0 && f.count == 0 && !progress &&
                                            f.synced_lsn == 0 && f.payload.size == 0;
        case ReplicaProofKind::Entry: return f.ordinal != 0 && f.count == 0 && !progress &&
                                            f.synced_lsn == 0 && f.payload.size != 0;
        case ReplicaProofKind::End: return f.ordinal == f.count && !progress &&
                                          f.payload.size == 0 && f.synced_lsn >= f.cut_lsn;
        case ReplicaProofKind::Ack: return f.payload.size == 0 && f.count == 0 && f.synced_lsn == 0 &&
                                          f.ordinal == f.received_ordinal;
        case ReplicaProofKind::Decline: break;
    }
    return false;
}
inline void encode_header(const ReplicaProofFrame& f, std::uint8_t* out) noexcept {
    assert(out != nullptr); assert(valid(f));
    std::memset(out, 0, kReplicaProofHeaderBytes);
    write_le(out, kReplicaProofMagic); write_le(out + 4, kReplicaProofVersion);
    out[6] = static_cast<std::uint8_t>(f.kind);
    write_le(out + 8, kReplicaProofHeaderBytes + f.payload.size);
    std::memcpy(out + 16, f.source_epoch, 32);
    write_le(out + 48, f.assignment); write_le(out + 56, f.manifest); write_le(out + 64, f.replay);
    write_le(out + 72, f.domain); write_le(out + 76, f.table); write_le(out + 80, f.incarnation);
    write_le(out + 84, static_cast<std::uint16_t>(f.reason));
    write_le(out + 88, f.base_lsn); write_le(out + 96, f.cut_lsn); write_le(out + 104, f.synced_lsn);
    write_le(out + 112, f.ordinal); write_le(out + 120, f.count);
    std::memcpy(out + 128, f.schema, 32); std::memcpy(out + 160, f.chain, 32);
    write_le(out + 192, f.payload.size);
    write_le(out + 196, f.received_ordinal); write_le(out + 204, f.durable_ordinal);
    write_le(out + 212, f.applied_ordinal);
}
inline bool decode_header(wire::ByteView bytes, ReplicaProofFrame& f) noexcept {
    if (bytes.data == nullptr || bytes.size < kReplicaProofHeaderBytes) return false;
    wire::ByteReader r{bytes.data, bytes.size, 0};
    std::uint32_t magic = 0, length = 0, reserved = 0, payload = 0;
    std::uint16_t version = 0, reason = 0, pad = 0;
    std::uint8_t kind = 0, flags = 0;
    if (!wire::read_le(r, magic) || !wire::read_le(r, version) || !wire::read_le(r, kind) ||
        !wire::read_le(r, flags) || !wire::read_le(r, length) || !wire::read_le(r, reserved) ||
        magic != kReplicaProofMagic || version != kReplicaProofVersion || flags != 0 || reserved != 0 ||
        length != bytes.size || length > kReplicaProofHeaderBytes + kReplicaProofMaxPayload) return false;
    f.kind = static_cast<ReplicaProofKind>(kind);
    std::memcpy(f.source_epoch, bytes.data + r.offset, 32); r.offset += 32;
    if (!wire::read_le(r, f.assignment) || !wire::read_le(r, f.manifest) || !wire::read_le(r, f.replay) ||
        !wire::read_le(r, f.domain) || !wire::read_le(r, f.table) || !wire::read_le(r, f.incarnation) ||
        !wire::read_le(r, reason) || !wire::read_le(r, pad) || pad != 0 ||
        !wire::read_le(r, f.base_lsn) || !wire::read_le(r, f.cut_lsn) || !wire::read_le(r, f.synced_lsn) ||
        !wire::read_le(r, f.ordinal) || !wire::read_le(r, f.count)) return false;
    f.reason = static_cast<ReplicaProofReason>(reason);
    std::memcpy(f.schema, bytes.data + r.offset, 32); r.offset += 32;
    std::memcpy(f.chain, bytes.data + r.offset, 32); r.offset += 32;
    if (!wire::read_le(r, payload) || payload != bytes.size - kReplicaProofHeaderBytes ||
        !wire::read_le(r, f.received_ordinal) || !wire::read_le(r, f.durable_ordinal) ||
        !wire::read_le(r, f.applied_ordinal)) return false;
    f.payload = {bytes.data + kReplicaProofHeaderBytes, payload};
    return valid(f);
}
}  // namespace replica_proof_detail
// Failure leaves caller bytes/destination unchanged. Inputs and output storage
// must not overlap; caller owns all buffers for the duration of the operation.
inline std::uint32_t replica_proof_encode(const ReplicaProofFrame& f,
    std::uint8_t* out, std::uint32_t capacity) noexcept {
    if (out == nullptr || !replica_proof_detail::valid(f) ||
        capacity < kReplicaProofHeaderBytes + f.payload.size) return 0;
    assert(capacity >= kReplicaProofHeaderBytes); assert(f.payload.size <= kReplicaProofMaxPayload);
    replica_proof_detail::encode_header(f, out);
    if (f.payload.size != 0) std::memcpy(out + kReplicaProofHeaderBytes, f.payload.data, f.payload.size);
    replica_proof_detail::integrity(out, 220, {out + kReplicaProofHeaderBytes, f.payload.size}, out + 220);
    return kReplicaProofHeaderBytes + f.payload.size;
}
inline bool replica_proof_parse(wire::ByteView bytes, ReplicaProofFrame& out) noexcept {
    ReplicaProofFrame candidate{};
    if (!replica_proof_detail::decode_header(bytes, candidate)) return false;
    assert(bytes.size >= kReplicaProofHeaderBytes); assert(candidate.payload.size <= kReplicaProofMaxPayload);
    std::uint8_t digest[32]{};
    replica_proof_detail::integrity(bytes.data, 220, candidate.payload, digest);
    if (!replica_proof_detail::equal_digest(digest, bytes.data + 220)) return false;
    out = candidate;
    return true;
}
// Begin.chain names the already-proven baseline, or zero for a fresh namespace.
// Seed includes the entire Begin identity/cut. It grants no authority or readiness.
inline bool replica_proof_seed(wire::ByteView encoded, std::uint8_t out[32]) noexcept {
    if (out == nullptr) return false;
    ReplicaProofFrame begin{};
    if (!replica_proof_parse(encoded, begin) || begin.kind != ReplicaProofKind::Begin) return false;
    assert(encoded.size == kReplicaProofHeaderBytes); assert(begin.ordinal == 0);
    crypto::sha256(encoded.data, encoded.size, out);
    return true;
}
// Digest chaining binds each exact Entry envelope, including lane/epoch/cut and
// ordinal. Entry.chain holds the prior chain; End.chain holds the final chain.
// LSNs may skip; ordinals are replay-delivery order, never LSN+1.
inline void replica_proof_chain(const std::uint8_t prior[32], wire::ByteView encoded,
                                std::uint8_t out[32]) noexcept {
    assert(prior != nullptr && out != nullptr);
    assert(encoded.data != nullptr && encoded.size >= kReplicaProofHeaderBytes);
    std::uint8_t pair[64]{};
    std::memcpy(pair, prior, 32);
    crypto::sha256(encoded.data, encoded.size, pair + 32);
    crypto::sha256(pair, sizeof(pair), out);
}
}  // namespace bolt::net
