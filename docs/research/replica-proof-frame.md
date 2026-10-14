# Bounded replica coverage frames

`bolt/net/replica_proof_frame.h` defines the version-2 Begin, Entry, End,
Decline and Ack codec. Its fixed little-endian header binds source namespace,
assignment, manifest/replay identity, lane, table incarnation/schema and finite
source cut. Entry ordinals describe delivery order; source LSNs may skip.
Begin seeds the digest chain, Entry includes the previous chain, and End
binds the final count and chain. A digest provides integrity, not authority.

Encoding and parsing use caller-owned buffers and leave the destination
unchanged on failure. The codec checks canonical reserved/per-kind fields,
DDL-only domain zero, payload/count limits, ordered progress prefixes and
checksums. The envelope fits the existing WAL fragmentation limit. A trusted
receiver must separately verify the assignment, exact sent prefix, durable
log and applied state. End or Ack alone cannot establish a retained query
snapshot or authorize a peer.

`test_bolt_replica_proof_frame` passes seven cases: independent wire offsets,
round trip, all truncations and single-byte corruption, limits, canonical
per-kind fields, digest chaining and Begin identity binding. The domain/table
checks include valid-checksum malformed frames across all five kinds and
unchanged parse destinations. Final Clang Release build emitted no warnings;
focused CTest passed in 0.26 seconds. This is codec correctness evidence,
not replica recovery, end-to-end admission or performance qualification.
