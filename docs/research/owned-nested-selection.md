# Selected packed nested payloads

The existing nested_gather kernel accepts VarBinary leaves as well as Flat and
Nested trees. It packs only selected byte spans into arena-owned output, honors
validity_offset, nulls, arbitrary order and duplicate row identities, and rejects
invalid selected offsets or a payload past the signed 32-bit offset bound.
Unselected payloads are not interpreted. This lets an owned storage consumer
select live rows before decoding a typed nested value; tombstone payloads are
not values. Flat StringView spill bytes retain the existing borrowed contract.

The NestedGather.PackedBytesKeepSelectionNullsAndOwnPayload regression and the
13 existing nested Parquet selection cases pass in the Gestalt2 Clang build.
