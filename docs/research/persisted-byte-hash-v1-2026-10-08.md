# Stable byte hashing for persisted filters

MarbleDB's retained StripeBlooms recorded one hash kind while their finalizer followed `BOLT_HASH_TIER`. Consulting those filters across different build tiers could wrongly reject a present key. Generic hashing remains tier-selected; persistence now has explicit fixed-v1 helpers.

`hash_bytes_accumulate_v1` freezes the existing seed, little-endian 8-byte block folding and tail packing before finalization. `hash_bytes_wyhash3_v1` applies the fixed WYHASH3 finalizer. `hash_bytes_sv_wyhash3_v1` and its batch overload preserve StringView inline/overflow resolution using compile-time dispatch. They allocate nothing. A future change to this contract needs a new named version and a new consumer format tag. The generic `hash_bytes` and `hash_bytes_sv` results retain their previous tier-selected semantics.

MarbleDB's new tag identifies fixed WYHASH3-v1. Its legacy reader can derive each of the three supported historical finalizers from one byte fold and reject only when all miss. This is conservative, with a higher possible false-positive rate; it does not infer the writer build from a reader macro or one positive probe.

Validation: `test_bolt_hash_bytes` passes nine tests under each of WYHASH3, XXH3 and MURMUR3 (27 passes), including independent pinned digests for empty, short, 8-byte, multi-block and embedded-zero data, fixed scalar/SV equivalence and unchanged generic dispatch. Downstream MarbleDB passed nine Bloom tests, five related regression executables, 2,560 exact matched query checks and 65,792 exact profiled point queries. Those downstream numbers are not an isolated hash-throughput benchmark. Windows validation is deferred by the user. Local build logs have pre-existing header warnings.
