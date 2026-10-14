# LZ4 blocks with a bounded, unknown decoded size

`lz4_raw_decompress_bounded` accepts compressed bytes and caller-owned output
capacity, and reports the actual decoded length. Optiq MDG's compressed packet
body motivates this generic seam; no venue framing is part of Bolt.

The [official LZ4 block specification](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md)
describes compressed length plus an upper output bound as sufficient decoder
metadata. The existing Parquet-facing exact-size API keeps its stricter expected
length contract. Both entry points share the decoder through a compile-time
policy; no per-token mode branch, allocation or extra copy is introduced.

The bounded API requires a nonempty encoded block. A single zero token decodes
to an empty result. On failure the reported length is zero; output bytes are
scratch until success. Source, destination and length storage must be disjoint.
Capacity is not an expected length or integrity proof: callers must validate
their complete packet/message framing and preserve raw provenance separately.

Local Release Clang `test_bolt_lz4_raw` passes both APIs against committed liblz4
vectors, generated differential/oracle workloads, capacity canaries, malformed
offsets/literal truncation, and empty/null cases. The exact API still rejects
wrong expected output lengths. No new throughput or latency claim is made.
