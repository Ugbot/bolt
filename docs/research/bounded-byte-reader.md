# Bounded borrowed byte reader

`include/bolt/wire/byte_reader.h` supplies a small shared parsing primitive:
unsigned/signed 1-, 2-, 4- and 8-byte little-endian reads, borrowed subspans and
checked skips. Every failure preserves the cursor and destination. Bounds use
subtraction after cursor validation, so a hostile count cannot wrap an offset
addition. Signed values preserve their two's-complement bits through
`std::bit_cast`; no alignment or host-endian assumption is required.

The caller supplies the actual immutable storage length and owns its lifetime.
The reader performs no allocation, IO, format recognition or semantic
validation. Existing format-private wire helpers remain unchanged.

`test_bolt_byte_reader` checks unaligned values, signed narrow/extreme values,
every truncated u64 prefix, null/invalid cursors, huge skips, end/empty spans
and unchanged outputs after failures. Chukonu's MDP3 framing module uses the
same primitive before any body decoding is enabled.
