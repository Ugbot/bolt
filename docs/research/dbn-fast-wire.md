# Bounded DBN and FAST wire primitives — 14 October 2026

`wire/dbn.h` frames Databento DBN versions1/2/3 with explicit little-endian
reads. Metadata and record views borrow immutable caller bytes. Version-specific
metadata offsets, counted symbol lists/mapping intervals and v3 alignment are
validated within the declared metadata body; counts cannot consume records.
Callers supply the metadata byte budget. Record lengths are u8 counts of
four-byte words, so the protocol maximum is1020bytes. Optional `ts_out` belongs
to that record length and is separate from the payload. NeedMore never advances
the caller cursor or publishes partial output.

The framing layer retains enum, string, date and sentinel bytes; it does not
interpret them as calendar dates, venue ticks, verified symbols or book state.
It frames unknown rtypes losslessly so typed consumers can reject unsupported
layouts. Generic framing is not a Databento live session or historical downloader.
Typed extraction/file reading belongs to Chukonu's `connectors/databento` module.

The primary format pin is Databento's Apache-2.0 reference repository
[`d368005a636bae84bff443cd6ddae6430a0ffd00`, tag0.71.0](https://github.com/databento/dbn/tree/d368005a636bae84bff443cd6ddae6430a0ffd00).
The independent reference CLI archive SHA-256 is
`19b169e76feb61a0f75080ddb36e8ec750e1eb5cdc5d5ab0b191ff4d113119a9`
(official macOS arm64 release). Chukonu's retained public fixtures and reference
decoder TSV values exercise this boundary; no third-party runtime is linked.

`wire/fast.h` supplies the separate, allocation-free FAST1.1 subset needed by
T7 EMDI: bounded stop-bit scalars, presence maps, fixed dictionary cells,
copy/increment/default/delta operators and exact decimal exponent/mantissa.
Undefined, absent and null are distinct. Invalid/unsupported inputs preserve
state; unsupported decimal-delta XML initial values refuse explicitly. An
absent scalar initial never supplies a dictionary base. Venue templates and
dictionary reset scope belong to the Chukonu adapter, not Bolt.

The FAST normative pin is [FIX FAST1.1 final specification](https://www.fixtrading.org/standards/fast/);
the acquired final1.x.1 artifact SHA-256 is
`5047ced8d6a06195eab777c12e09b41c139eadcfca87bccfbc4dacaab6e297b3`.
The T7 manifest in Chukonu records its exact URL, selected operators and
unsupported scope. `test_fast_wire` covers scalar extrema, nullability,
presence-map exhaustion and dictionary failure atomicity. These primitives
do not claim a generic XML interpreter, FAST1.2, live feed or venue qualification.
