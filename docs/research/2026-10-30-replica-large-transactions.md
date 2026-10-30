# Existing large-transaction WAL classes in strict envelopes

G2ICE-404 extends the strict data envelope's source-kind allowlist from stable
Marble WAL classes 1–10 to 1–12: staged transaction parts (11) and their final
atomic apply record (12). Named wire constants live beside the existing
transaction-envelope kind in `net/wal_frame.h`.

The codec validates syntax and integrity only. Membership, source identity,
part completeness, duplicate custody and promotion belong to consumers. Frame
size remains 4 MiB; the envelope version, byte layout, checksum, normal-source
parser and legacy codecs are unchanged. Unknown classes starting at 13 refuse.
The ten existing codec cases pass, with the golden round-trip loop extended to
both classes. This does not claim that a consumer supports large transactions.
