# SBE wire fixtures

`sbe_fixtures.h::limits50` is the 54-byte SBE message inside the 68-byte
UDP example published by [CME](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457638169),
retrieved 2026-10-13. Remove its 12-byte packet header and 2-byte inclusive
message size; no locally implemented encoder produces these fixture bytes.
The original packet SHA-256 is
`0c16aa4f93b18641e714235a6bfc0da8d6dce66ea00d36c4da525b92740ac5f6`.
The page's embedded historical schema is ID 1, version 9, description
20210310, template 50 (`MDIncrementalRefreshLimitsBanding50`). This is not a
production Book46 schema, licensed feed or current energy capture.

Expected fixed header: block length 11, template 50, schema 1, version 9.
Root: transaction time 1633099253937623627 and indicator 0, then two padding
bytes. Dimensions: uint16 block length 32 / uint8 count 1. Entry: optional
high-limit mantissa INT64_MAX (null), low 9000000000000, band 10000000000,
security 5620, report sequence 1869. Prices have constant exponent -9, which
does not occupy bytes. The public page retains CME copyright notices; this
small attributed protocol example does not establish redistribution rights
for full schemas or captures.

`nested` is a separate owned, synthetic 42-byte test case, manually assembled
from `sbe-synthetic-schema.xml`. It covers two outer entries, one nonempty and
one empty nested group, float 1.5, signed integers -2/9, packed enum/set raw
values, per-entry data `abc`/empty, and trailing data `xyz`. Expected values
are hard-coded independently of the decoder. It is not an exchange capture
or a substitute for differential testing against generated codecs. Generic
wire primitives preserve raw enum/set values; schema adapters validate them.
