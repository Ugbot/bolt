# Wire decoder bounds

The October 2026 Linux landing gate exposed a seeded malformed 17 KB blob
that claimed over 34 billion rows. An all-null Constant attempted to allocate
and clear a 4 GB validity bitmap before a later column could reject the row
count. Native success had hidden this allocation failure mode.

Both `bolt_wire_deserialize` and `bolt_wire_view` now preflight the complete
descriptor tree before allocating output ownership. Row-dependent buffer
lengths and VarBinary offsets must agree with their recorded spans. Nonzero
VarBinary origins remain valid: offsets must be nonnegative, monotone, bounded
by the payload, and end at its recorded length.

Synthesized all-null Constant bitmaps have a **64 MiB aggregate budget per
decode**, including nested and dictionary children. Configure it consistently
across a build with `BOLT_WIRE_MAX_NULL_BITMAP_BYTES`. Exceeding the budget
returns `false` before allocation. This changes no wire bytes, but readers can
refuse otherwise valid batches above their configured resource budget. Compact
non-null Constant and Sequence columns retain the existing row limit.

The original 20,000-iteration hostile-input sweep remains enabled. Deterministic
tests cover impossible rows, short VarBinary offsets, aggregate nested budget
boundaries, and nonzero-origin serialize/deserialize/view round trips. These
checks passed on Linux Clang and macOS; Windows validation remains pending.

The decoder validates structure and bounds; this does not replace frame CRC
verification. Future representation changes could avoid synthesizing a
per-row bitmap for all-null Constants, but are outside this repair.
