# Bounded owned Zstandard stream decoding

G2FEAT-152 adds `bolt/ingest/bolt_zstd_stream.h` alongside the existing
whole-buffer decoder. The caller admits a history window and logical output
limit, then supplies fixed state and disjoint history/input/output buffers.
Each step performs one bounded transition or block action, consuming/producing
at most128KiB. Memory depends on the admitted window and block scratch, not
total file output. No allocation, I/O or engine dependency occurs during steps.

Input consumption, output backpressure, frame boundaries and physical EOF are
distinct. A last input suffix must be resubmitted until consumed; empty calls
can subsequently drain buffered output. Frame output is provisional until its
trailer validates. Checksum failure is sticky and cannot retract prior output;
an unchecksummed frame never reports checksum verification. Cancellation is the
caller's responsibility between bounded steps.

The admitted subset uses offset codes0..31, windows at mostUINT32_MAX and no
dictionary-ID field. Unsupported extended windows have a distinct status;
caller window/output budgets refuse explicitly. Concatenated frames reset
history/entropy/repeats; skippable frames do not become decoded bytes.

Private entropy helpers are shared by compile-time contiguous and ring output
policies. The existing whole-buffer API preserves its prior behavior; stricter
bitstream-end/repeat/history checks apply to the new stream policy. This is an
additive codec, not a silent behavior change to Parquet page decoding.

Root's macOS Release Clang checks passed the existing decoder corpus and16
stream tests on2026-10-14. Tests include independent reference frames, split
input and one-byte output, checksums, exact consumed-byte accounting,
truncation, frame-local history/window boundaries, alias admission and a
128MiB generated stream with fixed1KiB history and no C++ heap allocation
during decode. Authored malformed Huffman/sequence vectors test strict
exhaustion separately. These establish scoped correctness, not matched
throughput/tail performance, operating-system-wide allocation freedom or
Windows/Linux qualification. Chukonu's DBN Source still uses its existing
whole-file path until its versioned streaming adapter is implemented.

Primary specifications: [RFC8878](https://www.rfc-editor.org/rfc/rfc8878.html),
[Zstandard1.5.7 format](https://github.com/facebook/zstd/blob/v1.5.7/doc/zstd_compression_format.md).
The inclusive window-distance boundary follows the pinned reference decoder;
the Chukonu incremental-Zstd proposal records the RFC wording nuance and
remaining engine integration contracts.
