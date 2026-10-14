# Exact integer-literal filters

`kernels/bolt_integer_filter.h` compares signed8/16/32/64 and unsigned8/16/32/64
columns with a signed64 literal. It classifies the literal against the column's
range before any narrowing. Negative unsigned thresholds and out-of-range
narrow thresholds therefore produce the correct constant predicate rather
than wrapping. The all-valid path reuses owned comparison kernels; the nullable
path honors bitmap offsets and skips null payloads. Dense and selected inputs
retain source index order. No allocation occurs in the kernel.

Root's macOS Release Clang test passed all eight widths, six comparisons,
fourteen boundary literals, dense/selected inputs and null masks on2026-10-14.
Chukonu physical tests also cover composed sliced views and frozen arena
growth. This is a literal-comparison primitive, not general mixed arithmetic
or an unsigned SQL-literal parser. Chukonu's existing Int64 fast path is
unchanged. Generic Compute/IN expressions over these additional widths must
remain explicit unsupported operations until their typed evaluator is ready;
they may not reinterpret unsigned bits as signed values.
