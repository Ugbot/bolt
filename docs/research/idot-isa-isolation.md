# IDOT compilation and performance isolation

The runtime-dispatched IDOT library must compile its scalar and dispatch
translation units at the compiler baseline. AVX2 and AVX512/VNNI sources receive
their explicit source options. Consumers keep the configured Bolt core ISA.

Previously the library linked core with PUBLIC usage requirements. Core's
native/AVX compile options consequently reached every library source, contrary
to the intended dispatch contract. An old-main generated x86 Ninja metadata
control confirms `-march=native` on scalar and dispatch, and native plus AVX2
on the AVX2 source. This metadata observation does not execute cross-target
code or establish the precise cause of an x86 benchmark ratio failure.

The library now links core through its consumer interface, receives shared
layout/feature definitions explicitly, and retains its own hardening, include
paths and thread requirements. No global compiler options supplied by a parent
are stripped. Parents requesting a portable binary must also avoid incompatible
global ISA flags. Kernel implementations and runtime tier selection are unchanged.

The existing IDOT benchmark runs serially within CTest. Its numerical controls,
shapes, iteration counts and five-times speedup floor remain unchanged. Earlier
Linux CI measured ratios near2.5 while the scheduler simulation ran alongside
it; neither contention nor native-option leakage alone establishes a regression.

Required qualification: inspect generated baseline and per-source options on
the relevant toolchains, run numeric tier-equivalence and alignment controls,
compare old and corrected code on the same x86 runner, preserve the existing
speedup floor, and complete Windows/Linux/macOS CI. An arm64 scalar-only result
cannot qualify x86 SIMD performance. This source change is not a passing result.
