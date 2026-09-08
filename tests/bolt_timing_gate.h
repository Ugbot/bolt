// bolt_timing_gate.h — when a WALL-CLOCK assertion is allowed to gate.
//
// THE PROBLEM THIS SOLVES.  A correctness suite that contains a wall-clock
// assertion is a gate that cries wolf.  `test_bolt_argsort_parallel.cpp`
// asserted `EXPECT_LT(ns_per_elem_par, ns_per_elem_single)` — "parallel must
// be faster" — inside ctest, on a box whose OWN performance boards refuse to
// publish a timing at all because it has never been quiet enough.  It failed
// roughly one run in six under load while its real assertion, that the two
// kernels produce the identical permutation, passed every time.
//
// That is not a harmless flake.  This campaign has now twice had a REAL red
// misattributed to noise, because reds on this box are routine; every
// wall-clock assertion in the correctness suite raises the background rate of
// meaningless reds and makes the next real one cheaper to dismiss.
//
// THE RULE.  Correctness assertions gate, unconditionally and always.  A
// timing observation is REPORTED by default, and is only allowed to gate when
// the caller has established that the box is quiet — and "quiet" means the
// repo's own definition, `benchmarks/provenance.py: quiet_box()`, which
// checks load per CPU, the busiest foreign process, swap in use, free RAM and
// running containers, and reports the measured value beside every threshold.
//
// WHY AN ENV VAR AND NOT A CHECK IN C++.  Re-implementing quiet_box()'s
// thresholds here would create a second copy of them, free to drift from the
// first — which is precisely the defect class the rest of this wave is about.
// So this header holds NO thresholds.  It asks whether a caller that already
// ran quiet_box() said yes.  The one supported caller is
// `benchmarks/perf/timing_gate.py`, which calls quiet_box() and sets the
// variable only when it returns quiet; anything else setting it by hand is
// asserting the box is quiet on its own authority, and the test says so in
// its output so a reader can tell which happened.
//
// Next candidate for this header: `test_bolt_perf_gate.cpp`, which carries
// six bare absolute-threshold `EXPECT_LT`s and is the documented flake whose
// SERIAL number — containing no parallel code at all — swung 125.757 ->
// 464.451 ns/elem on identical input.  Not converted here; named so it is not
// forgotten.

#ifndef BOLT_TESTS_BOLT_TIMING_GATE_H
#define BOLT_TESTS_BOLT_TIMING_GATE_H

#include <cstdlib>
#include <cstring>

namespace bolt_test {

// True only when a caller that has run `provenance.quiet_box()` and got a
// quiet answer asked for timing assertions to gate.
inline bool timing_gate_enabled() {
    const char* v = std::getenv("BOLT_TIMING_GATE");
    return v != nullptr && std::strcmp(v, "0") != 0 && v[0] != '\0';
}

// What to print beside the measurement, so the output states which mode ran
// rather than leaving a reader to infer it from the absence of a failure.
inline const char* timing_gate_reason() {
    return timing_gate_enabled()
        ? "GATING (BOLT_TIMING_GATE set: the caller certified a quiet box "
          "via provenance.quiet_box())"
        : "REPORTED, NOT ASSERTED (BOLT_TIMING_GATE unset; a wall-clock "
          "assertion on a loud box is a gate that cries wolf — run "
          "benchmarks/perf/timing_gate.py to gate on a quiet box)";
}

}  // namespace bolt_test

#endif  // BOLT_TESTS_BOLT_TIMING_GATE_H
