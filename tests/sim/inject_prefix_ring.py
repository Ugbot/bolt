#!/usr/bin/env python3
"""Prove the deterministic scheduler simulation can SEE the W4-L1 ring defect.

A gate that has never failed on broken code is not known to be a gate. This
script re-injects the pre-fix claim-then-read ordering into
`include/bolt/bolt_scheduler.h`, rebuilds, and requires `test_bolt_sched_sim` to
FAIL; then restores the header, rebuilds, and requires it to PASS. The header is
restored in a `finally`, so an interrupted run does not leave the tree broken.

The defect (bolt f390bae, docs/research/scheduler-ring-exactly-once.md):
`try_claim_and_execute` used to bump `tail` and only then read the slot. Since
`submit()`'s backpressure keys on `tail`, the instant a claim advances tail the
producer may overwrite the slot that claim just took — the worker then runs the
producer's NEW task and the claimed one never runs, in exactly balanced pairs.

Usage:
    python3 tests/sim/inject_prefix_ring.py --build-dir /path/to/build
    python3 tests/sim/inject_prefix_ring.py --build-dir DIR --seeds 4

Exit status 0 means: the suite FAILED on the pre-fix ring and PASSED on the
shipping one — i.e. it discriminates.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve()
BOLT = HERE.parents[2]
HEADER = BOLT / "include" / "bolt" / "bolt_scheduler.h"
TARGET = "test_bolt_sched_sim"

# The shipping (fixed) ordering: copy the slot, THEN claim it.
SHIPPING = """        // Copy BEFORE claiming — see the contract above.
        const Task task = ring[t & kTaskRingMask];
        Sim::point(sched_point::kClaimCopied);

        if (!tail.compare_exchange_weak(t, t + 1,
                std::memory_order_release, std::memory_order_relaxed)) {
            Sim::point(sched_point::kClaimLost);
            return false;  // Another worker claimed it; discard the copy.
        }
        Sim::point(sched_point::kClaimWon);
"""

# The pre-fix ordering: claim, THEN read the slot. Byte-for-byte the W4-L1 bug,
# with the interleaving points left exactly where they are in the shipping body
# so the only difference under simulation is the ordering itself.
PREFIX = """        // *** INJECTED PRE-FIX ORDERING (inject_prefix_ring.py) ***
        Sim::point(sched_point::kClaimCopied);

        if (!tail.compare_exchange_weak(t, t + 1,
                std::memory_order_release, std::memory_order_relaxed)) {
            Sim::point(sched_point::kClaimLost);
            return false;
        }
        Sim::point(sched_point::kClaimWon);
        const Task task = ring[t & kTaskRingMask];   // <-- too late
"""


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(BOLT), text=True,
                          capture_output=True, **kw)


def build(build_dir):
    r = run(["cmake", "--build", build_dir, "--target", TARGET, "-j", "8"])
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit(f"build of {TARGET} failed")


def run_suite(build_dir, seeds):
    exe = pathlib.Path(build_dir) / "tests" / TARGET
    if not exe.exists():
        raise SystemExit(f"{exe} not found -- wrong --build-dir?")
    t0 = time.time()
    r = subprocess.run([str(exe), "--gtest_filter=*SeedSweep*"],
                       text=True, capture_output=True,
                       env={**__import__("os").environ,
                            "BOLT_SIM_SEEDS": str(seeds)})
    return r, time.time() - t0


def failing_seeds(output):
    return re.findall(r"FAILING SEED: scenario=(\S+) seed=(\d+)", output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    original = HEADER.read_text()
    if SHIPPING not in original:
        raise SystemExit(
            "the shipping claim ordering was not found verbatim in "
            f"{HEADER}. Refusing to run: a silent no-op injection would report "
            "discriminating power this gate does not have.")

    ok = False
    try:
        HEADER.write_text(original.replace(SHIPPING, PREFIX, 1))
        print("== pre-fix ring injected; rebuilding ==")
        build(args.build_dir)
        bad, bad_secs = run_suite(args.build_dir, args.seeds)
        seeds = failing_seeds(bad.stdout + bad.stderr)
        print(f"   pre-fix run: rc={bad.returncode} in {bad_secs:.2f}s, "
              f"{len(seeds)} failing (scenario, seed) pairs")
        for name, s in seeds[:10]:
            print(f"     FAIL  {name:<26} seed={s}")
    finally:
        HEADER.write_text(original)

    print("== header restored; rebuilding shipping ring ==")
    build(args.build_dir)
    good, good_secs = run_suite(args.build_dir, args.seeds)
    print(f"   shipping run: rc={good.returncode} in {good_secs:.2f}s")

    if bad.returncode == 0:
        print("\nVERDICT: NO DISCRIMINATING POWER — the suite passed on the "
              "pre-fix ring. Do not trust it as a gate.")
    elif good.returncode != 0:
        print("\nVERDICT: INCONCLUSIVE — the suite also fails on the shipping "
              "ring, so its failure on the pre-fix ring proves nothing.")
        sys.stderr.write(good.stdout[-4000:])
    else:
        ok = True
        print(f"\nVERDICT: DISCRIMINATES. Pre-fix FAILS ({len(seeds)} pairs, "
              f"{bad_secs:.2f}s), shipping PASSES ({good_secs:.2f}s).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
