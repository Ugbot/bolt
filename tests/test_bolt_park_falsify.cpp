// test_bolt_park_falsify.cpp — falsifier twins of the park lost-wake stress
// (G2CHK-1293). Built with BOLT_PARK_FALSIFY set to a lever that models a
// forbidden reordering; the test passes only if the stress reports a lost wake,
// which proves the stress can see one.
#include "test_bolt_park_stress.h"

#include <cstdio>

int main() {
    const auto r = park_stress::run(20000, bolt::park::WaitMode::kPark, true);
    std::printf("%s %llu (max late %llu ns, lever %d)\n", r.lost != 0 ? "LOST_WAKE" : "NO_LOST_WAKE",
                static_cast<unsigned long long>(r.lost), static_cast<unsigned long long>(r.max_late_ns),
                BOLT_PARK_FALSIFY);
    return 0;
}
