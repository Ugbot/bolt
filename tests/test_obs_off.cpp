// Compiled with every feature off: call sites must build and be nothing.
#include "bolt/obs/log.h"
#include "bolt/obs/metrics.h"
#include "bolt/obs/trace.h"

static_assert(BOLT_OBS_LOG == 0 && BOLT_OBS_METRICS == 0 && BOLT_OBS_TRACE == 0);

BOLT_LOG_MODULE("off");

int obs_off_probe(int x) {
    bolt::metrics::Counter c;
    bolt::metrics::Histogram h;
    c.inc();
    h.observe(static_cast<std::uint64_t>(x));
    BLOG_ERROR("value %d", x);
    bolt::trace::Span span("off");
    return x + 1;
}
