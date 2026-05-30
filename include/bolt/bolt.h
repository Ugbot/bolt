#pragma once
// Single-include convenience header. Pulls the full bolt::core surface.
//
// Borrowed from cglm: one umbrella header so consumers who don't want
// module-by-module includes get a one-liner. Sibling modules (kernels,
// join, wire, arrow) auto-include themselves when their headers exist.
#include "bolt/bolt_port.h"
#include "bolt/bolt_types.h"
#include "bolt/bolt_arena.h"
#include "bolt/bolt_channel.h"
#include "bolt/bolt_column.h"
#include "bolt/bolt_branchless.h"
#include "bolt/bolt_scheduler.h"

// Future siblings — pulled in opportunistically once they land.
#if __has_include("bolt/kernels/bolt_numeric.h")
    #include "bolt/kernels/bolt_numeric.h"
#endif
#if __has_include("bolt/kernels/bolt_string.h")
    #include "bolt/kernels/bolt_string.h"
#endif
#if __has_include("bolt/kernels/bolt_temporal.h")
    #include "bolt/kernels/bolt_temporal.h"
#endif
#if __has_include("bolt/kernels/bolt_parallel.h")
    #include "bolt/kernels/bolt_parallel.h"
#endif
#if __has_include("bolt/join/bolt_swiss.h")
    #include "bolt/join/bolt_swiss.h"
#endif
#if __has_include("bolt/join/bolt_hashjoin.h")
    #include "bolt/join/bolt_hashjoin.h"
#endif
#if __has_include("bolt/join/bolt_groupby.h")
    #include "bolt/join/bolt_groupby.h"
#endif
#if __has_include("bolt/join/bolt_joinkernel.h")
    #include "bolt/join/bolt_joinkernel.h"
#endif
#if __has_include("bolt/wire/bolt_wire.h")
    #include "bolt/wire/bolt_wire.h"
#endif
#if __has_include("bolt/arrow/bolt_arrow.h")
    #include "bolt/arrow/bolt_arrow.h"
#endif
#if __has_include("bolt/stream/bolt_stream.h")
    #include "bolt/stream/bolt_stream.h"
#endif
#if __has_include("bolt/ingest/bolt_csv.h")
    #include "bolt/ingest/bolt_csv.h"
#endif
