// bolt/obs/obs_limits.h — capacities and run-time switches of bolt::log,
// bolt::metrics and bolt::trace.
#pragma once

#include "bolt/bolt_limits.h"

#define BOLT_OBS_LIMITS(X) \
    X(obs_log_ring_slots, kFixedAtAlloc, "records", 256, 16, 1u << 16, \
      "BOLT_OBS_LOG_RING_SLOTS", "GESTALT_OBS_LOG_RING_SLOTS", "log records per thread ring (power of two)") \
    X(obs_log_rings, kFixedAtAlloc, "rings", 1024, 1, 1u << 16, \
      "BOLT_OBS_LOG_RINGS", "GESTALT_OBS_LOG_RINGS", "threads with a private log ring") \
    X(obs_log_tail_bytes, kFixedAtAlloc, "bytes", 1u << 20, 4096, 1u << 28, \
      "BOLT_OBS_LOG_TAIL_BYTES", "GESTALT_OBS_LOG_TAIL_BYTES", "formatted log bytes retained in memory for support bundles and crash files") \
    X(obs_log_file_bytes, kBudget, "bytes", 256u << 20, 1u << 16, 1ull << 40, \
      "BOLT_OBS_LOG_FILE_BYTES", "GESTALT_OBS_LOG_FILE_BYTES", "rotate the log file at this size") \
    X(obs_log_file_keep, kBudget, "files", 8, 1, 1024, \
      "BOLT_OBS_LOG_FILE_KEEP", "GESTALT_OBS_LOG_FILE_KEEP", "rotated log files kept") \
    X(obs_log_flush_ms, kBudget, "ms", 10, 1, 10000, \
      "BOLT_OBS_LOG_FLUSH_MS", "GESTALT_OBS_LOG_FLUSH_MS", "log writer idle poll interval") \
    X(obs_metrics_enabled, kBudget, "bool", 1, 0, 1, \
      "BOLT_OBS_METRICS_ENABLED", "GESTALT_OBS_METRICS", "record metrics at run time (live: admin.obs.set)") \
    X(obs_metrics_cells, kFixedAtAlloc, "cells", 1u << 15, 1024, 1u << 22, \
      "BOLT_OBS_METRICS_CELLS", "GESTALT_OBS_METRICS_CELLS", "counter and histogram-bucket cells per shard") \
    X(obs_metrics_series, kFixedAtAlloc, "series", 1u << 14, 256, 1u << 22, \
      "BOLT_OBS_METRICS_SERIES", "GESTALT_OBS_METRICS_SERIES", "registered series; past it a series shares the overflow cell") \
    X(obs_metrics_shards, kFixedAtAlloc, "shards", 64, 1, 1024, \
      "BOLT_OBS_METRICS_SHARDS", "GESTALT_OBS_METRICS_SHARDS", "counter shards; a thread owns one, threads past this share") \
    X(obs_trace_sample_ppm, kBudget, "ppm", 0, 0, 1000000, \
      "BOLT_OBS_TRACE_SAMPLE_PPM", "GESTALT_OBS_TRACE_SAMPLE_PPM", "head sampling rate per million requests; 0 is off (live: admin.obs.set)") \
    X(obs_trace_ring_spans, kFixedAtAlloc, "spans", 4096, 64, 1u << 20, \
      "BOLT_OBS_TRACE_RING_SPANS", "GESTALT_OBS_TRACE_RING_SPANS", "finished spans queued for export; past it spans drop and count (power of two)") \
    X(obs_trace_retained_spans, kFixedAtAlloc, "spans", 4096, 64, 1u << 20, \
      "BOLT_OBS_TRACE_RETAINED_SPANS", "GESTALT_OBS_TRACE_RETAINED_SPANS", "finished spans kept for admin.trace.list") \
    X(obs_profile_enabled, kBudget, "bool", 0, 0, 1, \
      "BOLT_OBS_PROFILE_ENABLED", "GESTALT_OBS_PROFILE", "allow the sampling profiler action (live: admin.obs.set)") \
    X(obs_profile_max_seconds, kBudget, "s", 30, 1, 300, \
      "BOLT_OBS_PROFILE_MAX_SECONDS", "GESTALT_OBS_PROFILE_MAX_SECONDS", "longest profiler run") \
    X(obs_profile_samples, kFixedAtAlloc, "samples", 16384, 256, 1u << 20, \
      "BOLT_OBS_PROFILE_SAMPLES", "GESTALT_OBS_PROFILE_SAMPLES", "samples one profiler run keeps; past it samples drop and count")

namespace bolt::obs {
BOLT_LIMITS_TABLE(obs_limits, "bolt.obs", BOLT_OBS_LIMITS)
}  // namespace bolt::obs
