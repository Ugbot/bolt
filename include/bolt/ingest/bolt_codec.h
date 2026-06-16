// bolt/ingest/bolt_codec.h — shared result codes for the block codec family
// (zstd / gzip / lz4). int 0 = ok, nonzero = error (bolt I/O convention).

#pragma once

namespace bolt {
namespace ingest {

enum : int {
    kCodecOk           = 0,
    kCodecUnavailable  = 1,   // library not compiled in
    kCodecBadArg       = 2,   // null/zero where not allowed
    kCodecBufferSmall  = 3,   // dst capacity insufficient
    kCodecCorrupt      = 4,   // malformed compressed stream / lib error
};

}  // namespace ingest
}  // namespace bolt
