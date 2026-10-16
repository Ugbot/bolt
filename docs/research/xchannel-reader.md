# Read-only xchannel framing

`wire/xchannel.h` implements bounded framing over caller-pinned mapped bytes.
It targets xchannel 7.0.0, FORMAT4, upstream commit
`49063b06aa31cba3ddd659a19808f1aed2e6fc26`. FORMAT3 explicitly returns
`Unsupported`; this is not a claim about an unknown client deployment.

`admit_file` validates the committed Channel prefix, geometry, identity CRC,
publication, expected sequence/generation and optional exact base index.
Prepared files never become admitted history. `begin` starts at the validated
208-byte prefix. `peek` returns a borrowed User, Skip or Roll without advancing;
`consume` validates the token and advances User/Skip only. `follow_successor`
checks a committed Roll against the admitted successor identity, predecessor,
generation, base, geometry and name before returning its starting cursor.
Failures preserve outputs/cursors. Missing/pruned files are caller errors to
classify, never an instruction to skip history.

The caller owns file discovery, mapping, retention and payload decoding. The
prefix and region must remain pinned during each operation; no borrowed view
may escape asynchronously without an owned pin or copy. A cursor must originate
from admission/replay; arbitrary saved offsets are not proven record boundaries.
Little-endian hosts, aligned storage and lock-free byte/64-bit hardware atomics
are required. Interprocess synchronization relies on the host mmap/atomic ABI.
Acquiring commitment establishes publication visibility, not fsync durability.
Generation zero is explicit but cannot detect recreation by itself.

Idle polls inspect only the commitment byte. Record count is read only after a
committed Roll, when the old segment's count is final. Advisory write position
is not used. Payload bytes remain opaque: message type and user metadata do not
imply SBE, Arrow, event time or a benchmark timestamp.

`tests/test_xchannel.cpp` contains independent literal upstream prefix/Roll
goldens, bounded truncation/mutation cases, exact successor checks, cursor
refusal, Skip transitions and delayed release/acquire publication. Local native
tests pass. The test target is registered only when `bolt::io` exists, because
identity/Roll checks reuse the owned CRC32C primitive. Both enabled and disabled
IO CMake configurations generate successfully.

Actual upstream Rust-writer/read-only mapped interoperability and deployment
platform qualification remain separate checks. These framing tests establish
neither a Chukonu connector nor query/action latency.
