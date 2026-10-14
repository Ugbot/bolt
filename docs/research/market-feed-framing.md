# Borrowed MoldUDP64 and SoupBinTCP framing

Generic wire primitives, tracked as a prerequisite of G2FEAT-130. Neither
primitive opens sockets, owns a TCP accumulator or manages session recovery.
Venue decoding and publication rules remain in Chukonu.

`wire/mold_udp64.h` validates the entire downstream UDP datagram before exposing
its borrowed packet view. The ten-byte session is opaque, fields are big-endian,
count zero is a heartbeat and count 65535 is end-of-session. Message lengths can
be zero. Format limits, caller storage/message budgets, trailing bytes and
sequence arithmetic are checked; errors preserve the output. The caller owns
immutable packet lifetime and application-level admission.

`wire/soup_bin_tcp.h` reads exactly one logical frame from a caller-owned TCP
prefix. Partial input returns NeedMore, coalesced input returns the exact consumed
length, and declared capacity/direction/type/fixed-size violations refuse. ASCII
sequence parsing accepts left-space padding and checks uint64 overflow without
floating point. A Soup sequenced frame does not embed a binary sequence number;
transport owners must track their admitted stream and higher-layer semantics.

The selected primary specifications are Nasdaq
[MoldUDP64 1.00](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/moldudp64.pdf)
and [SoupBinTCP 3.00](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/soupbintcp.pdf).
The Chukonu Nasdaq package manifest records retrieved hashes and sizes; vendor
documents are retained locally rather than committed.

Local Release Clang built and passed `test_bolt_mold_udp64` and
`test_bolt_soup_bin_tcp` on 2026-10-14, alongside `chukonu_nasdaq_itch`.
Independent literal fixtures cover endian values, empty messages, truncation,
trailing bytes, partial/coalesced streams, type/direction, capacity and sequence
overflow. No throughput, tail latency, allocation-counter, live transport or
cross-platform qualification is inferred from these correctness tests.
