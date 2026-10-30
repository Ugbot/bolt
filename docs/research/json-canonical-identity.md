# Lossless JSON object identity

`bolt/parse/bolt_json_canonical.h` adds an explicit comparison form for durable
request identity. Existing source-faithful and options writer modes are
unchanged. This API is not RFC8785/JCS and never converts number text through
floating point. Integer precision, decimals, exponent spelling and signed zero
are retained exactly;1.0 and1e0 remain distinct. It sorts object keys by decoded
UTF-8 bytes, preserves array order and normalizes string/key escapes through
the maintained ASCII emitter. Unicode normalization is not performed.

Decoded duplicate keys at every depth, malformed JSON, invalid UTF-8 and bad
Unicode escapes refuse. Input/output bytes, value nodes and depth have explicit
caller budgets. Outputs remain unchanged on failure; arena scratch may be
consumed. Output has independent arena custody. No IO, external dependency or
mutation of input. The source must survive the call, not the returned output.

`measure_serialized` exposes the options writer's exact size/UTF-8/depth pass
without allocation. Canonicalization uses it to enforce its output byte budget
before serialization, including ASCII expansion. The original byte-faithful
writer remains useful where ordering/number/escape spelling is itself part of
the identity; callers choose the explicit API rather than changing its default.

Shared Gestalt Clang build verification: canonical7 and existing writer9 cases
pass. Tests include escaped-equivalent keys/strings, decoded duplicate keys,
embedded NUL keys, strict Unicode/UTF-8 refusal, preserved wide numerics, array
order, budgets, idempotence, independent output lifetime and original writer
behavior. Existing shared-header/link warnings remain; no platform or
performance claim.
