# V1 Iceberg schema history representations

V1 metadata can contain both legacy `schema` and modern `schemas`. The reader
previously appended both, duplicating the current schema and making its result
depend on JSON member order. The modern history now takes precedence. If legacy
was read first its appended count is reset before modern history; if modern was
read first the legacy object is skipped. Mixed forms require a declared current
schema belonging to the modern history. Duplicate modern arrays refuse.

Five focused cases pass through the Gestalt shared Clang build: both member
orders preserving two historical schemas, legacy-only/modern-only compatibility,
missing current schema in mixed forms, duplicate modern arrays, and current ID
absent from modern history. Existing metadata IO23 and interop4 also pass.
Generated complete v1 metadata parses with PyIceberg0.12.0 and round-trips its
serializer. General duplicate IDs/default-current behavior outside mixed forms
is unchanged; no claim of global strictness or lossless Metadata POD coverage.
