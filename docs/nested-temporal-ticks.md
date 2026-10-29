# Retained nested timestamp precision

BNT1 named schemas distinguish local Timestamp (enum 1) from TimestampUtc
(enum 5), retaining the declared unit independently of Int64 value storage.
Prior enum values and encoded row bytes are unchanged. Unknown enum members
refuse; readers that predate the extension refuse enum 5.

The maintained whole-file tree and immutable byte-source nested readers accept
an optional retain_nanoseconds flag. False preserves the existing microsecond
contract, including refusal of sub-microsecond data. True emits nanosecond
Timestamp/Duration leaves as exact Int64 ticks; the caller must retain their
unit and UTC metadata from the same footer. Other units retain existing behavior.
No data is truncated to make the opt-in path pass. Ordinary Iceberg nested
scanning opts in, preserving nested timestamp_ns/timestamptz_ns and absent-field
Int64 fill lanes. Strict private pinned nested admission remains unfinished.

The PyArrow-generated timestamp_ns.parquet fixture contains local/UTC values,
negative epochs, genuine sub-microsecond digits, null parents/leaves and two
row groups. Whole and bounded reads check exact integers after transport scratch
poisoning. Reproduce it with scripts/make_nested_ns_fixture.py. All 62 maintained
nested reader tests pass with existing oracle expectations unchanged; all 11
carrier/schema tests pass, including distinct local/UTC BNT1 round trips.
