# G2CHK-714: bounded StringView spill cloning

`BoltColumn::clone_into` previously allocated/copied `[0,max(offset+length))`
for valid Flat/View Utf8 references. A dense slice can retain offsets into a
large parent spill buffer, making unused prefixes and gaps dominate its clone.

The owned `kernels/bolt_utf8_clone.h` helper now plans with checked unsigned
64-bit arithmetic over the actual declared clone extent:

- `span = max(offset+length) - min(offset)` over valid spilled rows.
- `sum = sum(length)` over those rows, including repeated references.
- `copy_bytes = min(span,sum)`, with ties choosing one span copy.

The span path preserves duplicate/overlapping sharing and rebases offsets by
subtracting the minimum. The packed path copies each row's bytes and rewrites
its destination offset. Both normalize the single-base descriptor's `buf_idx`
to zero, matching the owned `BoltColumn::utf8_at` base-plus-offset accessor.
The decision stays outside the copy loop. Work is O(rows + copy_bytes), with
O(1) planning scratch, two descriptor passes and one spill allocation.

The guarantee excludes descriptors, bitmap bytes and arena alignment/slack:
`copy_bytes <= span <= old_prefix`, and `copy_bytes <= sum`. A thousand
references sharing a 100-byte string copy 100 spill bytes. This is not an
interval-union algorithm: neither a unique-byte bound nor a constant bound
independent of row count follows for widely separated repeated ranges.

## Contract and refusal

Flat/View cloning preserves scalar/logical metadata and stats for unchanged
rows; View becomes Flat. Source validity is checked before reading descriptor
fields, including arbitrary NULL bytes, and cloned bitmaps are rebased to zero.
Byte-aligned validity offsets retain the memcpy path; only unaligned bit slices
need the per-row rebasing loop.
No valid tail leaves a null overflow base. Ephemeral sidecars and non-Dictionary
advisory dictionary hints are cleared without changing the source.

Constant uses one physical descriptor if any logical row is valid, and owns
its logical bitmap. Dictionary keys never enter the StringView walk; its real
child is recursively cloned and a nonempty child refusal propagates. VarBinary
retains its offsets-child copy and validity rebasing. Other encodings retain
their existing dispatch; this increment is not a general clone-format audit.

Length, bitmap end, descriptor/offset bytes, range addition, sum and size_t /
ptrdiff_t allocation/address bounds are checked before spill copying. Packed
starting offsets must fit uint32; an unrepresentable packed plan fails instead
of selecting a larger span. A final payload can extend beyond 4 GiB if its
starting offset fits. The representability limit includes 64 bytes of arena
alignment headroom. A valid long descriptor with no spill base refuses.
BoltColumn has no spill capacity, so valid source backing ranges remain a
caller precondition; this does not validate arbitrary corrupt descriptors.

Refusal returns the existing empty-column sentinel. Source buffers remain
immutable. Allocations already consumed remain in the arena until its lifetime
ends: there is logical failure atomicity, not allocator rollback. No shared
query arena reset, compaction or cap increase is introduced.

## Validation and lifetime

Registered `test_bolt_primitives` cases cover scattered high offsets, shared
and overlapping tails, thousand-row duplicate sharing, Flat/View validity
across byte boundaries, NULL garbage, all-null/zero-row and Constant payloads,
Dictionary keys/child refusal, VarBinary, missing bases and malformed extents.
Pure arithmetic cases test wrap/width refusal and final tails past 4 GiB
without huge allocations. A real fixed 256-slot arena tests descriptor/key
allocation followed by spill refusal and unchanged source.

Chukonu's recycled-producer fixture independently asserts retained snapshot
allocation bytes, poisons source buffers before replay, and proves a legacy
prefix request can refuse at the unchanged cap while the bounded clone fits.

String snapshots retain their existing query lifetime. Removing unused prefix
copies does not establish a total-memory plateau. Packed per-row memcpy may
change performance; real query parity/performance and platform gates remain
separate evidence. The clone tests do not establish SF10 completion or speedup.
