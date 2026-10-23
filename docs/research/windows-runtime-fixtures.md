# Windows runtime fixtures and Iceberg diagnostics

## Evidence and scope

The Windows CI runtime evidence in
`/private/tmp/iceberg-bolt-msvc-fourth-job.log` showed fixture setup failures in
file I/O/aligned writer, insufficient ring wrap work for Windows geometry, and
CRT corruption reopen refusal while a mapped owner requested delete sharing.
Four Parquet tests crashed at large automatic metadata owners. Root separately
observed those unchanged suites passing with normal Linux stack and overflowing
with a 1 MiB stack. This establishes stack sensitivity; Windows exception/fixed
runtime qualification remains root's execution responsibility.

Concurrent commit and retention-after-reopen failures remain unclassified. Their
test changes are observational. No production I/O or conflict semantics change.

## Temporary file ownership and setup failure

`tests/test_temp_owner.h` owns a freshly created OS temporary directory. It uses
the checked error-code filesystem temp-directory lookup, process ID, monotonic
tick, atomic sequence, and at most 64 exclusive directory-creation attempts.
It never falls back to `/tmp` or the working directory. Only a directory it
created is eligible for recursive cleanup. The two file suites fatally check
ownership before using a path. External benchmark inputs do not create an
unused temporary owner.

File creation, random-page generation, file readback and throughput helpers
return checked success before callers consume state. Partial writer failure
aborts the actual writer; stdio helpers close their files before returning.
Invalid-directory controls require prompt refusal without publishing generated
page/size outputs. Direct/buffered modes, boundary/content checks and performance
floors/samples remain. The payload replay consumes the same RNG unit draw.

The CRC corruption test builds a finite index of nonempty generated pages, then
performs exactly 10,000 actual reversible byte flips and CRC comparisons. No
failed-open/empty-page decrement-and-retry loop remains. Empty/padded pages still
receive their existing layout and padding validation elsewhere in the suite.

## Mapped-ring controls

Wrap work is derived from capacity and the minimum 128-byte packed nonempty
record geometry, with at least the original 200 records. All sequence, byte
pattern, consumption and committed-position-greater-than-capacity assertions
remain. Capacity and production mapping geometry do not change.

Windows corruption injection uses a separate checked CreateFileW handle with
READ|WRITE|DELETE sharing, matching the existing production owner's sharing.
SetFilePointerEx, WriteFile, byte count and CloseHandle are checked; native errors
are sampled immediately at the failed OS operation, before assertions/cleanup
can overwrite them. POSIX stdio mutation remains supported. Caller-level fatal
checks prevent an injection failure from masquerading as a Corrupt result.
The exact Corrupt, pin and successful-close discriminators remain.

## Parquet fixture backing

All large PqMeta locals in the four assigned suites now have checked caller-owned
Arena backing and in-place zero initialization. Reference names preserve the
existing parser/value assertions. Page-index ParsedFile owns its metadata arena
and a pointer, rather than a large automatic PqMeta aggregate. Callers propagate
fatal parse/index/readback helper failures before consuming output.

Malformed-list metadata scratch is an arena per repeated-column sweep, reset and
reused before every truncation/flip/saturation attempt. Outer metadata has a
different owner and remains valid. Corpus inputs, loop geometry, repeated-column,
attempt/decode counters and expected values remain. No public metadata layout,
global stack setting, shared static metadata or reduced corpus is introduced.

## Async-I/O allocation fixture

The existing scalar ordinary new/delete replacement uses counted malloc/free.
Observed GTest nothrow scalar allocation previously entered the sanitizer's new
family and later reached the fixture's free-based ordinary delete. Matching
nothrow new and nothrow deallocation now use the same counted malloc/free
family; failed nothrow allocation returns null. A direct control covers the
observed nothrow-new/ordinary-delete path and matching nothrow deallocation.
No unrelated aligned/array family replacement is added without evidence.
The original steady-state zero-allocation assertion remains unchanged.

Root's clean lifetime lane is explicitly ASAN/LSAN-only. It is separate from
alignment/division UBSAN and the retained failed mixed-RTTI vptr pilot. No
sanitizer suppression or alloc_dealloc_mismatch disable is part of this repair.

## Iceberg observations, not repair

Each assigned test has a local filesystem ObjectStore vtable wrapper preserving
every original return value, optional capability and classification. It records
the last 12 operations plus the first unexpected non-NotFound/non-Exists result:
operation, bounded object key, returned status, phase and native error sampled
immediately when the trait returns. errno/native state is restored after tracing.
No object bodies or credentials are logged.

The sample is labelled `post_return_errno`/`post_return_native`: it is not proof
of an internal syscall error. In particular, publish_no_replace captures a
MoveFileExW error and removes staging before returning. A fixture wrapper cannot
recover a value overwritten during that cleanup. Further production diagnostics
would need separate authorization and actual failure evidence.

Concurrent attempts report writer, c/round, retry attempt, open/append_open/write/
commit phase and CommitError before safe handle cleanup. Only genuine kConflict
continues the original retry loop. At-most-one winner, successful rows, snapshot
counts and all original attempt limits remain.

Retention diagnostics compare before- and after-reopen append stages, actual
current snapshot/parent and retained count, current parent manifest-list key,
and an additional HEAD observation immediately before each retention-case
append. Retention DELETE results are logged with their actual operation phase.
`observed_metadata_version` comes from actual successful numeric metadata-key
GET/conditional-PUT calls; `opened_base_version` captures that observation after
open. They do not expose or claim an opaque private TableHandle field. Missing
observations remain -1. The final exact 80-row assertion is unchanged.

Diagnostic I/O/logging may perturb race timing. A clean diagnostic run alone does
not select a production repair or prove an earlier failure benign.

## Limits classification

`api/core/accept_retirement_limits.h` centralizes the existing Kind B allocation
bounds: 4096u retirement slots and 65536 readiness descriptors, with static
assertions. Existing public constants remain aliases with the same uint32_t/int
types, values and record layout. Only the two obsolete epoll/kqueue kMaxFds
literal-debt baseline entries are removed. No new debt, lint bypass or accept
behavior change is introduced.

## Qualification still required

Root owns actual Windows runs of all assigned suites, limited-stack Parquet
controls, labelled ASAN/LSAN allocation checks, existing sanitizer gates and fresh
limits lint. Neither source inspection nor compile success qualifies Windows.
Iceberg failures require the next diagnostic Windows trace before any production
fix is chosen. Full Iceberg offline installation/reopen/source transport remains
outside this fixture batch.
