# Cold current-thread scheduling request

`bolt::thread` exports `bolt/thread/bolt_thread_qos.h` and is included by the
umbrella target. Its public header contains no platform SDK headers. Inherit
returns without an OS call and does not reset any previously explicit setting.
Invalid enum values also return before OS access.

Latency requests macOS USER_INTERACTIVE with relative priority zero, or Windows
THREAD_PRIORITY_HIGHEST. Configuration success requires the setter to succeed
and immediate native readback to match. Other platforms return Unsupported.
Native call failures retain their returned error code; a configuration mismatch
returns Failed with observed values and no invented OS error. A successful setter
followed by failed readback may leave the thread configuration changed.

Caller owns this thread and its policy until explicitly changed or thread exit.
No process policy, realtime scheduler, CPU isolation or system setting changes.
Configured does not promise effective priority, execution placement, scheduling
independence or latency. Tests run in isolated threads and independently read
native configuration; Windows/Linux execution is not established by Mac tests.

Primary contracts: [Apple pthread QoS header](https://github.com/apple-oss-distributions/libpthread/blob/main/include/pthread/qos.h),
[Windows setter](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadpriority),
[Windows getter](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadpriority).

2026-10-17 local verification: canonical Gestalt2 Apple Clang Release build and
both `test_bolt_thread_qos` isolated-thread cases pass. New module compilation
has no warnings; the enclosing Chukonu build retains existing diagnostics.
