// crash.cpp — write a backtrace and the log tail on a fatal signal, then die by
// that signal, so core dumps and exit status stay what they were.
#include "bolt/obs/crash.h"

#if BOLT_OBS_CRASH && !defined(_WIN32)

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>

#include "bolt/obs/context.h"
#include "bolt/obs/log.h"
#include "bolt/obs/obs_limits.h"

namespace bolt::crash {

namespace {

constexpr std::size_t kAltStackBytes = 64u * 1024u;
constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT};

char g_dir[1024];
char g_identity[256];
volatile std::sig_atomic_t g_in_crash = 0;

void put(int fd, const char* s) noexcept {
    if (fd >= 0) (void)!::write(fd, s, std::strlen(s));
}

void report(int fd, int sig, siginfo_t* info, void* pcs[], int frames) noexcept {
    char b[512];
    std::snprintf(b, sizeof(b), "fatal signal %d (%s) addr=%p pid=%d time=%lld\n%s\n", sig, strsignal(sig),
                  info != nullptr ? info->si_addr : nullptr, static_cast<int>(::getpid()),
                  static_cast<long long>(std::time(nullptr)), g_identity);
    put(fd, b);
    put(2, b);
#if BOLT_OBS_LOG || BOLT_OBS_TRACE
    const obs::TraceCtx& c = obs::trace_ctx();
    if (c.lo != 0) {
        std::snprintf(b, sizeof(b), "trace_id=%016llx%016llx\n", static_cast<unsigned long long>(c.hi), static_cast<unsigned long long>(c.lo));
        put(fd, b);
    }
#endif
    put(fd, "--- backtrace\n");
    if (fd >= 0) backtrace_symbols_fd(pcs, frames, fd);
    backtrace_symbols_fd(pcs, frames, 2);
    put(fd, "--- log tail\n");
}

extern "C" void bolt_crash_handler(int sig, siginfo_t* info, void*) {
    if (g_in_crash != 0) { std::signal(sig, SIG_DFL); ::raise(sig); return; }
    g_in_crash = 1;
    ::alarm(10);   // a wedged report must not outlive the process it describes
    void* pcs[kMaxFrames];
    const int frames = backtrace(pcs, kMaxFrames);
    int fd = -1;
    char path[1200];
    if (g_dir[0] != '\0') {
        std::snprintf(path, sizeof(path), "%s/crash-%lld-%d.txt", g_dir, static_cast<long long>(std::time(nullptr)), static_cast<int>(::getpid()));
        fd = ::open(path, O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0640);
    }
    report(fd, sig, info, pcs, frames);
#if BOLT_OBS_LOG
    bolt::log::crash_flush(fd);
#endif
    if (fd >= 0) ::close(fd);
    std::signal(sig, SIG_DFL);
    ::raise(sig);
}

}  // namespace

void thread_init() noexcept {
    if (g_identity[0] == '\0') return;   // not installed
    stack_t st{};
    st.ss_sp = std::malloc(kAltStackBytes);
    if (st.ss_sp == nullptr) return;
    st.ss_size = kAltStackBytes;
    if (::sigaltstack(&st, nullptr) != 0) std::free(st.ss_sp);
}

void thread_fini() noexcept {
    stack_t st{};
    if (::sigaltstack(nullptr, &st) != 0 || st.ss_sp == nullptr || (st.ss_flags & SS_DISABLE) != 0) return;
    void* mem = st.ss_sp;
    st.ss_flags = SS_DISABLE;
    if (::sigaltstack(&st, nullptr) == 0) std::free(mem);
}

bool install(const Config& c) noexcept {
    assert(c.identity != nullptr);
    std::snprintf(g_dir, sizeof(g_dir), "%s", c.dir != nullptr ? c.dir : "");
    std::snprintf(g_identity, sizeof(g_identity), "%s", c.identity);
    void* warm[4];
    (void)backtrace(warm, 4);   // loads the unwinder now, not inside the handler
    thread_init();
    struct sigaction sa{};
    sa.sa_sigaction = bolt_crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    bool ok = true;
    for (int sig : kSignals) ok = ::sigaction(sig, &sa, nullptr) == 0 && ok;
    return ok;
}

}  // namespace bolt::crash

#elif BOLT_OBS_CRASH

namespace bolt::crash {
bool install(const Config&) noexcept { return false; }
void thread_init() noexcept {}
void thread_fini() noexcept {}
}  // namespace bolt::crash

#endif
