// EventCount: every interleaving of the protocol in a small model, with
// negative controls, then a stress run of the real thing.
#include "bolt/bolt_event_count.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// Model: producers claim, publish and notify on a ring whose pop needs the
// head slot published; a consumer that gets an item never comes back (a long
// task). Memory is sequentially consistent. Property: when nothing can move,
// min(items, consumers) consumers hold an item.
constexpr int kP = 3, kC = 3;
struct Rules { bool recheck = true; bool chain = true; bool announce_first = true; };

enum PStep : uint8_t { PClaim, PPublish, PLook, PBump, PWake, PDone };
enum CStep : uint8_t { CTry, CAnnounce, CKey, CRecheck, CCancel, CWait, CParked, CWoken, CChain, CLook, CBump, CWake, CBusy };

struct State {
    uint8_t p[kP], slot[kP], c[kC];
    uint8_t fresh[kC];  // key still equals the epoch
    uint8_t pub, head, tail, sleepers;

    void bump() { std::fill(fresh, fresh + kC, uint8_t{0}); }

    // Producers and consumers are interchangeable: one representative each.
    std::string canonical() const {
        std::array<uint16_t, kP> a;
        std::array<uint16_t, kC> b;
        for (int i = 0; i < kP; ++i)
            a[i] = static_cast<uint16_t>(p[i] << 8 | (p[i] == 1 ? slot[i] : 0));
        for (int i = 0; i < kC; ++i) b[i] = static_cast<uint16_t>(c[i] << 8 | fresh[i]);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::string k(reinterpret_cast<const char*>(a.data()), sizeof a);
        k.append(reinterpret_cast<const char*>(b.data()), sizeof b);
        for (uint8_t v : {pub, head, tail, sleepers}) k.push_back(static_cast<char>(v));
        return k;
    }
};

struct Model {
    Rules rules;
    int items, consumers;
    std::unordered_set<std::string> seen;
    bool violated = false;
    uint64_t terminals = 0;

    static bool ready(const State& s) { return s.head < kP && (s.pub >> s.head & 1u); }

    void wake(State s, std::vector<State>& out, const State& none) const {
        bool any = false;
        for (int i = 0; i < consumers; ++i) {
            if (s.c[i] != CParked) continue;
            any = true;
            State t = s;
            t.c[i] = CWoken;
            out.push_back(t);
        }
        if (!any) out.push_back(none);
    }

    void producer(const State& s, int i, std::vector<State>& out) const {
        State t = s;
        switch (s.p[i]) {
            case PClaim: t.slot[i] = t.tail++; t.p[i] = PPublish; break;
            case PPublish: t.pub = static_cast<uint8_t>(t.pub | 1u << t.slot[i]); t.p[i] = PLook; break;
            case PLook: t.p[i] = s.sleepers ? PBump : PDone; break;
            case PBump: t.bump(); t.p[i] = PWake; break;
            case PWake: t.p[i] = PDone; wake(t, out, t); return;
            default: return;
        }
        out.push_back(t);
    }

    void consumer(const State& s, int i, std::vector<State>& out) const {
        State t = s;
        switch (s.c[i]) {
            case CTry:
                if (ready(s)) { ++t.head; t.c[i] = CChain; } else t.c[i] = rules.announce_first ? CAnnounce : CKey;
                break;
            case CAnnounce: ++t.sleepers; t.c[i] = rules.announce_first ? CKey : CWait; break;
            case CKey: t.fresh[i] = 1; t.c[i] = rules.recheck ? CRecheck : (rules.announce_first ? CWait : CAnnounce); break;
            case CRecheck:
                if (ready(s)) { ++t.head; t.c[i] = rules.announce_first ? CCancel : CChain; }
                else t.c[i] = rules.announce_first ? CWait : CAnnounce;
                break;
            case CCancel: --t.sleepers; t.c[i] = CChain; break;
            case CWait: t.c[i] = s.fresh[i] ? CParked : CWoken; break;
            case CWoken:
                if (s.fresh[i]) t.c[i] = CParked;  // atomic::wait re-checks
                else { --t.sleepers; t.c[i] = CTry; }
                break;
            case CChain: t.fresh[i] = 0; t.c[i] = rules.chain && s.head < s.tail ? CLook : CBusy; break;
            case CLook: t.c[i] = s.sleepers ? CBump : CBusy; break;
            case CBump: t.bump(); t.c[i] = CWake; break;
            case CWake: t.c[i] = CBusy; wake(t, out, t); return;
            default: return;
        }
        out.push_back(t);
    }

    void run() {
        State s{};
        for (int i = items; i < kP; ++i) s.p[i] = PDone;
        for (int i = consumers; i < kC; ++i) s.c[i] = CBusy;
        std::vector<State> stack{s};
        std::vector<State> next;
        while (!stack.empty()) {
            const State cur = stack.back();
            stack.pop_back();
            if (!seen.insert(cur.canonical()).second) continue;
            next.clear();
            for (int i = 0; i < items; ++i) producer(cur, i, next);
            for (int i = 0; i < consumers; ++i) consumer(cur, i, next);
            if (next.empty()) {
                ++terminals;
                violated |= cur.head != std::min(items, consumers);
            }
            stack.insert(stack.end(), next.begin(), next.end());
        }
    }
};

TEST(EventCountModel, NoInterleavingStrandsAnItemWhileAConsumerSleeps) {
    for (int items = 1; items <= kP; ++items) {
        for (int consumers = 1; consumers <= kC; ++consumers) {
            Model m{{}, items, consumers, {}, false, 0};
            m.run();
            EXPECT_FALSE(m.violated) << items << " items, " << consumers << " consumers";
            EXPECT_GT(m.terminals, 0u);
        }
    }
}

TEST(EventCountModel, EachRuleIsNeeded) {
    Model no_recheck{{.recheck = false}, 1, 1, {}, false, 0};
    no_recheck.run();
    EXPECT_TRUE(no_recheck.violated);
    Model late{{.announce_first = false}, 1, 1, {}, false, 0};
    late.run();
    EXPECT_TRUE(late.violated);
    Model no_chain{{.chain = false}, 2, 2, {}, false, 0};
    no_chain.run();
    EXPECT_TRUE(no_chain.violated);
}

// One item in flight at a time: every push finds the consumers spinning,
// announcing or parked, and a lost wakeup hangs the ping-pong.
TEST(EventCountStress, PingPongNeverLosesAWake) {
    constexpr uint64_t kRounds = 200'000;
    bolt::EventCount to, back;
    alignas(64) std::atomic<uint64_t> a{0};
    alignas(64) std::atomic<uint64_t> b{0};
    std::thread peer([&] {
        for (uint64_t r = 1; r <= kRounds; ++r) {
            to.await(r % 3 ? 0 : 64, [&] { return a.load(std::memory_order_relaxed) >= r; });
            b.store(r, std::memory_order_relaxed);
            back.notify_one();
        }
    });
    for (uint64_t r = 1; r <= kRounds; ++r) {
        a.store(r, std::memory_order_relaxed);
        to.notify_one();
        back.await(r % 5 ? 0 : 64, [&] { return b.load(std::memory_order_relaxed) >= r; });
    }
    peer.join();
    EXPECT_EQ(b.load(), kRounds);
    EXPECT_EQ(to.sleepers(), 0u);
    EXPECT_EQ(back.sleepers(), 0u);
}

TEST(EventCount, NotifyAllReleasesEverySleeper) {
    bolt::EventCount ec;
    std::atomic<bool> go{false};
    std::vector<std::thread> t;
    for (int i = 0; i < 8; ++i)
        t.emplace_back([&] { ec.await(0, [&] { return go.load(std::memory_order_acquire); }); });
    while (ec.sleepers() != 8u) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    ec.notify_all();
    for (auto& x : t) x.join();
    EXPECT_EQ(ec.sleepers(), 0u);
}

}  // namespace
