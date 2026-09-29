// bolt_regex_limits.h — invariants of the backtracking regex engine
// (bolt_regex.h regex_compile / regex_search). Kind A: fixed by the compiled
// pattern layout and the matcher's stack frames, reported, never set.
#pragma once

#include <cstdint>

namespace bolt {
namespace kernels {
namespace regex {

// Largest m / n accepted in {m}, {m,}, {m,n}. RE2 (DuckDB's engine) refuses
// a count above 1000 with "invalid repetition size"; so does this one.
constexpr uint16_t kRegexMaxRepeat = 1000;

// Repetitions of a quantified GROUP the matcher can backtrack over (one
// int32 end offset each, on the stack of that group's frame). A {m,...} group
// whose m exceeds this is refused at compile; a match that needs more is a
// loud exhaustion (MatchResult::exhausted), never a shorter match.
constexpr int32_t kRegexMaxGroupReps = 256;

// Floor of the per-search backtracking step budget (added to the
// per-byte budget), so short inputs with counted repeats are not starved.
constexpr int32_t kRegexMinStepBudget = 1 << 16;

}  // namespace regex
}  // namespace kernels
}  // namespace bolt
