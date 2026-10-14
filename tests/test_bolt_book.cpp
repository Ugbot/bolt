#include "bolt/kernels/fintech/book_l2.h"
#include "bolt/kernels/fintech/book_mbp.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); } } while (false)
namespace {
using namespace bolt::fintech;
BookDefinition definition() { return {7, 11, 13, 17, 1000000000, 1000000, BookLiquidity::Implied}; }
bool equal(BookLevel a, BookLevel b) {
    return a.price_ticks == b.price_ticks && a.quantity == b.quantity &&
        a.order_count == b.order_count && a.order_count_valid == b.order_count_valid;
}
struct L2Fixture {
    BookL2 book{};
    std::array<unsigned char, 65536> storage{};
    explicit L2Fixture(BookLimits limits = {64, 32, 128}) {
        storage.front() = storage.back() = 0xa5;
        CHECK(book_l2_init(&book, definition(), limits, storage.data() + 1, storage.size() - 2) == BookStatus::Ok);
        CHECK(book_l2_begin(book, {55, 1, 0}, BookEventKind::Replacement) == BookStatus::Ok);
        CHECK(book_l2_commit(book) == BookStatus::Ok);
    }
    void canary() const { CHECK(storage.front() == 0xa5 && storage.back() == 0xa5); }
};
struct MBPFixture {
    BookMBP book{};
    std::array<unsigned char, 65536> storage{};
    explicit MBPFixture(BookLimits limits = {64, 32, 128}) {
        storage.front() = storage.back() = 0xa5;
        CHECK(book_mbp_init(&book, definition(), limits, storage.data() + 1, storage.size() - 2) == BookStatus::Ok);
        CHECK(book_mbp_begin(book, {55, 1, 0}, BookEventKind::Replacement) == BookStatus::Ok);
        CHECK(book_mbp_commit(book) == BookStatus::Ok);
    }
    void canary() const { CHECK(storage.front() == 0xa5 && storage.back() == 0xa5); }
};
using L2Oracle = std::array<std::map<std::int64_t, BookLevel>, 2>;
using MBPOracle = std::array<std::vector<BookLevel>, 2>;
void check_l2(const BookL2& book, const L2Oracle& oracle) {
    for (unsigned side = 0; side < 2; ++side) {
        BookLevel levels[64]{}, top{}; std::uint32_t count = 99;
        CHECK(book_l2_depth(book, BookSide(side), 0, levels, 64, count) == BookStatus::Ok);
        CHECK(count == oracle[side].size());
        std::vector<BookLevel> expected;
        for (const auto& [price, level] : oracle[side]) { CHECK(price == level.price_ticks); expected.push_back(level); }
        if (side == 0) std::reverse(expected.begin(), expected.end());
        std::uint64_t quantity = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            CHECK(equal(levels[i], expected[i])); quantity += expected[i].quantity;
            BookLevel looked{};
            CHECK(book_l2_lookup(book, BookSide(side), expected[i].price_ticks, looked) == BookStatus::Ok);
            CHECK(equal(looked, expected[i]));
        }
        CHECK(book_l2_info(book).quantity[side] == quantity);
        CHECK(book_l2_top(book, BookSide(side), top) == (count ? BookStatus::Ok : BookStatus::NotFound));
        if (count) CHECK(equal(top, expected[0]));
        CHECK(book_l2_depth(book, BookSide(side), 1, levels, 3, count) == BookStatus::Ok);
        CHECK(count == std::min<std::size_t>(3, expected.empty() ? 0 : expected.size() - 1));
        for (std::uint32_t i = 0; i < count; ++i) CHECK(equal(levels[i], expected[i + 1]));
    }
}
void check_mbp(const BookMBP& book, const MBPOracle& oracle) {
    for (unsigned side = 0; side < 2; ++side) {
        BookLevel levels[64]{}; std::uint32_t count = 99;
        CHECK(book_mbp_depth(book, BookSide(side), 0, levels, 64, count) == BookStatus::Ok);
        CHECK(count == oracle[side].size());
        std::uint64_t quantity = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            CHECK(equal(levels[i], oracle[side][i])); quantity += levels[i].quantity;
            BookLevel looked{};
            CHECK(book_mbp_lookup(book, BookSide(side), i + 1, looked) == BookStatus::Ok);
            CHECK(equal(looked, levels[i]));
        }
        CHECK(book_mbp_info(book).quantity[side] == quantity);
    }
}
std::uint64_t random(std::uint64_t& state) { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state; }
void test_l2_oracle() {
    L2Fixture f; L2Oracle oracle; std::uint64_t rng = 23;
    for (std::uint64_t event = 1; event <= 2000; ++event) {
        const bool snapshot = event % 47 == 0, late_bad = event % 13 == 0;
        auto candidate = snapshot ? L2Oracle{} : oracle;
        CHECK(book_l2_begin(f.book, {55, 1, event}, snapshot ? BookEventKind::Replacement : BookEventKind::Delta) == BookStatus::Ok);
        bool failed = false;
        for (unsigned chunk = 0; chunk < 5; ++chunk) {
            const unsigned side = unsigned(random(rng) % 2);
            const auto price = std::int64_t(random(rng) % 81) - 40;
            const bool remove = random(rng) % 3 == 0;
            BookLevel level{price, remove ? 0 : random(rng) % 100 + 1, 0, 0};
            const auto status = book_l2_stage(f.book, {BookSide(side), remove ? BookL2Action::Delete : BookL2Action::Upsert, level});
            check_l2(f.book, oracle); // The complete committed book remains visible between chunks.
            if (remove && !candidate[side].contains(price)) { CHECK(status == BookStatus::NotFound); failed = true; break; }
            CHECK(status == BookStatus::Ok);
            if (remove) candidate[side].erase(price); else candidate[side][price] = level;
        }
        if (!failed && late_bad) {
            CHECK(book_l2_stage(f.book, {BookSide::Ask, BookL2Action(255), {}}) == BookStatus::Invalid); failed = true;
        }
        if (failed) { CHECK(book_l2_commit(f.book) != BookStatus::Ok); book_l2_abort(f.book); }
        else { CHECK(book_l2_commit(f.book) == BookStatus::Ok); oracle = candidate; }
        check_l2(f.book, oracle); f.canary();
    }
}
void test_mbp_oracle() {
    MBPFixture f; MBPOracle oracle; std::uint64_t rng = 97;
    for (std::uint64_t event = 1; event <= 1500; ++event) {
        const bool snapshot = event % 29 == 0, late_bad = event % 11 == 0;
        auto candidate = snapshot ? MBPOracle{} : oracle;
        CHECK(book_mbp_begin(f.book, {55, 1, event}, snapshot ? BookEventKind::Replacement : BookEventKind::Delta) == BookStatus::Ok);
        for (unsigned chunk = 0; chunk < 4; ++chunk) {
            const unsigned side = unsigned(random(rng) % 2);
            auto& levels = candidate[side];
            const auto action = levels.empty() ? BookMBPAction::Insert : BookMBPAction(random(rng) % 3);
            const auto index = std::uint32_t(random(rng) % (levels.size() + unsigned(action == BookMBPAction::Insert)));
            BookLevel level{std::int64_t(random(rng) % 100) - 50, random(rng) % 100 + 1, random(rng) % 20, 1};
            if (action == BookMBPAction::Delete) level = {};
            CHECK(book_mbp_stage(f.book, {BookSide(side), action, index + 1, level}) == BookStatus::Ok);
            check_mbp(f.book, oracle);
            if (action == BookMBPAction::Insert) levels.insert(levels.begin() + index, level);
            else if (action == BookMBPAction::Delete) levels.erase(levels.begin() + index);
            else levels[index] = level;
        }
        if (late_bad) {
            CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Change, 0, {1, 1, 0, 0}}) == BookStatus::Invalid);
            CHECK(book_mbp_commit(f.book) == BookStatus::Invalid); book_mbp_abort(f.book);
        } else { CHECK(book_mbp_commit(f.book) == BookStatus::Ok); oracle = candidate; }
        check_mbp(f.book, oracle); f.canary();
    }
}
void test_l2_slot_reuse_rotation_abort() {
    L2Fixture f; L2Oracle old;
    CHECK(book_l2_begin(f.book, {55, 1, 1}, BookEventKind::Delta) == BookStatus::Ok);
    for (const auto price : {30, 20, 10, 40, 50, 25, 15}) {
        BookLevel level{price, 5, 7, 1}; old[0][price] = level;
        CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, level}) == BookStatus::Ok);
    }
    CHECK(book_l2_commit(f.book) == BookStatus::Ok); check_l2(f.book, old);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    const auto id = book_detail::find(f.book, 0, 20, false);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Delete, {20, 0, 0, 0}}) == BookStatus::Ok);
    const auto freed = f.book.working.free_head;
    CHECK(book_l2_stage(f.book, {BookSide::Ask, BookL2Action::Upsert, {-5, 9, 0, 0}}) == BookStatus::Ok);
    CHECK(book_detail::find(f.book, 1, -5, false) == freed);
    CHECK(id != kBookNull); check_l2(f.book, old);
    for (const auto price : {60, 70, 80}) {
        CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {price, 1, 0, 0}}) == BookStatus::Ok);
        check_l2(f.book, old);
    }
    CHECK(book_l2_stage(f.book, {BookSide(9), BookL2Action::Clear, {}}) == BookStatus::Invalid);
    CHECK(book_l2_commit(f.book) == BookStatus::Invalid); book_l2_abort(f.book); check_l2(f.book, old);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Clear, {}}) == BookStatus::Ok);
    CHECK(book_l2_commit(f.book) == BookStatus::Ok); old[0].clear(); check_l2(f.book, old);
}
void test_replacements_and_identity() {
    L2Fixture f; BookLevel level{};
    CHECK(book_l2_begin(f.book, {55, 1, 1}, BookEventKind::Delta) == BookStatus::Ok);
    for (const auto price : {INT64_MIN, INT64_MAX})
        CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {price, 1, UINT64_MAX, 1}}) == BookStatus::Ok);
    CHECK(book_l2_commit(f.book) == BookStatus::Ok);
    CHECK(book_l2_top(f.book, BookSide::Bid, level) == BookStatus::Ok && level.price_ticks == INT64_MAX);
    CHECK(book_l2_begin(f.book, {UINT64_MAX, 1, 2}, BookEventKind::Replacement) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {1, 2, 0}, BookEventKind::Delta) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {1, 2, 0}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Ask, BookL2Action::Upsert, {-1, 2, 0, 0}}) == BookStatus::Ok);
    CHECK(book_l2_commit(f.book) == BookStatus::Ok);
    CHECK(book_l2_lookup(f.book, BookSide::Bid, INT64_MIN, level) == BookStatus::NotFound);
    CHECK(book_l2_info(f.book).cut.epoch == 1 && book_l2_info(f.book).cut.generation == 2);
    CHECK(book_l2_begin(f.book, {1, 2, 0}, BookEventKind::Replacement) == BookStatus::Invalid);
    CHECK(book_l2_invalidate(f.book) == BookStatus::Ok);
    CHECK(book_l2_top(f.book, BookSide::Ask, level) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {1, 2, 1}, BookEventKind::Delta) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {1, 2, 1}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_l2_commit(f.book) == BookStatus::Ok);
    CHECK(book_l2_info(f.book).valid && book_l2_info(f.book).levels[1] == 0);
}
void test_limits_overflow_and_journal() {
    L2Fixture f({2, 2, 1}); BookLevel out{};
    CHECK(book_l2_begin(f.book, {55, 1, 1}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {-7, UINT64_MAX, 0, 0}}) == BookStatus::Ok);
    CHECK(book_l2_commit(f.book) == BookStatus::Ok);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {-7, 10, 0, 0}}) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {-7, 11, 0, 0}}) == BookStatus::Ok);
    CHECK(f.book.touched == 1);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {-7, 12, 0, 0}}) == BookStatus::Capacity);
    CHECK(book_l2_commit(f.book) == BookStatus::Capacity); book_l2_abort(f.book);
    CHECK(book_l2_lookup(f.book, BookSide::Bid, -7, out) == BookStatus::Ok && out.quantity == UINT64_MAX);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {0, 1, 0, 0}}) == BookStatus::Overflow);
    book_l2_abort(f.book);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Ask, BookL2Action::Upsert, {8, 1, 0, 0}}) == BookStatus::Ok);
    CHECK(book_l2_stage(f.book, {BookSide::Ask, BookL2Action::Upsert, {9, 1, 0, 0}}) == BookStatus::Capacity);
    CHECK(book_l2_commit(f.book) == BookStatus::Capacity); book_l2_abort(f.book);
    CHECK(book_l2_info(f.book).cut.position == 1 && book_l2_info(f.book).levels[1] == 0);
    f.book.committed.read.revision = UINT64_MAX;
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Overflow); f.canary();
}
void test_mbp_failures() {
    MBPFixture f({2, 8, 4}); BookLevel out{};
    CHECK(book_mbp_begin(f.book, {55, 1, 1}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Insert, 1, {INT64_MIN, 1, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Insert, 2, {INT64_MAX, 1, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Delete, 1, {}}) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Insert, 1, {-9, UINT64_MAX, 0, 0}}) == BookStatus::Overflow);
    CHECK(book_mbp_commit(f.book) == BookStatus::Overflow); book_mbp_abort(f.book);
    CHECK(book_mbp_lookup(f.book, BookSide::Ask, 1, out) == BookStatus::Ok && out.price_ticks == INT64_MIN);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Insert, 1, {-8, 1, 0, 0}}) == BookStatus::Capacity);
    book_mbp_abort(f.book);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Insert, 1, {-10, 3, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_top(f.book, BookSide::Ask, out) == BookStatus::NotFound);
    CHECK(book_mbp_begin(f.book, {55, 1, 3}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Clear, 0, {}}) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_info(f.book).levels[0] == 0); f.canary();
    CHECK(book_mbp_begin(f.book, {55, 1, 4}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_info(f.book).levels[0] == 0 && book_mbp_info(f.book).levels[1] == 0);
}
void test_mbp_small_journal_and_operations() {
    MBPFixture f({3, 2, 1}); BookLevel out{};
    CHECK(book_mbp_begin(f.book, {55, 1, 1}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Insert, 1, {10, 1, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Insert, 2, {9, 1, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Change, 1, {10, 2, 0, 0}}) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Change, 1, {10, 3, 0, 0}}) == BookStatus::Ok);
    CHECK(f.book.touched == 1);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Change, 1, {10, 4, 0, 0}}) == BookStatus::Capacity);
    CHECK(book_mbp_commit(f.book) == BookStatus::Capacity); book_mbp_abort(f.book);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Bid, BookMBPAction::Insert, 1, {11, 1, 0, 0}}) == BookStatus::Capacity);
    book_mbp_abort(f.book);
    CHECK(book_mbp_lookup(f.book, BookSide::Bid, 1, out) == BookStatus::Ok && out.quantity == 1 && out.price_ticks == 10);
    CHECK(book_mbp_lookup(f.book, BookSide::Bid, 2, out) == BookStatus::Ok && out.price_ticks == 9);
    CHECK(book_mbp_begin(f.book, {55, 1, 2}, BookEventKind::Replacement) == BookStatus::Ok);
    CHECK(book_mbp_stage(f.book, {BookSide::Ask, BookMBPAction::Insert, 1, {0, UINT64_MAX, UINT64_MAX, 1}}) == BookStatus::Ok);
    CHECK(book_mbp_commit(f.book) == BookStatus::Ok);
    CHECK(book_mbp_info(f.book).quantity[1] == UINT64_MAX); f.canary();
}
void test_storage_and_read_bounds() {
    std::uint64_t bytes = 17;
    CHECK(book_l2_storage_size({0, 1, 1}, bytes) == BookStatus::Invalid && bytes == 0);
    CHECK(book_mbp_storage_size({UINT32_MAX, 1, 1}, bytes) == BookStatus::Invalid && bytes == 0);
    CHECK(book_l2_storage_size({kBookMaxLevels, kBookMaxEventOperations, 2 * kBookMaxLevels}, bytes) == BookStatus::Ok);
    CHECK(bytes > 2 * std::uint64_t(kBookMaxLevels) * sizeof(book_detail::L2Node));
    BookL2 raw{}; unsigned char tiny[16]{};
    CHECK(book_l2_init(&raw, definition(), {2, 2, 4}, tiny, sizeof(tiny)) == BookStatus::Capacity && !raw.initialized);
    auto bad_def = definition(); bad_def.price_scale = 0;
    CHECK(book_l2_init(&raw, bad_def, {2, 2, 4}, tiny, sizeof(tiny)) == BookStatus::Invalid);
    L2Fixture f; BookLevel out[2]{}; std::uint32_t count = 7;
    CHECK(book_l2_depth(f.book, BookSide::Ask, 0, out, UINT32_MAX, count) == BookStatus::Invalid && count == 0);
    CHECK(book_l2_depth(f.book, BookSide::Ask, 0, nullptr, 1, count) == BookStatus::Invalid);
    CHECK(book_l2_depth(f.book, BookSide::Ask, 0, nullptr, 0, count) == BookStatus::Ok && count == 0);
    CHECK(book_l2_lookup(f.book, BookSide::Bid, 1, f.book.bank[0][0].level) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {0, 2, 1}, BookEventKind::Replacement) == BookStatus::Invalid);
    CHECK(book_l2_begin(f.book, {55, 1, 1}, BookEventKind::Delta) == BookStatus::Ok);
    CHECK(book_l2_begin(f.book, {55, 1, 2}, BookEventKind::Delta) == BookStatus::Busy);
    CHECK(book_l2_invalidate(f.book) == BookStatus::Busy);
    CHECK(book_l2_stage(f.book, {BookSide::Bid, BookL2Action::Upsert, {1, 1, 9, 0}}) == BookStatus::Invalid);
    book_l2_abort(f.book); f.canary();
}
} // namespace
int main() {
    test_storage_and_read_bounds(); test_replacements_and_identity();
    test_limits_overflow_and_journal(); test_l2_slot_reuse_rotation_abort();
    test_mbp_failures(); test_mbp_small_journal_and_operations(); test_l2_oracle(); test_mbp_oracle();
    std::puts("bounded L2/MBP book oracle checks passed");
}
