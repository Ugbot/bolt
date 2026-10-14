#include "bolt/ingest/bolt_zstd_stream.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>
#include <utility>
namespace { std::atomic<std::uint64_t> heap_allocations{0}; }
void* operator new(std::size_t n) {
    heap_allocations.fetch_add(1,std::memory_order_relaxed);
    auto* p=std::malloc(n==0?1:n); if (p==nullptr) std::abort(); return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p,std::size_t) noexcept { ::operator delete(p); }
namespace {
using namespace bolt::ingest;
using Status=ZstdStreamStatus;
std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> v;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<size_t>(n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) v.clear();
    return v;
}

std::string data_path(const char* name) {
    return std::string(BOLT_TEST_DATA_DIR) + "/" + name;
}

// --- the deterministic patterns, mirroring make_zstd_vectors.py -------------
enum : uint32_t {
    kInline = 0, kZeros = 1, kConst = 2, kRepeat = 3,
    kLcg = 4, kAlpha = 5, kChunks = 6,
};
constexpr uint64_t kMul = 6364136223846793005ull;
constexpr uint64_t kInc = 1442695040888963407ull;

void lcg_fill(uint8_t* dst, uint64_t n, uint64_t seed) {
    uint64_t s = seed;
    for (uint64_t i = 0; i < n; ++i) {
        s = s * kMul + kInc;
        dst[i] = static_cast<uint8_t>((s >> 33) & 0xFFu);
    }
}

struct Vector {
    std::string          name;
    uint32_t             kind = 0;
    uint64_t             raw_len = 0;
    uint64_t             seed = 0;
    uint32_t             aux = 0;
    std::vector<uint8_t> pat;
    std::vector<uint8_t> inl;
    std::vector<uint8_t> comp;
};

std::vector<uint8_t> materialize(const Vector& v) {
    std::vector<uint8_t> r;
    switch (v.kind) {
        case kInline: return v.inl;
        case kZeros:  r.assign(static_cast<size_t>(v.raw_len), 0); return r;
        case kConst:
            r.assign(static_cast<size_t>(v.raw_len),
                     static_cast<uint8_t>(v.seed & 0xFFu));
            return r;
        case kRepeat: {
            r.resize(static_cast<size_t>(v.raw_len));
            if (v.pat.empty()) return r;
            for (size_t i = 0; i < r.size(); ++i) r[i] = v.pat[i % v.pat.size()];
            return r;
        }
        case kLcg:
            r.resize(static_cast<size_t>(v.raw_len));
            if (!r.empty()) lcg_fill(r.data(), r.size(), v.seed);
            return r;
        case kAlpha: {
            r.resize(static_cast<size_t>(v.raw_len));
            uint64_t s = v.seed;
            for (size_t i = 0; i < r.size(); ++i) {
                s = s * kMul + kInc;
                r[i] = static_cast<uint8_t>('a' + ((s >> 33) % v.aux));
            }
            return r;
        }
        case kChunks: {
            uint32_t clen = 0, reps = 0;
            std::memcpy(&clen, v.pat.data(), 4);
            std::memcpy(&reps, v.pat.data() + 4, 4);
            const uint8_t lit = v.pat[8];
            const uint32_t k = v.aux;
            std::vector<std::vector<uint8_t>> chunks(k);
            for (uint32_t i = 0; i < k; ++i) {
                chunks[i].resize(clen);
                lcg_fill(chunks[i].data(), clen, v.seed + 1u + i);
            }
            r.reserve(static_cast<size_t>(k) * clen +
                      static_cast<size_t>(reps) * (clen + 1u));
            for (uint32_t i = 0; i < k; ++i)
                r.insert(r.end(), chunks[i].begin(), chunks[i].end());
            uint64_t s = v.seed;
            for (uint32_t i = 0; i < reps; ++i) {
                s = s * kMul + kInc;
                r.push_back(lit);
                const auto& c = chunks[(s >> 33) % k];
                r.insert(r.end(), c.begin(), c.end());
            }
            return r;
        }
        default: return r;
    }
}

std::vector<Vector> load_vectors() {
    std::vector<Vector> out;
    std::vector<uint8_t> b = slurp(data_path("zstd_vectors.bin"));
    if (b.size() < 4) return out;
    size_t o = 0;
    auto need = [&](size_t k) { return o + k <= b.size(); };
    auto r32 = [&]() { uint32_t v; std::memcpy(&v, b.data() + o, 4); o += 4; return v; };
    auto r64 = [&]() { uint64_t v; std::memcpy(&v, b.data() + o, 8); o += 8; return v; };
    const uint32_t n = r32();
    for (uint32_t i = 0; i < n && need(4); ++i) {
        Vector v;
        const uint32_t nl = r32();
        if (!need(nl)) break;
        v.name.assign(reinterpret_cast<const char*>(b.data() + o), nl); o += nl;
        v.kind = r32();
        v.raw_len = r64();
        v.seed = r64();
        v.aux = r32();
        const uint32_t pl = r32();
        if (!need(pl)) break;
        v.pat.assign(b.data() + o, b.data() + o + pl); o += pl;
        const uint64_t il = r64();
        if (!need(static_cast<size_t>(il))) break;
        v.inl.assign(b.data() + o, b.data() + o + static_cast<size_t>(il));
        o += static_cast<size_t>(il);
        const uint64_t cl = r64();
        if (!need(static_cast<size_t>(cl))) break;
        v.comp.assign(b.data() + o, b.data() + o + static_cast<size_t>(cl));
        o += static_cast<size_t>(cl);
        out.push_back(std::move(v));
    }
    return out;
}

struct Work {
    std::vector<std::uint64_t> words;
    std::vector<std::uint8_t> history;
    ZstdStream* stream=nullptr;
    explicit Work(std::uint64_t window=8u<<20,std::uint64_t limit=UINT64_MAX)
        : words(static_cast<std::size_t>((zstd_stream_state_size()+7)/8+2),0xfeedfacebeefcafeull),
          history(static_cast<std::size_t>(window)+16,0xcd) {
        EXPECT_EQ(zstd_stream_init(words.data()+1,zstd_stream_state_size(),history.data()+8,window,
            {window,limit},&stream),Status::Progress);
    }
    void canaries() const {
        EXPECT_EQ(words.front(),0xfeedfacebeefcafeull); EXPECT_EQ(words.back(),0xfeedfacebeefcafeull);
        for (auto i=0u;i<8;++i) { EXPECT_EQ(history[i],0xcd); EXPECT_EQ(history[history.size()-1-i],0xcd); }
    }
};
struct Run {
    Status status=Status::Progress; std::vector<std::uint8_t> output;
    std::uint64_t consumed=0,frames=0; ZstdStep last{};
};
Run decode(Work& work,const std::vector<std::uint8_t>& bytes,std::uint32_t input_chunk,
    std::uint32_t output_chunk,bool stall=false) {
    Run run; std::array<std::uint8_t,kZstdStreamBlockBytes> output{};
    for (auto step=0u;step<(1u<<24);++step) {
        const auto n=static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes.size()-run.consumed,input_chunk));
        const bool final=run.consumed+n==bytes.size();
        const auto capacity=stall && step%7==0?0:output_chunk;
        const auto r=zstd_stream_step(work.stream,n==0?nullptr:bytes.data()+run.consumed,n,final,output.data(),capacity);
        EXPECT_LE(r.input_consumed,n); EXPECT_LE(r.output_produced,capacity);
        EXPECT_LE(r.input_consumed,kZstdStreamBlockBytes); EXPECT_LE(r.output_produced,kZstdStreamBlockBytes);
        run.consumed+=r.input_consumed; run.output.insert(run.output.end(),output.begin(),output.begin()+static_cast<std::ptrdiff_t>(r.output_produced));
        run.status=r.status; run.last=r;
        if (r.status==Status::FrameEnd) ++run.frames;
        if (r.status==Status::Finished || r.status>=Status::InvalidArgument) return run;
    }
    ADD_FAILURE()<<"bounded decoder harness exhausted"; return run;
}
void append_le(std::vector<std::uint8_t>& out,std::uint64_t value,std::uint32_t n) {
    for (auto i=0u;i<n;++i) out.push_back(static_cast<std::uint8_t>(value>>(8*i)));
}
std::vector<std::uint8_t> window_header() { return {0x28,0xb5,0x2f,0xfd,0,0}; }
void block(std::vector<std::uint8_t>& frame,std::uint32_t type,const std::vector<std::uint8_t>& bytes,
    bool last,std::uint32_t regenerated=0) {
    const auto size=type==1?regenerated:static_cast<std::uint32_t>(bytes.size());
    append_le(frame,(size<<3)|(type<<1)|std::uint32_t(last),3);
    frame.insert(frame.end(),bytes.begin(),bytes.end());
}
std::vector<std::uint8_t> raw_frame(const std::vector<std::uint8_t>& bytes) {
    auto frame=window_header(); block(frame,0,bytes,true); return frame;
}
const Vector* find_vector(const std::vector<Vector>& vectors,const char* name) {
    for (const auto& v:vectors) if (v.name==name) return &v;
    return nullptr;
}
TEST(ZstdStream, ExistingIndependentReferenceCorpusByteExactWithFragmentationAndBackpressure) {
    const auto vectors=load_vectors(); ASSERT_GE(vectors.size(),30u);
    for (const auto& vector:vectors) {
        SCOPED_TRACE(vector.name); Work work;
        const auto run=decode(work,vector.comp,11,257,true);
        EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.consumed,vector.comp.size());
        EXPECT_EQ(run.output,materialize(vector)); EXPECT_EQ(run.last.total_decoded_bytes,vector.raw_len);
        EXPECT_GE(run.frames,1u); work.canaries();
    }
}
TEST(ZstdStream, EverySplitOfSmallOfficialFramesAndOneByteOutput) {
    const auto vectors=load_vectors(); ASSERT_GE(vectors.size(),30u); std::uint32_t checked=0;
    for (const auto& vector:vectors) {
        if (vector.comp.size()>64 || vector.raw_len>4096) continue;
        SCOPED_TRACE(vector.name);
        for (auto chunk=1u;chunk<=vector.comp.size();++chunk) {
            Work work; const auto run=decode(work,vector.comp,chunk,1,true);
            EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.output,materialize(vector));
        }
        ++checked;
    }
    EXPECT_GT(checked,0u);
}
TEST(ZstdStream, ChecksumVerifiedOnlyAtBoundaryAndCorruptionIsSticky) {
    const auto vectors=load_vectors(); const auto* v=find_vector(vectors,"cksum"); ASSERT_NE(v,nullptr);
    Work valid; auto run=decode(valid,v->comp,1,31); ASSERT_EQ(run.status,Status::Finished);
    EXPECT_EQ(run.last.integrity,ZstdFrameIntegrity::ChecksumVerified); EXPECT_EQ(run.output,materialize(*v));
    auto damaged=v->comp; damaged.back()^=1; Work bad;
    run=decode(bad,damaged,3,127); EXPECT_EQ(run.status,Status::ChecksumMismatch);
    EXPECT_EQ(run.last.integrity,ZstdFrameIntegrity::Pending);
    std::array<std::uint8_t,4> output{1,2,3,4};
    const auto repeated=zstd_stream_step(bad.stream,nullptr,0,true,output.data(),4);
    EXPECT_EQ(repeated.status,Status::ChecksumMismatch); EXPECT_EQ(repeated.output_produced,0u);
    EXPECT_EQ(output,(std::array<std::uint8_t,4>{1,2,3,4}));
}
TEST(ZstdStream, IncrementalChecksumAcrossAuthoredBlocksUsesIndependentReferenceTrailer) {
    const auto vectors=load_vectors(); const auto* v=find_vector(vectors,"cksum"); ASSERT_NE(v,nullptr);
    const auto raw=materialize(*v); auto bytes=window_header(); bytes[4]=4;
    constexpr std::uint32_t chunks[]{31,32,33}; std::size_t offset=0;
    for (std::uint32_t i=0;i<raw.size();++i) {
        const auto n=std::min<std::size_t>(chunks[i%3],raw.size()-offset);
        const std::vector<std::uint8_t> part(raw.begin()+static_cast<std::ptrdiff_t>(offset),
            raw.begin()+static_cast<std::ptrdiff_t>(offset+n));
        block(bytes,0,part,offset+n==raw.size()); offset+=n; if (offset==raw.size()) break;
    }
    // Preserve the official reference compressor's checksum; do not calculate
    // expected XXH64 with the production helper under test.
    bytes.insert(bytes.end(),v->comp.end()-4,v->comp.end());
    Work work(1024); const auto run=decode(work,bytes,3,7,true);
    EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.output,raw);
    EXPECT_EQ(run.last.integrity,ZstdFrameIntegrity::ChecksumVerified);
}
TEST(ZstdStream, AllSkippableMagicsAndConcatenatedFramesConsumeExactly) {
    std::vector<std::uint8_t> bytes;
    for (auto i=0u;i<16;++i) {
        append_le(bytes,0x184d2a50u+i,4); append_le(bytes,i,4); bytes.insert(bytes.end(),i,static_cast<std::uint8_t>(i));
        const auto frame=raw_frame({static_cast<std::uint8_t>(i)}); bytes.insert(bytes.end(),frame.begin(),frame.end());
    }
    append_le(bytes,0x184d2a5fu,4); append_le(bytes,0,4);
    Work work(1024); const auto run=decode(work,bytes,1,1,true);
    ASSERT_EQ(run.status,Status::Finished); EXPECT_EQ(run.consumed,bytes.size()); EXPECT_EQ(run.frames,16u);
    ASSERT_EQ(run.output.size(),16u); for (auto i=0u;i<16;++i) EXPECT_EQ(run.output[i],i);
    EXPECT_EQ(run.last.integrity,ZstdFrameIntegrity::NoChecksum);
}
TEST(ZstdStream, FrameLocalHistoryCannotReadAnEarlierFrame) {
    auto bytes=raw_frame({0x5a}); auto second=window_header();
    // Authored RLE sequence tables: LL0, OF2 + extra00 => distance1, ML0=>3.
    block(second,2,{0,1,0x54,0,2,0,4},true); bytes.insert(bytes.end(),second.begin(),second.end());
    Work work(1024); const auto run=decode(work,bytes,2,17);
    EXPECT_EQ(run.status,Status::BadInput); EXPECT_EQ(run.output,(std::vector<std::uint8_t>{0x5a})); EXPECT_EQ(run.frames,1u);
}
TEST(ZstdStream, InclusiveWindowDistanceAndRingWrapAreExact) {
    std::vector<std::uint8_t> first(1024); for (auto i=0u;i<first.size();++i) first[i]=static_cast<std::uint8_t>(i%251);
    auto bytes=window_header(); block(bytes,0,first,false); block(bytes,0,std::vector<std::uint8_t>(1023,0x5a),false);
    // Authored distance1024: offset value1027, OF10 extra0000000011, ML3.
    block(bytes,2,{0,1,0x54,0,10,0,3,4},true);
    auto expected=first; expected.insert(expected.end(),1023,0x5a);
    expected.push_back(first.back()); expected.push_back(0x5a); expected.push_back(0x5a);
    Work work(1024); const auto run=decode(work,bytes,7,13,true);
    EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.output,expected); work.canaries();
    bytes[bytes.size()-2]=2; Work shorter(1024);
    auto shorter_expected=expected; std::fill(shorter_expected.end()-3,shorter_expected.end(),0x5a);
    EXPECT_EQ(decode(shorter,bytes,7,13).output,shorter_expected); // distance1023.
    // Same framing but distance1025 (extra4) exceeds the advertised window.
    bytes[bytes.size()-2]=4; Work over(1024); EXPECT_EQ(decode(over,bytes,7,13).status,Status::BadInput);
}
TEST(ZstdStream, ZeroRepeatAndTrailingSequenceBitsRefuseWithoutPublishingBlock) {
    for (const auto& body: {std::vector<std::uint8_t>{0,1,0x54,0,1,0,3},
                           std::vector<std::uint8_t>{0,1,0x54,0,2,0,0,4}}) {
        auto bytes=window_header(); block(bytes,0,{0x5a},false); block(bytes,2,body,true);
        Work work(1024); const auto run=decode(work,bytes,1,32);
        EXPECT_EQ(run.status,Status::BadInput); EXPECT_EQ(run.output,(std::vector<std::uint8_t>{0x5a}));
    }
}
TEST(ZstdStream, AuthoredHuffmanExactUsefulBitsAndExtraMissingBitRefusal) {
    // Direct tree: one transmitted weight1 (symbol0), inferred weight1
    // (symbol1). Each symbol has a1-bit code. Regen3/comp3 header is32 c0 00;
    // 0a contains marker bit3 followed by exactly010 => bytes{0,1,0}.
    // These are authored format examples, not official compressor captures.
    for (const auto bits : {0x0au,0x1au,0x06u}) {
        auto bytes=window_header();
        block(bytes,2,{0x32,0xc0,0,0x80,0x10,static_cast<std::uint8_t>(bits),0},true);
        Work work(1024); const auto run=decode(work,bytes,1,1,true);
        if (bits==0x0a) {
            EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.output,(std::vector<std::uint8_t>{0,1,0}));
        } else { EXPECT_EQ(run.status,Status::BadInput); EXPECT_TRUE(run.output.empty()); }
    }
}
TEST(ZstdStream, UnsupportedWindowProfilePrecedesCallerBudgetWithoutHugeHistory) {
    auto header=window_header(); header[5]=0xb0; // WindowLog32 =>4GiB.
    Work work(1024); auto run=decode(work,header,1,1);
    EXPECT_EQ(run.status,Status::UnsupportedWindow); EXPECT_TRUE(run.output.empty());
    header[5]=8; //2KiB is inside profile, outside the caller's1KiB budget.
    Work limited(1024); EXPECT_EQ(decode(limited,header,1,1).status,Status::WindowLimit);
}
TEST(ZstdStream, BlockLimitUsesWindowAndLogicalLimitsNeverGrow) {
    auto frame=raw_frame(std::vector<std::uint8_t>(1025,0x5a)); Work small(1024);
    EXPECT_EQ(decode(small,frame,1024,1024).status,Status::BadInput);
    frame=raw_frame(std::vector<std::uint8_t>(1024,0x5a)); Work limited(1024,1023);
    const auto run=decode(limited,frame,1024,1024); EXPECT_EQ(run.status,Status::OutputLimit); EXPECT_TRUE(run.output.empty());
    // Single-segment content size declares required history, even before data.
    const std::vector<std::uint8_t> single{0x28,0xb5,0x2f,0xfd,0x20,128,1,4,0}; Work window(64);
    EXPECT_EQ(decode(window,single,1,32).status,Status::WindowLimit);
}
TEST(ZstdStream, TruncationAtEveryByteAndFinalSuffixNeverReportsPrematureEof) {
    const auto bytes=raw_frame({1,2,3,4,5,6,7});
    for (auto n=0u;n<bytes.size();++n) {
        Work work(1024); const std::vector<std::uint8_t> prefix(bytes.begin(),bytes.begin()+static_cast<std::ptrdiff_t>(n));
        EXPECT_EQ(decode(work,prefix,1,1,true).status,Status::Truncated);
    }
    auto trailing=bytes; trailing.push_back(0); Work bad(1024); EXPECT_EQ(decode(bad,trailing,4,4).status,Status::Truncated);
    Work good(1024); auto run=decode(good,bytes,static_cast<std::uint32_t>(bytes.size()),1,true);
    EXPECT_EQ(run.status,Status::Finished); EXPECT_EQ(run.consumed,bytes.size()); EXPECT_EQ(run.output.size(),7u);
    std::uint8_t byte=0; EXPECT_EQ(zstd_stream_step(good.stream,&byte,1,true,nullptr,0).status,Status::InvalidArgument);
}
TEST(ZstdStream, FinalInputIsRetainedBeforeOutputDrainAndNoChecksumFcsMustMatch) {
    const auto bytes=raw_frame({1,2,3}); Work work(1024); std::uint32_t consumed=0;
    for (auto i=0u;i<16 && consumed<bytes.size();++i) {
        const auto r=zstd_stream_step(work.stream,bytes.data()+consumed,
            static_cast<std::uint32_t>(bytes.size())-consumed,true,nullptr,0);
        consumed+=r.input_consumed;
    }
    ASSERT_EQ(consumed,bytes.size()); std::uint8_t extra=9;
    EXPECT_EQ(zstd_stream_step(work.stream,&extra,1,false,nullptr,0).status,Status::InvalidArgument);
    std::array<std::uint8_t,8> output{}; std::uint32_t produced=0; ZstdStep last{};
    for (auto i=0u;i<16;++i) {
        last=zstd_stream_step(work.stream,nullptr,0,false,output.data()+produced,8-produced);
        produced+=last.output_produced; if (last.status==Status::Finished) break;
    }
    EXPECT_EQ(last.status,Status::Finished); EXPECT_EQ(produced,3u);
    for (const auto fcs : {2u,4u}) {
        std::vector<std::uint8_t> frame{0x28,0xb5,0x2f,0xfd,0x20,static_cast<std::uint8_t>(fcs)};
        block(frame,0,{1,2,3},true); Work mismatch(1024);
        EXPECT_EQ(decode(mismatch,frame,1,1).status,Status::BadInput);
    }
}
TEST(ZstdStream, DictionaryEncodingAndReservedFieldsRefuseExplicitly) {
    const std::vector<std::uint8_t> dict{0x28,0xb5,0x2f,0xfd,0x21,0,0,1,0,0};
    Work dictionary(1024); EXPECT_EQ(decode(dictionary,dict,1,8).status,Status::UnsupportedDictionaryEncoding);
    auto bytes=raw_frame({1}); bytes[4]=8; Work reserved(1024); EXPECT_EQ(decode(reserved,bytes,1,8).status,Status::BadInput);
    bytes=raw_frame({1}); bytes[4]=0x10; Work unused(1024); EXPECT_EQ(decode(unused,bytes,1,8).status,Status::Finished);
}
TEST(ZstdStream, AliasAdmissionPreservesStateAndCallerBytes) {
    Work work(1024); auto frame=raw_frame({1,2,3});
    auto* memory=reinterpret_cast<std::uint8_t*>(work.words.data()+1);
    const auto snapshot=work.words; const auto history=work.history;
    EXPECT_EQ(zstd_stream_step(work.stream,frame.data(),1,false,frame.data(),1).status,Status::InvalidArgument);
    EXPECT_EQ(zstd_stream_step(work.stream,memory+zstd_stream_state_size()-1,1,false,nullptr,0).status,Status::InvalidArgument);
    EXPECT_EQ(zstd_stream_step(work.stream,nullptr,0,false,work.history.data()+8+1023,1).status,Status::InvalidArgument);
    EXPECT_EQ(work.words,snapshot); EXPECT_EQ(work.history,history);
    EXPECT_EQ(decode(work,frame,1,1).status,Status::Finished);
    ZstdStream* sentinel=work.stream;
    EXPECT_EQ(zstd_stream_init(memory,8,work.history.data()+8,1024,{1024,100},&sentinel),Status::WorkspaceSmall);
    EXPECT_EQ(sentinel,work.stream);
    EXPECT_EQ(zstd_stream_init(memory,zstd_stream_state_size(),memory,1024,{1024,100},&sentinel),Status::InvalidArgument);
    EXPECT_EQ(sentinel,work.stream); work.canaries();
    const auto* impossible=reinterpret_cast<const std::uint8_t*>(UINTPTR_MAX-1);
    EXPECT_EQ(zstd_stream_step(work.stream,impossible,4,true,nullptr,0).status,Status::InvalidArgument);
}
TEST(ZstdStream, LargeFrameUsesFixedCallerMemoryAndNoCppHeapDuringDecode) {
    // Authored128MiB logical frame: no whole-frame encoded or decoded fixture.
    // One reusable1KiB RLE block yields each1KiB of output, under a1KiB window.
    constexpr std::uint64_t blocks=128u*1024u; Work work(1024,blocks*1024);
    const std::array<std::uint8_t,6> header{0x28,0xb5,0x2f,0xfd,0,0};
    std::array<std::uint8_t,4> encoded{2,32,0,0x5a}; // RLE1024, not final.
    std::array<std::uint8_t,4093> output{};
    const auto state_capacity=work.words.capacity(), history_capacity=work.history.capacity();
    const auto allocations=heap_allocations.load(std::memory_order_relaxed);
    std::uint64_t block_index=0,total=0,calls=0; std::uint32_t offset=0;
    bool header_phase=true,good=true; ZstdStep last{};
    for (;calls<blocks*12+32;++calls) {
        const auto* input=header_phase?header.data():encoded.data();
        const auto length=header_phase?6u:4u;
        const bool final=!header_phase && block_index+1==blocks;
        if (final) encoded[0]=3;
        last=zstd_stream_step(work.stream,input+offset,length-offset,final,output.data(),static_cast<std::uint32_t>(output.size()));
        offset+=last.input_consumed; total+=last.output_produced;
        for (auto i=0u;i<last.output_produced;++i) if (output[i]!=0x5a) good=false;
        if (offset==length) {
            if (header_phase) { header_phase=false; offset=0; }
            else if (!final) { ++block_index; offset=0; }
        }
        if (last.status==Status::Finished || last.status>=Status::InvalidArgument) break;
    }
    const auto delta=heap_allocations.load(std::memory_order_relaxed)-allocations;
    EXPECT_EQ(delta,0u); EXPECT_EQ(last.status,Status::Finished); EXPECT_TRUE(good); EXPECT_EQ(total,blocks*1024);
    EXPECT_EQ(work.words.capacity(),state_capacity); EXPECT_EQ(work.history.capacity(),history_capacity); work.canaries();
}
} // namespace
