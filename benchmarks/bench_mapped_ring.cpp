// Bounded owner-local mapped-ring round trip; not packet/CME pipeline latency.
#include "bolt/io/bolt_mapped_ring.h"
#include "bolt/io/bolt_file_io.h"
#include "bolt/bolt_port.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {
using Clock=std::chrono::steady_clock;
using namespace bolt::io;
constexpr std::uint64_t kDefaultEvents=100000,kMaxEvents=10000000;
volatile unsigned marker_state=0;
struct Options {std::uint64_t events=kDefaultEvents;bool control=false;};
struct Usage {
    bool available=false;
    std::int64_t minor=0,major=0,voluntary=0,involuntary=0,input_blocks=0,output_blocks=0;
};
struct Result {std::uint64_t completed=0,failures=0,elapsed_ns=0;Usage usage{};};
Usage process_usage() noexcept {
    Usage out{};
#if !defined(_WIN32)
    rusage usage{};
    if(::getrusage(RUSAGE_SELF,&usage)==0) {
        out={true,usage.ru_minflt,usage.ru_majflt,usage.ru_nvcsw,usage.ru_nivcsw,
             usage.ru_inblock,usage.ru_oublock};
    }
#endif
    return out;
}
Usage usage_delta(const Usage& before,const Usage& after) noexcept {
    if(!before.available || !after.available)return {};
    return {true,after.minor-before.minor,after.major-before.major,
        after.voluntary-before.voluntary,after.involuntary-before.involuntary,
        after.input_blocks-before.input_blocks,after.output_blocks-before.output_blocks};
}
std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}
}
extern "C" BOLT_NOINLINE void g2_mapped_ring_begin() noexcept {marker_state=1;}
extern "C" BOLT_NOINLINE void g2_mapped_ring_end() noexcept {marker_state=2;}
extern "C" {void* volatile g2_mapped_ring_control_escape=nullptr;}
extern "C" BOLT_NOINLINE bool g2_mapped_ring_allocation_control() noexcept {
    auto* p=std::malloc(73);if(!p)return false;
    g2_mapped_ring_control_escape=p;static_cast<volatile std::uint8_t*>(p)[0]=0x5a;
    g2_mapped_ring_control_escape=nullptr;std::free(p);return true;
}
namespace {
bool parse(int argc,char** argv,Options* out) noexcept {
    assert(out!=nullptr);assert(argc>=1);
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--allocation-loop")==0)continue;
        if(std::strcmp(argv[i],"--allocation-control")==0){out->control=true;continue;}
        if(std::strcmp(argv[i],"--events")!=0 || ++i>=argc)return false;
        const auto* p=argv[i];if(*p=='\0')return false;std::uint64_t n=0;
        for(unsigned j=0;p[j]!='\0';++j) {
            if(j>=8 || p[j]<'0' || p[j]>'9')return false;
            n=n*10+static_cast<unsigned>(p[j]-'0');
        }
        if(n==0 || n>kMaxEvents)return false;
        out->events=n;
    }
    assert(out->events>0 && out->events<=kMaxEvents);return true;
}
bool round_trip(MappedRing* ring,std::uint64_t sequence) noexcept {
    assert(ring!=nullptr);assert(sequence>0);
    // Each literal word is independently checked, including binary zero bytes.
    const std::uint64_t words[8]={sequence,0,UINT64_MAX,0x0123456789abcdefULL,
        0xff00ff00ff00ff00ULL,0x4142434445464748ULL,sequence^UINT64_MAX,73};
    RingReservation r{};if(mapped_ring_reserve(ring,sizeof(words),&r)!=RingStatus::kOk)return false;
    std::memcpy(r.data,words,sizeof(words));
    if(mapped_ring_commit(ring,r)!=RingStatus::kOk){(void)mapped_ring_cancel(ring,r);return false;}
    RingView view{};if(mapped_ring_peek(ring,0,&view)!=RingStatus::kOk)return false;
    const bool valid=view.sequence==sequence && view.bytes==sizeof(words) &&
        std::memcmp(view.data,words,sizeof(words))==0;
    const auto released=mapped_ring_release(ring,view);
    return valid && released==RingStatus::kOk;
}
Result measure(MappedRing* ring,const Options& options,std::uint64_t* samples) noexcept {
    assert(ring!=nullptr);assert(samples!=nullptr);
    Result result{};const auto usage_before=process_usage();
    const auto begin=now_ns();g2_mapped_ring_begin();
    if(options.control && !g2_mapped_ring_allocation_control())++result.failures;
    for(std::uint64_t i=0;i<options.events;++i) {
        const auto start=now_ns();const bool ok=round_trip(ring,i+2);
        samples[i]=now_ns()-start;
        if(!ok){++result.failures;break;}
        ++result.completed;
    }
    g2_mapped_ring_end();result.elapsed_ns=now_ns()-begin;
    const auto usage_after=process_usage();result.usage=usage_delta(usage_before,usage_after);
    assert(result.completed<=options.events);assert(marker_state==2);return result;
}
void report(const Options& options,const Result& result,std::uint64_t* samples,
            const RingSnapshot& snapshot) noexcept {
    assert(samples!=nullptr);assert(result.completed<=options.events);
    std::sort(samples,samples+result.completed);
    const auto percentile=[&](std::uint64_t p) noexcept {
        return result.completed?samples[(result.completed*p+99)/100-1]:0;
    };
    std::printf("mode=direct payload_bytes=64 requested=%llu completed=%llu failures=%llu\n",
        static_cast<unsigned long long>(options.events),static_cast<unsigned long long>(result.completed),
        static_cast<unsigned long long>(result.failures));
    std::printf("reserve_fill_commit_peek_verify_release_ns p50=%llu p99=%llu max=%llu elapsed_ns=%llu\n",
        static_cast<unsigned long long>(percentile(50)),static_cast<unsigned long long>(percentile(99)),
        static_cast<unsigned long long>(result.completed?samples[result.completed-1]:0),
        static_cast<unsigned long long>(result.elapsed_ns));
    std::printf("capacity_bytes=%llu page_bytes=%u mapping_granularity=%u sample_bytes=%llu allocation_control=%u\n",
        static_cast<unsigned long long>(snapshot.capacity_bytes),snapshot.page_bytes,snapshot.mapping_granularity,
        static_cast<unsigned long long>(options.events*sizeof(std::uint64_t)),options.control?1u:0u);
    std::printf("resource_counters_scope=process available=%s sampling=outside_markers_and_elapsed_loop\n",
        result.usage.available?"true":"false");
    if(result.usage.available) {
        const auto& u=result.usage;
        std::printf("process_delta minor_faults=%lld major_faults=%lld voluntary_context_switches=%lld "
                    "involuntary_context_switches=%lld input_blocks=%lld output_blocks=%lld\n",
            static_cast<long long>(u.minor),static_cast<long long>(u.major),
            static_cast<long long>(u.voluntary),static_cast<long long>(u.involuntary),
            static_cast<long long>(u.input_blocks),static_cast<long long>(u.output_blocks));
    }
    std::puts("Owner-local ephemeral round trip; includes two clock reads; no clock-cost subtraction, flush or durability claim.");
}
}
int main(int argc,char** argv) {
    Options options{};if(!parse(argc,argv,&options)) {
        std::fprintf(stderr,"usage: %s [--events 1..10000000] [--allocation-loop|--allocation-control]\n",argv[0]);return 2;
    }
    const auto unit=std::max(bolt::io::io_page_size(),bolt::io::io_map_granularity());
    const auto capacity=((std::uint64_t{4}*1024*1024+unit-1)/unit)*unit;
    const std::string path=(std::filesystem::temp_directory_path()/
        ("bolt-ring-bench-"+std::to_string(now_ns()))).string();
    auto* samples=static_cast<std::uint64_t*>(std::calloc(static_cast<std::size_t>(options.events),sizeof(std::uint64_t)));
    if(!samples){std::fprintf(stderr,"sample allocation refused\n");return 1;}
    MappedRing* ring=nullptr;
    const RingOptions config{capacity,256,1,true,true,false,nullptr,nullptr,nullptr};
    const auto opened=mapped_ring_open(path.c_str(),config,&ring);
    if(opened!=RingStatus::kOk){std::fprintf(stderr,"ring open failed status=%u\n",static_cast<unsigned>(opened));std::free(samples);return 1;}
    // Warm clock/ring code and fault in every sample page before probe markers.
    for(std::uint64_t i=0;i<options.events;++i)samples[i]=now_ns();
    if(!round_trip(ring,1)) {
        std::fprintf(stderr,"warmup failed\n");(void)mapped_ring_close(&ring);std::free(samples);return 1;
    }
    const auto result=measure(ring,options,samples);
    report(options,result,samples,mapped_ring_snapshot(ring));
    const auto closed=mapped_ring_close(&ring);std::free(samples);
    if(closed!=RingStatus::kOk)std::fprintf(stderr,"ring close failed status=%u\n",static_cast<unsigned>(closed));
    return result.failures==0 && result.completed==options.events && closed==RingStatus::kOk?0:1;
}
