// ybolt_cli — Yjs interop CLI exercising the bolt::Arena-backed binding.
//
// Same subcommands as ycpp_cli (apply-and-emit / list-map / dump-text),
// but every allocation routes through bolt::Arena via
// bolt::ybolt::BoltArenaAllocator. The Node interop harness can drive
// this binary the same way it drives ycpp_cli to verify the production
// binding's wire behaviour matches.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
  #include <fcntl.h>
  #include <io.h>
#endif

#include "bolt/bolt_arena.h"
#include "bolt/ybolt.h"

namespace {

void set_binary_io() noexcept {
#if defined(_WIN32)
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

bool read_all_stdin(std::vector<uint8_t>* out) noexcept {
    uint8_t chunk[4096];
    while (true) {
        std::size_t n = std::fread(chunk, 1, sizeof(chunk), stdin);
        if (n > 0) out->insert(out->end(), chunk, chunk + n);
        if (n < sizeof(chunk)) {
            if (std::feof(stdin)) return true;
            if (std::ferror(stdin)) {
                std::fprintf(stderr, "ybolt_cli: stdin read error\n");
                return false;
            }
        }
    }
}

int do_apply_and_emit() {
    std::vector<uint8_t> in;
    if (!read_all_stdin(&in)) return 1;

    bolt::Arena arena;
    auto doc = bolt::ybolt::make_doc(arena, /*client_id=*/0xC0FFEE);

    const ycpp::Status s = doc.apply_update_v1(
        ycpp::ByteView{in.data(), in.size()});
    if (s != ycpp::Status::kOk) {
        std::fprintf(stderr, "ybolt_cli: apply failed: %s\n",
                     ycpp::status_name(s));
        return 1;
    }

    std::vector<uint8_t> out(1 << 20);
    ycpp::Writer w{out.data(), out.size()};
    const ycpp::Status es = bolt::ybolt::encode_diff_v1(doc, nullptr, &w);
    if (es != ycpp::Status::kOk) {
        std::fprintf(stderr, "ybolt_cli: encode failed: %s\n",
                     ycpp::status_name(es));
        return 1;
    }
    std::fwrite(out.data(), 1, w.pos(), stdout);
    return 0;
}

int do_list_map(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ybolt_cli list-map <root> <key>\n");
        return 1;
    }
    const char* root = argv[2];
    const char* key  = argv[3];

    std::vector<uint8_t> in;
    if (!read_all_stdin(&in)) return 1;

    bolt::Arena arena;
    auto doc = bolt::ybolt::make_doc(arena, 0xC0FFEE);

    const ycpp::Status s = doc.apply_update_v1(
        ycpp::ByteView{in.data(), in.size()});
    if (s != ycpp::Status::kOk) {
        std::fprintf(stderr, "ybolt_cli: apply failed: %s\n",
                     ycpp::status_name(s));
        return 1;
    }
    auto* m = doc.get_or_create_map(root);
    if (m == nullptr) return 2;
    auto* it = m->get(ycpp::ByteView{
        reinterpret_cast<const uint8_t*>(key), std::strlen(key)});
    if (it == nullptr) return 2;
    std::fwrite(it->content_view.data, 1, it->content_view.size, stdout);
    return 0;
}

int do_dump_text(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ybolt_cli dump-text <root>\n");
        return 1;
    }
    const char* root = argv[2];

    std::vector<uint8_t> in;
    if (!read_all_stdin(&in)) return 1;

    bolt::Arena arena;
    auto doc = bolt::ybolt::make_doc(arena, 0xC0FFEE);

    const ycpp::Status s = doc.apply_update_v1(
        ycpp::ByteView{in.data(), in.size()});
    if (s != ycpp::Status::kOk) {
        std::fprintf(stderr, "ybolt_cli: apply failed: %s\n",
                     ycpp::status_name(s));
        return 1;
    }
    auto text = doc.get_or_create_text(root);
    text.for_each_chunk([](ycpp::ByteView b) noexcept {
        if (b.size > 0) std::fwrite(b.data, 1, b.size, stdout);
    });
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    set_binary_io();
    if (argc < 2) {
        std::fprintf(stderr, "usage: ybolt_cli <apply-and-emit|list-map|dump-text> [args]\n");
        return 1;
    }
    const char* cmd = argv[1];
    if (std::strcmp(cmd, "apply-and-emit") == 0) return do_apply_and_emit();
    if (std::strcmp(cmd, "list-map"      ) == 0) return do_list_map(argc, argv);
    if (std::strcmp(cmd, "dump-text"     ) == 0) return do_dump_text(argc, argv);
    std::fprintf(stderr, "ybolt_cli: unknown command '%s'\n", cmd);
    return 1;
}
