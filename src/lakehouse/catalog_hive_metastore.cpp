// bolt/src/lakehouse/catalog_hive_metastore.cpp — Hive Metastore catalog.
//
// Minimal Thrift Binary Protocol codec over plain TCP. Thrift type IDs:
// 0=STOP 2=BOOL 3=BYTE 6=I16 8=I32 10=I64 4=DOUBLE 11=STRING 12=STRUCT
// 13=MAP 14=SET 15=LIST. Message envelope: i32 (version|type), string name,
// i32 seqid, args struct.
//
// Implements get_all_databases / get_all_tables(db). create_database /
// drop_database / create_table / drop_table are minimal wrappers.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "bolt/lakehouse/catalog/hive_metastore.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using HmsSock = SOCKET;
#  define HMS_BAD INVALID_SOCKET
#  define HMS_CLOSE(s) closesocket(s)
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
   using HmsSock = int;
#  define HMS_BAD (-1)
#  define HMS_CLOSE(s) ::close(s)
#endif

namespace bolt {
namespace lakehouse {
namespace hive_metastore {

namespace {

struct Impl {
    Config       cfg;
    bolt::Arena* arena;
};

// Big-endian writers/readers.
bool be_write_u8(uint8_t* b, uint32_t cap, uint32_t* c, uint8_t v) noexcept {
    if (*c + 1u > cap) return false;
    b[(*c)++] = v;
    return true;
}
bool be_write_i16(uint8_t* b, uint32_t cap, uint32_t* c, int16_t v) noexcept {
    if (*c + 2u > cap) return false;
    uint16_t u = static_cast<uint16_t>(v);
    b[(*c)++] = static_cast<uint8_t>((u >> 8) & 0xff);
    b[(*c)++] = static_cast<uint8_t>(u & 0xff);
    return true;
}
bool be_write_i32(uint8_t* b, uint32_t cap, uint32_t* c, int32_t v) noexcept {
    if (*c + 4u > cap) return false;
    uint32_t u = static_cast<uint32_t>(v);
    b[(*c)++] = static_cast<uint8_t>((u >> 24) & 0xff);
    b[(*c)++] = static_cast<uint8_t>((u >> 16) & 0xff);
    b[(*c)++] = static_cast<uint8_t>((u >> 8) & 0xff);
    b[(*c)++] = static_cast<uint8_t>(u & 0xff);
    return true;
}
bool be_read_u8(const uint8_t* b, uint32_t L, uint32_t* c, uint8_t* o) noexcept {
    if (*c + 1u > L) return false;
    *o = b[(*c)++];
    return true;
}
bool be_read_i16(const uint8_t* b, uint32_t L, uint32_t* c, int16_t* o) noexcept {
    if (*c + 2u > L) return false;
    uint16_t v = static_cast<uint16_t>(
        (static_cast<uint16_t>(b[*c]) << 8) | b[*c + 1u]);
    *c += 2u;
    *o = static_cast<int16_t>(v);
    return true;
}
bool be_read_i32(const uint8_t* b, uint32_t L, uint32_t* c, int32_t* o) noexcept {
    if (*c + 4u > L) return false;
    uint32_t v = (static_cast<uint32_t>(b[*c]) << 24) |
                 (static_cast<uint32_t>(b[*c + 1u]) << 16) |
                 (static_cast<uint32_t>(b[*c + 2u]) << 8) |
                  static_cast<uint32_t>(b[*c + 3u]);
    *c += 4u;
    *o = static_cast<int32_t>(v);
    return true;
}
bool be_read_i64(const uint8_t* b, uint32_t L, uint32_t* c, int64_t* o) noexcept {
    if (*c + 8u > L) return false;
    uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v = (v << 8) | b[*c + k];
    *c += 8u;
    *o = static_cast<int64_t>(v);
    return true;
}

// Skip one Thrift value of `type`. Returns false on overflow.
bool tb_skip_value(const uint8_t* b, uint32_t L, uint32_t* c, uint8_t type) noexcept {
    switch (type) {
        case 2: case 3: { uint8_t v; return be_read_u8(b, L, c, &v); }
        case 6: { int16_t v; return be_read_i16(b, L, c, &v); }
        case 8: { int32_t v; return be_read_i32(b, L, c, &v); }
        case 10: case 4: { int64_t v; return be_read_i64(b, L, c, &v); }
        case 11: {
            int32_t sn = 0;
            if (!be_read_i32(b, L, c, &sn) || sn < 0) return false;
            if (*c + static_cast<uint32_t>(sn) > L) return false;
            *c += static_cast<uint32_t>(sn);
            return true;
        }
        case 12: {
            // STRUCT: read fields until STOP.
            while (true) {
                uint8_t ft;
                if (!be_read_u8(b, L, c, &ft)) return false;
                if (ft == 0) return true;
                int16_t fid;
                if (!be_read_i16(b, L, c, &fid)) return false;
                if (!tb_skip_value(b, L, c, ft)) return false;
            }
        }
        case 13: {
            // MAP: kt, vt, size, then size*(k,v).
            uint8_t kt, vt;
            if (!be_read_u8(b, L, c, &kt)) return false;
            if (!be_read_u8(b, L, c, &vt)) return false;
            int32_t sz;
            if (!be_read_i32(b, L, c, &sz) || sz < 0) return false;
            for (int32_t k = 0; k < sz; ++k) {
                if (!tb_skip_value(b, L, c, kt)) return false;
                if (!tb_skip_value(b, L, c, vt)) return false;
            }
            return true;
        }
        case 14: case 15: {
            uint8_t et;
            if (!be_read_u8(b, L, c, &et)) return false;
            int32_t sz;
            if (!be_read_i32(b, L, c, &sz) || sz < 0) return false;
            for (int32_t k = 0; k < sz; ++k) {
                if (!tb_skip_value(b, L, c, et)) return false;
            }
            return true;
        }
        default: return false;
    }
}

HmsSock hms_connect(const char* host, uint16_t port) noexcept {
#ifdef _WIN32
    static bool inited = false;
    if (!inited) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return HMS_BAD;
        inited = true;
    }
#endif
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host, nullptr, &hints, &res) != 0 || res == nullptr) {
        return HMS_BAD;
    }
    sockaddr_in addr = *reinterpret_cast<const sockaddr_in*>(res->ai_addr);
    addr.sin_port = htons(port);
    freeaddrinfo(res);
    HmsSock s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == HMS_BAD) return HMS_BAD;
    if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        HMS_CLOSE(s);
        return HMS_BAD;
    }
    return s;
}

bool sock_send_all(HmsSock s, const void* d, uint32_t n) noexcept {
    const char* p = static_cast<const char*>(d);
    uint32_t off = 0;
    while (off < n) {
        int k = static_cast<int>(send(s, p + off,
                                      static_cast<int>(n - off), 0));
        if (k <= 0) return false;
        off += static_cast<uint32_t>(k);
    }
    return true;
}
bool sock_recv_all(HmsSock s, void* d, uint32_t n) noexcept {
    char* p = static_cast<char*>(d);
    uint32_t off = 0;
    while (off < n) {
        int k = static_cast<int>(recv(s, p + off,
                                      static_cast<int>(n - off), 0));
        if (k <= 0) return false;
        off += static_cast<uint32_t>(k);
    }
    return true;
}

// Build a CALL message: version|1, name, seqid, then args. args_buf already
// holds the struct including STOP. Writes to `out` and returns total length.
bool tb_build_call(uint8_t* out, uint32_t cap, uint32_t* out_len,
                   const char* method, int32_t seqid,
                   const uint8_t* args, uint32_t args_len) noexcept {
    uint32_t c = 0;
    int32_t vt = static_cast<int32_t>(0x80010001);   // version 1 | CALL
    if (!be_write_i32(out, cap, &c, vt)) return false;
    uint32_t mn = static_cast<uint32_t>(std::strlen(method));
    if (!be_write_i32(out, cap, &c, static_cast<int32_t>(mn))) return false;
    if (c + mn > cap) return false;
    std::memcpy(out + c, method, mn); c += mn;
    if (!be_write_i32(out, cap, &c, seqid)) return false;
    if (c + args_len > cap) return false;
    std::memcpy(out + c, args, args_len); c += args_len;
    *out_len = c;
    return true;
}

// Parse a REPLY message; returns cursor pointing at args struct (i.e. first
// field header). Returns false on error / exception type.
bool tb_parse_reply(const uint8_t* b, uint32_t L, uint32_t* out_cursor) noexcept {
    uint32_t c = 0;
    int32_t vt;
    if (!be_read_i32(b, L, &c, &vt)) return false;
    int32_t type = vt & 0xff;
    // type: 2 = REPLY, 3 = EXCEPTION
    if (type != 2 && type != 3) return false;
    int32_t mn;
    if (!be_read_i32(b, L, &c, &mn) || mn < 0) return false;
    c += static_cast<uint32_t>(mn);
    int32_t seq;
    if (!be_read_i32(b, L, &c, &seq)) return false;
    *out_cursor = c;
    return type == 2;
}

// Send args buffer and recv a length-framed reply. HMS uses TFramedTransport:
// 4-byte big-endian length prefix on each message. Returns reply size or
// negative on error. Caller-owned reply buffer (cap).
int hms_call(Impl* impl, const char* method, const uint8_t* args,
             uint32_t args_len, uint8_t* reply, uint32_t reply_cap,
             uint32_t* reply_len_out) noexcept {
    assert(impl != nullptr);
    assert(method != nullptr);
    HmsSock s = hms_connect(impl->cfg.host, impl->cfg.port);
    if (s == HMS_BAD) return -1;

    uint8_t msg[16384];
    uint32_t msg_len = 0;
    if (!tb_build_call(msg, sizeof(msg), &msg_len, method, 1, args, args_len)) {
        HMS_CLOSE(s); return -2;
    }
    // Frame: 4-byte big-endian length.
    uint8_t hdr[4];
    uint32_t L = msg_len;
    hdr[0] = static_cast<uint8_t>((L >> 24) & 0xff);
    hdr[1] = static_cast<uint8_t>((L >> 16) & 0xff);
    hdr[2] = static_cast<uint8_t>((L >> 8) & 0xff);
    hdr[3] = static_cast<uint8_t>(L & 0xff);
    if (!sock_send_all(s, hdr, 4)) { HMS_CLOSE(s); return -3; }
    if (!sock_send_all(s, msg, L)) { HMS_CLOSE(s); return -4; }

    if (!sock_recv_all(s, hdr, 4)) { HMS_CLOSE(s); return -5; }
    uint32_t rlen = (static_cast<uint32_t>(hdr[0]) << 24) |
                    (static_cast<uint32_t>(hdr[1]) << 16) |
                    (static_cast<uint32_t>(hdr[2]) << 8) |
                     static_cast<uint32_t>(hdr[3]);
    if (rlen > reply_cap) { HMS_CLOSE(s); return -6; }
    if (!sock_recv_all(s, reply, rlen)) { HMS_CLOSE(s); return -7; }
    HMS_CLOSE(s);
    *reply_len_out = rlen;
    return 0;
}

// list-of-strings args helper for thrift methods that take no args (just STOP)
// or one string arg.
uint32_t build_args_empty(uint8_t* buf) noexcept {
    buf[0] = 0;  // STOP
    return 1u;
}
uint32_t build_args_one_string(uint8_t* buf, uint32_t cap, int16_t fid,
                               const char* s) noexcept {
    uint32_t c = 0;
    uint32_t n = static_cast<uint32_t>(std::strlen(s));
    if (!be_write_u8(buf, cap, &c, 11)) return 0;
    if (!be_write_i16(buf, cap, &c, fid)) return 0;
    if (!be_write_i32(buf, cap, &c, static_cast<int32_t>(n))) return 0;
    if (c + n > cap) return 0;
    std::memcpy(buf + c, s, n); c += n;
    if (!be_write_u8(buf, cap, &c, 0)) return 0;
    return c;
}

// Parse reply that's a LIST<STRING> as fid=0 (single STRUCT field, type=15).
bool parse_reply_list_of_string(const uint8_t* b, uint32_t L,
                                CatalogName* out, uint32_t cap,
                                uint32_t* out_n) noexcept {
    *out_n = 0;
    uint32_t c = 0;
    if (!tb_parse_reply(b, L, &c)) return false;
    // Walk struct fields.
    while (true) {
        uint8_t ft;
        if (!be_read_u8(b, L, &c, &ft)) return false;
        if (ft == 0) break;
        int16_t fid;
        if (!be_read_i16(b, L, &c, &fid)) return false;
        if (ft == 15 && fid == 0) {
            uint8_t et;
            if (!be_read_u8(b, L, &c, &et)) return false;
            int32_t sz;
            if (!be_read_i32(b, L, &c, &sz) || sz < 0) return false;
            for (int32_t k = 0; k < sz; ++k) {
                if (et != 11) return false;
                int32_t slen;
                if (!be_read_i32(b, L, &c, &slen) || slen < 0) return false;
                if (c + static_cast<uint32_t>(slen) > L) return false;
                if (*out_n < cap) {
                    uint32_t n = static_cast<uint32_t>(slen);
                    if (n >= kCatMaxName) n = kCatMaxName - 1u;
                    std::memcpy(out[*out_n].name, b + c, n);
                    out[*out_n].name[n] = '\0';
                    ++(*out_n);
                }
                c += static_cast<uint32_t>(slen);
            }
        } else {
            if (!tb_skip_value(b, L, &c, ft)) return false;
        }
    }
    return true;
}

// Parse a get_table REPLY: success field fid=0 is a Table STRUCT. Walk it to
// field 7 (sd: StorageDescriptor STRUCT), then sd field 2 (location: STRING).
// Copies the location into out[0..cap). Returns false on a malformed reply.
bool parse_get_table_location(const uint8_t* b, uint32_t L, char* out,
                              uint32_t cap) noexcept {
    assert(b != nullptr && out != nullptr && cap > 0u);
    out[0] = '\0';
    uint32_t c = 0;
    if (!tb_parse_reply(b, L, &c)) return false;
    // Outer args struct: find field 0 (the Table struct).
    while (true) {
        uint8_t ft;
        if (!be_read_u8(b, L, &c, &ft)) return false;
        if (ft == 0) return true;                 // no success field present
        int16_t fid;
        if (!be_read_i16(b, L, &c, &fid)) return false;
        if (ft == 12 && fid == 0) {
            // Walk the Table struct fields.
            while (true) {
                uint8_t tf;
                if (!be_read_u8(b, L, &c, &tf)) return false;
                if (tf == 0) return true;
                int16_t tfid;
                if (!be_read_i16(b, L, &c, &tfid)) return false;
                if (tf == 12 && tfid == 7) {       // sd: StorageDescriptor
                    while (true) {
                        uint8_t sf;
                        if (!be_read_u8(b, L, &c, &sf)) return false;
                        if (sf == 0) break;
                        int16_t sfid;
                        if (!be_read_i16(b, L, &c, &sfid)) return false;
                        if (sf == 11 && sfid == 2) {  // location: STRING
                            return hms_decode_string(b, L, &c, out, cap);
                        }
                        if (!tb_skip_value(b, L, &c, sf)) return false;
                    }
                } else if (!tb_skip_value(b, L, &c, tf)) {
                    return false;
                }
            }
        } else if (!tb_skip_value(b, L, &c, ft)) {
            return false;
        }
    }
}

int vt_list_namespaces(void* impl_v, CatalogName* out, uint32_t cap,
                       uint32_t* out_n) noexcept {
    assert(impl_v != nullptr);
    assert(out != nullptr && out_n != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    uint8_t args[8];
    uint32_t alen = build_args_empty(args);
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "get_all_databases", args, alen, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    if (!parse_reply_list_of_string(reply, rlen, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

int vt_list_tables(void* impl_v, const char* ns, CatalogName* out,
                   uint32_t cap, uint32_t* out_n) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    uint8_t args[1024];
    uint32_t alen = build_args_one_string(args, sizeof(args), 1, ns);
    if (alen == 0) return kCatBadArg;
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "get_all_tables", args, alen, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    if (!parse_reply_list_of_string(reply, rlen, out, cap, out_n))
        return kCatIoError;
    return kCatOk;
}

int hms_get_table_internal(Impl* impl, const char* ns, const char* name,
                           char* out, uint32_t cap) noexcept {
    assert(impl != nullptr && ns != nullptr && name != nullptr);
    assert(out != nullptr && cap > 0u);
    uint8_t args[2048];
    uint32_t alen = hms_build_get_table_args(args, sizeof(args), ns, name);
    if (alen == 0u) return kCatBadArg;
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "get_table", args, alen, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    uint32_t probe = 0;
    if (!tb_parse_reply(reply, rlen, &probe)) return kCatNotFound;
    if (!parse_get_table_location(reply, rlen, out, cap)) return kCatIoError;
    return kCatOk;
}

int vt_table_path(void* impl_v, const char* ns, const char* name, char* out,
                  uint32_t cap) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr && out != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    return hms_get_table_internal(impl, ns, name, out, cap);
}

int vt_create_table(void* impl_v, const char* ns, const char* name,
                    TableLayout layout) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)impl_v; (void)ns; (void)name; (void)layout;
    // TODO(W7+): full Table struct (sd, partitionKeys, parameters).
    return kCatNotImplemented;
}

int vt_drop_table(void* impl_v, const char* ns, const char* name) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    Impl* impl = static_cast<Impl*>(impl_v);
    // drop_table(db, table, deleteData=true)
    uint8_t args[1024];
    uint32_t c = 0;
    uint32_t dbn = static_cast<uint32_t>(std::strlen(ns));
    uint32_t tn  = static_cast<uint32_t>(std::strlen(name));
    if (!be_write_u8(args, sizeof(args), &c, 11)) return kCatBadArg;
    if (!be_write_i16(args, sizeof(args), &c, 1)) return kCatBadArg;
    if (!be_write_i32(args, sizeof(args), &c, static_cast<int32_t>(dbn))) return kCatBadArg;
    if (c + dbn > sizeof(args)) return kCatBadArg;
    std::memcpy(args + c, ns, dbn); c += dbn;
    if (!be_write_u8(args, sizeof(args), &c, 11)) return kCatBadArg;
    if (!be_write_i16(args, sizeof(args), &c, 2)) return kCatBadArg;
    if (!be_write_i32(args, sizeof(args), &c, static_cast<int32_t>(tn))) return kCatBadArg;
    if (c + tn > sizeof(args)) return kCatBadArg;
    std::memcpy(args + c, name, tn); c += tn;
    if (!be_write_u8(args, sizeof(args), &c, 2)) return kCatBadArg;   // BOOL
    if (!be_write_i16(args, sizeof(args), &c, 3)) return kCatBadArg;
    if (!be_write_u8(args, sizeof(args), &c, 1)) return kCatBadArg;   // true
    if (!be_write_u8(args, sizeof(args), &c, 0)) return kCatBadArg;   // STOP
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "drop_table", args, c, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    uint32_t rc = 0;
    if (!tb_parse_reply(reply, rlen, &rc)) return kCatIoError;
    return kCatOk;
}

int vt_commit_metadata(void* impl_v, const char* ns, const char* name,
                       const char* rel, const uint8_t* d, uint64_t n) noexcept {
    assert(impl_v != nullptr);
    assert(ns != nullptr && name != nullptr);
    (void)impl_v; (void)ns; (void)name; (void)rel; (void)d; (void)n;
    return kCatNotImplemented;
}

const CatalogVT kVT = {
    vt_list_namespaces, vt_list_tables, vt_table_path,
    vt_create_table,    vt_drop_table,  vt_commit_metadata,
};

}  // namespace

bool open(Catalog* out, bolt::Arena* arena, const Config* cfg) noexcept {
    assert(out != nullptr);
    assert(arena != nullptr && cfg != nullptr);
    Impl* impl = static_cast<Impl*>(arena->allocate(sizeof(Impl), alignof(Impl)));
    if (impl == nullptr) return false;
    std::memset(impl, 0, sizeof(*impl));
    impl->cfg = *cfg;
    impl->arena = arena;
    out->vt = &kVT;
    out->impl = impl;
    return true;
}

bool hms_encode_string_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                             int16_t fid, const char* s, uint32_t n) noexcept {
    assert(buf != nullptr);
    assert(cursor != nullptr && s != nullptr);
    if (!be_write_u8(buf, cap, cursor, 11)) return false;
    if (!be_write_i16(buf, cap, cursor, fid)) return false;
    if (!be_write_i32(buf, cap, cursor, static_cast<int32_t>(n))) return false;
    if (*cursor + n > cap) return false;
    std::memcpy(buf + *cursor, s, n);
    *cursor += n;
    return true;
}

bool hms_encode_struct_stop(uint8_t* buf, uint32_t cap, uint32_t* cursor) noexcept {
    assert(buf != nullptr);
    assert(cursor != nullptr);
    return be_write_u8(buf, cap, cursor, 0);
}

bool hms_encode_i32_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                          int16_t fid, int32_t value) noexcept {
    assert(buf != nullptr && cursor != nullptr);
    if (!be_write_u8(buf, cap, cursor, 8)) return false;     // I32
    if (!be_write_i16(buf, cap, cursor, fid)) return false;
    return be_write_i32(buf, cap, cursor, value);
}

bool hms_encode_bool_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                           int16_t fid, bool value) noexcept {
    assert(buf != nullptr && cursor != nullptr);
    if (!be_write_u8(buf, cap, cursor, 2)) return false;     // BOOL
    if (!be_write_i16(buf, cap, cursor, fid)) return false;
    return be_write_u8(buf, cap, cursor, value ? 1u : 0u);
}

bool hms_encode_string_list_field(uint8_t* buf, uint32_t cap, uint32_t* cursor,
                                  int16_t fid, const char* const* items,
                                  uint32_t n) noexcept {
    assert(buf != nullptr && cursor != nullptr);
    assert(items != nullptr || n == 0u);
    if (!be_write_u8(buf, cap, cursor, 15)) return false;    // LIST
    if (!be_write_i16(buf, cap, cursor, fid)) return false;
    if (!be_write_u8(buf, cap, cursor, 11)) return false;    // element=STRING
    if (!be_write_i32(buf, cap, cursor, static_cast<int32_t>(n))) return false;
    for (uint32_t i = 0; i < n; ++i) {
        assert(items[i] != nullptr);
        uint32_t L = static_cast<uint32_t>(std::strlen(items[i]));
        if (!be_write_i32(buf, cap, cursor, static_cast<int32_t>(L))) return false;
        if (*cursor + L > cap) return false;
        std::memcpy(buf + *cursor, items[i], L);
        *cursor += L;
    }
    return true;
}

uint32_t hms_build_get_table_args(uint8_t* buf, uint32_t cap, const char* db,
                                  const char* table) noexcept {
    assert(buf != nullptr && db != nullptr && table != nullptr);
    uint32_t c = 0;
    if (!hms_encode_string_field(buf, cap, &c, 1, db,
                                 static_cast<uint32_t>(std::strlen(db))))
        return 0u;
    if (!hms_encode_string_field(buf, cap, &c, 2, table,
                                 static_cast<uint32_t>(std::strlen(table))))
        return 0u;
    if (!hms_encode_struct_stop(buf, cap, &c)) return 0u;
    return c;
}

uint32_t hms_build_drop_partition_args(uint8_t* buf, uint32_t cap,
                                       const char* db, const char* table,
                                       const char* const* values,
                                       uint32_t n_cols,
                                       bool delete_data) noexcept {
    assert(buf != nullptr && db != nullptr && table != nullptr);
    assert(values != nullptr || n_cols == 0u);
    uint32_t c = 0;
    if (!hms_encode_string_field(buf, cap, &c, 1, db,
                                 static_cast<uint32_t>(std::strlen(db))))
        return 0u;
    if (!hms_encode_string_field(buf, cap, &c, 2, table,
                                 static_cast<uint32_t>(std::strlen(table))))
        return 0u;
    if (!hms_encode_string_list_field(buf, cap, &c, 3, values, n_cols))
        return 0u;
    if (!hms_encode_bool_field(buf, cap, &c, 4, delete_data)) return 0u;
    if (!hms_encode_struct_stop(buf, cap, &c)) return 0u;
    return c;
}

bool hms_decode_string(const uint8_t* buf, uint32_t len, uint32_t* cursor,
                       char* out, uint32_t cap) noexcept {
    assert(buf != nullptr);
    assert(cursor != nullptr && out != nullptr);
    int32_t n;
    if (!be_read_i32(buf, len, cursor, &n) || n < 0) return false;
    if (*cursor + static_cast<uint32_t>(n) > len) return false;
    uint32_t take = static_cast<uint32_t>(n);
    if (take + 1u > cap) take = cap - 1u;
    std::memcpy(out, buf + *cursor, take);
    out[take] = '\0';
    *cursor += static_cast<uint32_t>(n);
    return true;
}

bool hms_decode_field_header(const uint8_t* buf, uint32_t len, uint32_t* cursor,
                             uint8_t* out_type, int16_t* out_fid) noexcept {
    assert(buf != nullptr);
    assert(cursor != nullptr && out_type != nullptr && out_fid != nullptr);
    if (!be_read_u8(buf, len, cursor, out_type)) return false;
    if (*out_type == 0) { *out_fid = 0; return true; }
    return be_read_i16(buf, len, cursor, out_fid);
}

// ===========================================================================
// High-level surface (G2CLUS-179).
// ===========================================================================

namespace {

Impl* himpl(Catalog* cat) noexcept {
    assert(cat != nullptr && cat->vt == &kVT && cat->impl != nullptr);
    return static_cast<Impl*>(cat->impl);
}

}  // namespace

int hms_get_table(Catalog* cat, const char* db, const char* table,
                  char* out_loc, uint32_t cap) noexcept {
    assert(cat != nullptr && db != nullptr && table != nullptr);
    assert(out_loc != nullptr && cap > 0u);
    return hms_get_table_internal(himpl(cat), db, table, out_loc, cap);
}

int hms_alter_table_location(Catalog* cat, const char* db, const char* table,
                             const char* new_location) noexcept {
    assert(cat != nullptr && db != nullptr && table != nullptr);
    assert(new_location != nullptr);
    Impl* impl = himpl(cat);
    // alter_table(dbname, tbl_name, new_tbl). The Table struct carries
    // tableName=1, dbName=2, sd=7{location=2}. We send a minimal Table with
    // just the renamed location so HMS rewrites sd.location.
    uint8_t args[4096];
    uint32_t c = 0;
    if (!hms_encode_string_field(args, sizeof(args), &c, 1, db,
                                 static_cast<uint32_t>(std::strlen(db))))
        return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 2, table,
                                 static_cast<uint32_t>(std::strlen(table))))
        return kCatBadArg;
    // field 3: new_tbl (Table STRUCT).
    if (!be_write_u8(args, sizeof(args), &c, 12)) return kCatBadArg;
    if (!be_write_i16(args, sizeof(args), &c, 3)) return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 1, table,
                                 static_cast<uint32_t>(std::strlen(table))))
        return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 2, db,
                                 static_cast<uint32_t>(std::strlen(db))))
        return kCatBadArg;
    // sd (field 7) STRUCT with location (field 2).
    if (!be_write_u8(args, sizeof(args), &c, 12)) return kCatBadArg;
    if (!be_write_i16(args, sizeof(args), &c, 7)) return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 2, new_location,
                                 static_cast<uint32_t>(std::strlen(new_location))))
        return kCatBadArg;
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // sd
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // new_tbl
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // args
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "alter_table", args, c, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    uint32_t rc = 0;
    if (!tb_parse_reply(reply, rlen, &rc)) return kCatIoError;
    return kCatOk;
}

int hms_add_partition(Catalog* cat, const char* db, const char* table,
                      const char* const* values, uint32_t n_cols,
                      const char* location) noexcept {
    assert(cat != nullptr && db != nullptr && table != nullptr);
    assert(values != nullptr || n_cols == 0u);
    assert(location != nullptr);
    Impl* impl = himpl(cat);
    // add_partition(Partition): Partition{values=1 (list<string>), dbName=2,
    // tableName=3, sd=4{location=2}}.
    uint8_t args[8192];
    uint32_t c = 0;
    if (!be_write_u8(args, sizeof(args), &c, 12)) return kCatBadArg;   // arg=STRUCT
    if (!be_write_i16(args, sizeof(args), &c, 1)) return kCatBadArg;
    if (!hms_encode_string_list_field(args, sizeof(args), &c, 1, values, n_cols))
        return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 2, db,
                                 static_cast<uint32_t>(std::strlen(db))))
        return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 3, table,
                                 static_cast<uint32_t>(std::strlen(table))))
        return kCatBadArg;
    if (!be_write_u8(args, sizeof(args), &c, 12)) return kCatBadArg;   // sd=STRUCT
    if (!be_write_i16(args, sizeof(args), &c, 4)) return kCatBadArg;
    if (!hms_encode_string_field(args, sizeof(args), &c, 2, location,
                                 static_cast<uint32_t>(std::strlen(location))))
        return kCatBadArg;
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // sd
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // Partition
    if (!hms_encode_struct_stop(args, sizeof(args), &c)) return kCatBadArg;  // args
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "add_partition", args, c, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    uint32_t rc = 0;
    if (!tb_parse_reply(reply, rlen, &rc)) return kCatIoError;
    return kCatOk;
}

int hms_drop_partition(Catalog* cat, const char* db, const char* table,
                       const char* const* values, uint32_t n_cols,
                       bool delete_data) noexcept {
    assert(cat != nullptr && db != nullptr && table != nullptr);
    assert(values != nullptr || n_cols == 0u);
    Impl* impl = himpl(cat);
    uint8_t args[4096];
    uint32_t alen = hms_build_drop_partition_args(args, sizeof(args), db, table,
                                                  values, n_cols, delete_data);
    if (alen == 0u) return kCatBadArg;
    uint8_t reply[65536];
    uint32_t rlen = 0;
    if (hms_call(impl, "drop_partition", args, alen, reply, sizeof(reply),
                 &rlen) != 0) return kCatIoError;
    uint32_t rc = 0;
    if (!tb_parse_reply(reply, rlen, &rc)) return kCatIoError;
    return kCatOk;
}

}  // namespace hive_metastore
}  // namespace lakehouse
}  // namespace bolt
