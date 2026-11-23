// tests/otlp_test_pb.h — a minimal protobuf writer and hand-built OTLP requests
// (shared with Gestalt2's writer tests).
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace otlp_test {

// A minimal protobuf writer for the test's requests.
struct Pb {
    std::string b;
    void varint(std::uint64_t v) { while (v >= 0x80) { b.push_back(static_cast<char>(v | 0x80)); v >>= 7; } b.push_back(static_cast<char>(v)); }
    void tag(std::uint32_t f, std::uint32_t w) { varint((static_cast<std::uint64_t>(f) << 3) | w); }
    void len(std::uint32_t f, const std::string& s) { tag(f, 2); varint(s.size()); b += s; }
    void u64(std::uint32_t f, std::uint64_t v) { tag(f, 0); varint(v); }
    void fixed64(std::uint32_t f, std::uint64_t v) { tag(f, 1); b.append(reinterpret_cast<const char*>(&v), 8); }
};

inline std::string any_str(const std::string& s) { Pb p; p.len(1, s); return p.b; }
inline std::string any_int(std::int64_t v) { Pb p; p.u64(3, static_cast<std::uint64_t>(v)); return p.b; }
inline std::string any_bool(bool v) { Pb p; p.u64(2, v ? 1 : 0); return p.b; }
inline std::string any_dbl(double d) { Pb p; std::uint64_t bits; std::memcpy(&bits, &d, 8); p.fixed64(4, bits); return p.b; }
inline std::string any_arr(const std::vector<std::string>& items) { Pb a; for (const auto& i : items) a.len(1, i); Pb p; p.len(5, a.b); return p.b; }
inline std::string kv(const std::string& k, const std::string& any) { Pb p; p.len(1, k); p.len(2, any); return p.b; }
inline std::string any_kvs(const std::vector<std::string>& kvs) { Pb l; for (const auto& x : kvs) l.len(1, x); Pb p; p.len(6, l.b); return p.b; }

namespace logs {

struct Rec {
    std::uint64_t time = 0, observed = 0, severity = 0;
    std::string severity_text, body_any, trace, span, event;
    std::vector<std::string> attrs;
};

inline std::string record(const Rec& r) {
    Pb p;
    if (r.time) p.fixed64(1, r.time);
    if (r.severity) p.u64(2, r.severity);
    if (!r.severity_text.empty()) p.len(3, r.severity_text);
    if (!r.body_any.empty()) p.len(5, r.body_any);
    for (const auto& a : r.attrs) p.len(6, a);
    if (!r.trace.empty()) p.len(9, r.trace);
    if (!r.span.empty()) p.len(10, r.span);
    if (r.observed) p.fixed64(11, r.observed);
    if (!r.event.empty()) p.len(12, r.event);
    return p.b;
}

inline std::string request(const std::vector<std::string>& resource_attrs, const std::string& scope_name,
                    const std::string& scope_version, const std::vector<Rec>& recs) {
    Pb res; for (const auto& a : resource_attrs) res.len(1, a);
    Pb scope; if (!scope_name.empty()) scope.len(1, scope_name); if (!scope_version.empty()) scope.len(2, scope_version);
    Pb sl; if (!scope.b.empty()) sl.len(1, scope.b);
    for (const auto& r : recs) sl.len(2, record(r));
    Pb rl; rl.len(1, res.b); rl.len(2, sl.b);
    Pb req; req.len(1, rl.b);
    return req.b;
}

inline const std::string kTrace("\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10", 16);
inline const std::string kSpan("\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8", 8);

inline std::string sample_request() {
    Rec a; a.time = 1700000000000000000ull; a.observed = 1700000000500000000ull; a.severity = 9; a.severity_text = "INFO";
    a.body_any = any_str("order placed \"ok\"\n"); a.trace = kTrace; a.span = kSpan; a.event = "order.placed";
    a.attrs = {kv("http.status_code", any_int(200)), kv("ok", any_bool(true)), kv("ratio", any_dbl(0.5)),
               kv("tags", any_arr({any_str("a"), any_int(2)})), kv("nested", any_kvs({kv("k", any_str("v"))}))};
    Rec b; b.observed = 1700000001000000000ull; b.severity = 17; b.body_any = any_kvs({kv("msg", any_str("boom"))});
    b.trace = std::string(16, '\0'); b.span = "short";
    Rec c;   // nothing set
    return request({kv("service.name", any_str("checkout")), kv("host.name", any_str("h1")), kv("process.pid", any_int(42))},
                   "io.demo", "1.2", {a, b, c});
}

}  // namespace logs

namespace spans {

inline const std::string kTrace("\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10", 16);
inline const std::string kRoot("\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8", 8);
inline const std::string kChild("\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8", 8);

struct Span {
    std::string trace = kTrace, id, parent, name, state, status_msg;
    std::uint64_t start = 0, end = 0, kind = 0, code = 0;
    std::vector<std::string> attrs, events, links;
};

inline std::string span(const Span& s) {
    Pb p;
    p.len(1, s.trace); p.len(2, s.id);
    if (!s.state.empty()) p.len(3, s.state);
    if (!s.parent.empty()) p.len(4, s.parent);
    p.len(5, s.name);
    if (s.kind) p.u64(6, s.kind);
    p.fixed64(7, s.start); p.fixed64(8, s.end);
    for (const auto& a : s.attrs) p.len(9, a);
    for (const auto& e : s.events) p.len(11, e);
    for (const auto& l : s.links) p.len(13, l);
    if (s.code || !s.status_msg.empty()) { Pb st; if (!s.status_msg.empty()) st.len(2, s.status_msg); if (s.code) st.u64(3, s.code); p.len(15, st.b); }
    return p.b;
}

inline std::string event(std::uint64_t ts, const std::string& name, const std::vector<std::string>& attrs) {
    Pb p; p.fixed64(1, ts); p.len(2, name); for (const auto& a : attrs) p.len(3, a); return p.b;
}
inline std::string link(const std::string& trace, const std::string& sid) { Pb p; p.len(1, trace); p.len(2, sid); return p.b; }

inline std::string request(const std::string& service, const std::vector<Span>& spans) {
    Pb res; res.len(1, kv("service.name", any_str(service)));
    Pb scope; scope.len(1, "io.demo");
    Pb ss; ss.len(1, scope.b);
    for (const auto& s : spans) ss.len(2, span(s));
    Pb rs; rs.len(1, res.b); rs.len(2, ss.b);
    Pb req; req.len(1, rs.b);
    return req.b;
}

inline std::string sample_request() {
    Span root; root.id = kRoot; root.name = "GET /checkout"; root.kind = 2; root.start = 1700000000000000000ull;
    root.end = 1700000000250000000ull; root.attrs = {kv("http.status_code", any_int(500))};
    root.code = 2; root.status_msg = "boom";
    root.events = {event(1700000000100000000ull, "exception", {kv("exception.type", any_str("IOError"))})};
    Span child; child.id = kChild; child.parent = kRoot; child.name = "db.query"; child.kind = 3; child.state = "k=v";
    child.start = 1700000000050000000ull; child.end = 1700000000090000000ull; child.links = {link(kTrace, kRoot)};
    return request("checkout", {root, child});
}

}  // namespace spans

}  // namespace otlp_test
