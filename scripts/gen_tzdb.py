#!/usr/bin/env python3
"""Generate bolt's embedded IANA time-zone database (src/datetime/bolt_tzdb_data.inc).

Reads compiled TZif (v2+) files from a zoneinfo directory: the 64-bit
transition table, the local-time types and the POSIX TZ footer that governs
instants after the last transition. Every zone file is deduplicated by
content; a link (US/Eastern, Japan, ...) is a name pointing at the same
zone record. Java's ZoneId.SHORT_IDS (what Spark resolves "PST", "CTT", ...
through) are added as aliases where no file of that name exists.

Usage: gen_tzdb.py [--zoneinfo /usr/share/zoneinfo] [--out src/datetime/bolt_tzdb_data.inc]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import sys

SKIP_TOP = {"posix", "right", "posixrules", "localtime", "Factory"}
MIN_TRANSITION_UTC = -(1 << 59)

# java.time.ZoneId.SHORT_IDS: Spark parses session/zone ids through it.
JAVA_SHORT_IDS = {
    "ACT": "Australia/Darwin", "AET": "Australia/Sydney",
    "AGT": "America/Argentina/Buenos_Aires", "ART": "Africa/Cairo",
    "AST": "America/Anchorage", "BET": "America/Sao_Paulo",
    "BST": "Asia/Dhaka", "CAT": "Africa/Harare", "CNT": "America/St_Johns",
    "CST": "America/Chicago", "CTT": "Asia/Shanghai",
    "EAT": "Africa/Addis_Ababa", "ECT": "Europe/Paris",
    "IET": "America/Indiana/Indianapolis", "IST": "Asia/Kolkata",
    "JST": "Asia/Tokyo", "MIT": "Pacific/Apia", "NET": "Asia/Yerevan",
    "NST": "Pacific/Auckland", "PLT": "Asia/Karachi",
    "PNT": "America/Phoenix", "PRT": "America/Puerto_Rico",
    "PST": "America/Los_Angeles", "SST": "Pacific/Guadalcanal",
    "VST": "Asia/Ho_Chi_Minh",
}


def read_tzif(data: bytes):
    if data[:4] != b"TZif":
        return None
    version = data[4:5]

    def header(off):
        cnts = struct.unpack(">6l", data[off + 20:off + 44])
        isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt = cnts
        return off + 44, isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt

    p, isutc, isstd, leap, timecnt, typecnt, charcnt = header(0)
    if version in (b"2", b"3", b"4"):
        p += timecnt * 5 + typecnt * 6 + charcnt + leap * 8 + isstd + isutc
        p, isutc, isstd, leap, timecnt, typecnt, charcnt = header(p)
        tsize = 8
    else:
        tsize = 4
    fmt = ">%dq" % timecnt if tsize == 8 else ">%dl" % timecnt
    times = list(struct.unpack(fmt, data[p:p + timecnt * tsize]))
    p += timecnt * tsize
    idx = list(data[p:p + timecnt])
    p += timecnt
    types = []
    for _ in range(typecnt):
        utoff, isdst, abbrind = struct.unpack(">lBB", data[p:p + 6])
        types.append((utoff, isdst, abbrind))
        p += 6
    chars = data[p:p + charcnt]
    p += charcnt + leap * (tsize + 4) + isstd + isutc
    footer = ""
    if tsize == 8 and p < len(data) and data[p:p + 1] == b"\n":
        end = data.index(b"\n", p + 1)
        footer = data[p + 1:end].decode("ascii")
    abbrs = []
    for (_, _, ai) in types:
        e = chars.index(b"\0", ai)
        abbrs.append(chars[ai:e].decode("ascii"))
    keep = [(t, i) for t, i in zip(times, idx) if t >= MIN_TRANSITION_UTC]
    return {
        "times": [t for t, _ in keep], "idx": [i for _, i in keep],
        "types": [(u, d, a) for (u, d, _), a in zip(types, abbrs)],
        "footer": footer,
    }


_NAME = r"(<[^>]+>|[A-Za-z]{3,})"
_OFF = r"([+-]?\d{1,2}(?::\d{1,2}(?::\d{1,2})?)?)"


def parse_hms(s: str) -> int:
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return sign * (parts[0] * 3600 + parts[1] * 60 + parts[2])


def parse_rule_date(s: str):
    time = 7200
    if "/" in s:
        s, t = s.split("/", 1)
        time = parse_hms(t)
    if s.startswith("M"):
        m, w, d = (int(x) for x in s[1:].split("."))
        return (1, m, w, d, 0, time)
    if s.startswith("J"):
        return (2, 0, 0, 0, int(s[1:]), time)
    return (3, 0, 0, 0, int(s), time)


def parse_posix(footer: str):
    """POSIX TZ string -> (std_abbr, std_utoff, dst_abbr|None, dst_utoff, start, end)."""
    m = re.match(_NAME + _OFF + r"(?:" + _NAME + _OFF + r"?(?:,([^,]+),([^,]+))?)?$", footer)
    if not m:
        raise ValueError("unparsed POSIX TZ footer %r" % footer)
    std_abbr, std_off, dst_abbr, dst_off, start, end = m.groups()
    std_utoff = -parse_hms(std_off)
    if dst_abbr is None:
        return (std_abbr.strip("<>"), std_utoff, None, std_utoff, None, None)
    dst_utoff = -parse_hms(dst_off) if dst_off else std_utoff + 3600
    if start is None:
        start, end = "M3.2.0", "M11.1.0"
    return (std_abbr.strip("<>"), std_utoff, dst_abbr.strip("<>"), dst_utoff,
            parse_rule_date(start), parse_rule_date(end))


def walk(root: str):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOP]
        for f in filenames:
            rel = f if rel_dir == "." else rel_dir + "/" + f
            if rel in SKIP_TOP or "." in f or f.startswith("+"):
                continue
            with open(os.path.join(dirpath, f), "rb") as fh:
                data = fh.read()
            if data[:4] == b"TZif":
                out[rel] = data
    return out


def generate(root: str) -> str:
    files = walk(root)
    version = "unknown"
    vpath = os.path.join(root, "+VERSION")
    if os.path.exists(vpath):
        version = open(vpath).read().strip()
    zones, zone_of_hash, names = [], {}, {}
    for name in sorted(files):
        h = hashlib.sha256(files[name]).hexdigest()
        if h not in zone_of_hash:
            parsed = read_tzif(files[name])
            zone_of_hash[h] = len(zones)
            zones.append(parsed)
        names[name] = zone_of_hash[h]
    for short, target in JAVA_SHORT_IDS.items():
        if short not in names and target in names:
            names[short] = names[target]
    abbr_ids, abbrs = {}, []

    def abbr(s):
        if s not in abbr_ids:
            abbr_ids[s] = len(abbrs)
            abbrs.append(s)
        return abbr_ids[s]

    rules, rule_ids = [], {}
    trans_utc, trans_type, types, zone_recs = [], [], [], []
    for z in zones:
        rule = 0xFFFF
        if z["footer"]:
            r = parse_posix(z["footer"])
            key = (abbr(r[0]), r[1], abbr(r[2]) if r[2] else 0xFFFF, r[3], r[4], r[5])
            if key not in rule_ids:
                rule_ids[key] = len(rules)
                rules.append(key)
            rule = rule_ids[key]
        assert len(z["types"]) <= 255
        tb = len(types)
        for (u, d, a) in z["types"]:
            types.append((u, d, abbr(a)))
        initial = 0  # RFC 8536 3.2: type 0 governs instants before the first transition
        zone_recs.append((len(trans_utc), len(z["times"]), tb, len(z["types"]), initial, rule))
        trans_utc.extend(z["times"])
        trans_type.extend(z["idx"])
    assert len(abbrs) < 0xFFFF and len(rules) < 0xFFFF and len(zones) < 0xFFFF
    return emit(version, names, zone_recs, trans_utc, trans_type, types, abbrs, rules)


def blob(strings):
    offs, buf = [], []
    pos = 0
    for s in strings:
        offs.append((pos, len(s)))
        buf.append(s)
        pos += len(s)
    return offs, "".join(buf)


def c_str(s: str, width=100):
    out = []
    for i in range(0, len(s), width):
        out.append('    "%s"' % s[i:i + width])
    return "\n".join(out) if out else '    ""'


def rule_date(d):
    if d is None:
        return "{0, 0, 0, 0, 0, 0}"
    return "{%d, %d, %d, %d, %d, %d}" % d


def emit(version, names, zone_recs, trans_utc, trans_type, types, abbrs, rules):
    sorted_names = sorted(names)
    name_offs, name_blob = blob(sorted_names)
    abbr_offs, abbr_blob = blob(abbrs)
    L = []
    L.append("// GENERATED by scripts/gen_tzdb.py from IANA tzdata %s. Do not edit." % version)
    L.append("// %d names, %d zones, %d transitions, %d rules." %
             (len(sorted_names), len(zone_recs), len(trans_utc), len(rules)))
    L.append("// clang-format off")
    L.append('inline constexpr char kTzdbVersion[] = "%s";' % version)
    L.append("inline constexpr std::uint32_t kZoneNameCount = %d;" % len(sorted_names))
    L.append("inline constexpr std::uint32_t kZoneCount = %d;" % len(zone_recs))
    L.append("inline constexpr std::uint32_t kTransitionCount = %d;" % len(trans_utc))
    L.append("inline constexpr std::uint32_t kTypeCount = %d;" % len(types))
    L.append("inline constexpr std::uint32_t kAbbrCount = %d;" % len(abbrs))
    L.append("inline constexpr std::uint32_t kRuleCount = %d;" % len(rules))
    L.append("inline constexpr char kNameBlob[] =\n%s;" % c_str(name_blob))
    L.append("inline constexpr ZoneNameRec kZoneNames[kZoneNameCount] = {")
    for n, (o, ln) in zip(sorted_names, name_offs):
        L.append("    {%d, %d, %d}," % (o, ln, names[n]))
    L.append("};")
    L.append("inline constexpr ZoneRec kZones[kZoneCount] = {")
    for z in zone_recs:
        L.append("    {%d, %d, %d, %d, %d, %d}," % z)
    L.append("};")
    L.append("inline constexpr std::int64_t kTransUtc[kTransitionCount + 1] = {")
    for i in range(0, len(trans_utc), 6):
        L.append("    " + ", ".join("%dLL" % t for t in trans_utc[i:i + 6]) + ",")
    L.append("    0};")
    L.append("inline constexpr std::uint8_t kTransType[kTransitionCount + 1] = {")
    for i in range(0, len(trans_type), 24):
        L.append("    " + ", ".join(str(t) for t in trans_type[i:i + 24]) + ",")
    L.append("    0};")
    L.append("inline constexpr TypeRec kTypes[kTypeCount] = {")
    for (u, d, a) in types:
        L.append("    {%d, %d, %d}," % (u, d, a))
    L.append("};")
    L.append("inline constexpr char kAbbrBlob[] =\n%s;" % c_str(abbr_blob))
    L.append("inline constexpr AbbrRec kAbbrs[kAbbrCount] = {")
    for (o, ln) in abbr_offs:
        L.append("    {%d, %d}," % (o, ln))
    L.append("};")
    L.append("inline constexpr RuleRec kRules[kRuleCount + 1] = {")
    for (sa, so, da, do, st, en) in rules:
        L.append("    {%d, %d, %d, %d, %s, %s}," % (so, do, sa, da, rule_date(st), rule_date(en)))
    L.append("    {0, 0, 0, 0xFFFF, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}};")
    L.append("// clang-format on")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoneinfo", default="/usr/share/zoneinfo")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "src", "datetime", "bolt_tzdb_data.inc"))
    args = ap.parse_args(argv)
    text = generate(os.path.realpath(args.zoneinfo))
    with open(args.out, "w") as fh:
        fh.write(text)
    print("wrote", args.out, len(text), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
