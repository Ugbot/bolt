import re, sys
src = open(sys.argv[1]).read().splitlines()
def block(name):
    i = next(k for k, l in enumerate(src) if f"val {name} = Array(" in l)
    rows = []
    for l in src[i+1:]:
        s = l.strip()
        if s == ")": break
        if s.startswith("Array("):
            body = s[len("Array("):s.rindex(")")]
            rows.append([t.strip() for t in body.split(",") if t.strip()])
    assert len(rows) == 15, (name, len(rows))
    return rows
raw, bias = block("RAW_ESTIMATE_DATA"), block("BIAS_DATA")
for r, b in zip(raw, bias): assert len(r) == len(b)
def lit(t):
    return t if ("." in t or "e" in t.lower()) else t + ".0"
out = ["// bolt_hll_spark_tables.h -- generated from Spark 4.0.1 HyperLogLogPlusPlusHelper.scala",
       "// (RAW_ESTIMATE_DATA, BIAS_DATA, THRESHOLDS; precisions 4..18) by tools/gen_hll_tables.py.",
       "// Do not edit by hand: the estimator must equal Spark's bit for bit.",
       "#pragma once", "", "#include <cstdint>", "", "namespace bolt::kernels::hll::spark_tables {", ""]
def emit(name, rows):
    for p, r in enumerate(rows, start=4):
        out.append(f"inline constexpr double k_{name}_p{p}[] = {{")
        for k in range(0, len(r), 8):
            out.append("    " + ", ".join(lit(t) for t in r[k:k+8]) + ",")
        out.append("};")
emit("raw", raw); emit("bias", bias)
out.append("")
out.append("inline constexpr const double* k_raw[15] = {" + ", ".join(f"k_raw_p{p}" for p in range(4, 19)) + "};")
out.append("inline constexpr const double* k_bias[15] = {" + ", ".join(f"k_bias_p{p}" for p in range(4, 19)) + "};")
out.append("inline constexpr std::uint32_t k_len[15] = {" + ", ".join(str(len(r)) for r in raw) + "};")
thr = re.search(r"THRESHOLDS = Array\[Double\]\(([^)]*)\)", "\n".join(src)).group(1)
out.append("inline constexpr double k_thresholds[15] = {" + ", ".join(lit(t.strip()) for t in thr.split(",")) + "};")
out += ["", "}  // namespace bolt::kernels::hll::spark_tables", ""]
open(sys.argv[2], "w").write("\n".join(out))
