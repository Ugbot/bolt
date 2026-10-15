"""Test-only independent oracle generation, PyArrow 21.0.0. No Bolt imports."""
import hashlib
import json
import struct
from decimal import Decimal
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
assert pa.__version__ == "21.0.0"
HERE = Path(__file__).parent
cols, fields = [], []
def add(name, typ, values):
    fields.append(pa.field(name, typ, nullable=True, metadata={b"field": name.encode()}))
    cols.append(pa.array(values, type=typ))
for bits in (8, 16, 32, 64):
    add("i"+str(bits), getattr(pa, "int"+str(bits))(), [-(1 << (bits-1)), None, (1 << (bits-1))-1])
    add("u"+str(bits), getattr(pa, "uint"+str(bits))(), [0, None, (1 << bits)-1])
fields.append(pa.field("half", pa.float16()))
cols.append(pa.Array.from_buffers(pa.float16(), 3, [None, pa.py_buffer(struct.pack("<HHH", 0x8000, 0x7e55, 0x3c00))]))
add("single", pa.float32(), [-0.0, None, 1.5])
add("double", pa.float64(), [-0.0, None, 1.5])
add("truth", pa.bool_(), [True, None, False])
add("text", pa.string(), ["alpha", None, "a long string with snowman \u2603"])
add("bytes", pa.binary(), [b"\x00\xff", None, b""])
add("money", pa.decimal128(38, 9), [Decimal("-1234567890123456789.123456789"), None, Decimal("99999999999999999999999999999.999999999")])
add("days", pa.date32(), [-1, None, 100])
add("millis", pa.date64(), [-86400000, None, 86400000])
add("clock32", pa.time32("s"), [0, None, 86399])
add("clock64", pa.time64("ns"), [0, None, 86399999999999])
add("epoch_s", pa.timestamp("s", tz="America/New_York"), [-1, None, 123456789])
add("epoch_us", pa.timestamp("us"), [-1, None, 123456789])
add("duration", pa.duration("ns"), [-9223372036854775807, None, 9223372036854775807])
add("fixed", pa.binary(3), [b"abc", None, b"\x00\xffz"])
schema = pa.schema(fields, metadata={b"schema": b"exact\x00metadata"})
batch = pa.record_batch(cols, schema=schema)
for is_file in (False, True):
    path = HERE / ("flat.arrow" if is_file else "flat.arrows")
    with pa.OSFile(str(path), "wb") as sink:
        factory = ipc.new_file if is_file else ipc.new_stream
        with factory(sink, schema, **({"metadata": {b"footer": b"owned"}} if is_file else {})) as writer:
            writer.write_batch(batch)
            writer.write_batch(batch.slice(0, 0))
            writer.write_batch(batch)
for name, options in (("legacy.arrows", ipc.IpcWriteOptions(use_legacy_format=True)),
                      ("v4.arrows", ipc.IpcWriteOptions(metadata_version=ipc.MetadataVersion.V4))):
    with pa.OSFile(str(HERE/name), "wb") as sink:
        with ipc.new_stream(sink, schema, options=options) as writer:
            writer.write_batch(batch)
            writer.write_batch(batch.slice(0, 0))
            writer.write_batch(batch)
# Each unsupported fixture independently exercises a distinct admission gate.
for name, array in (("nested", pa.array([[1,2],None], pa.list_(pa.int32()))),
                    ("dictionary", pa.array(["a","b"]).dictionary_encode()),
                    ("negative_scale", pa.array([Decimal("1E2")], pa.decimal128(3,-2)))):
    b = pa.record_batch([array], names=[name])
    with pa.OSFile(str(HERE / (name+".arrows")), "wb") as sink:
        with ipc.new_stream(sink, b.schema) as writer:
            writer.write_batch(b)
with pa.OSFile(str(HERE / "compressed.arrows"), "wb") as sink:
    with ipc.new_stream(sink, schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
        writer.write_batch(batch)
with pa.OSFile(str(HERE / "empty.arrow"), "wb") as sink:
    with ipc.new_file(sink, pa.schema([])) as writer:
        writer.write_batch(pa.record_batch([], schema=pa.schema([])))
manifest = {"generator": "PyArrow 21.0.0 official Apache distribution; authored values, not captured vendor traffic",
            "format_pin": "apache-arrow-21.0.0", "files": {}}
for p in sorted(HERE.glob("*.arrow*")):
    if p.suffix not in (".arrow", ".arrows"): continue
    data=p.read_bytes()
    manifest["files"][p.name]={"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
