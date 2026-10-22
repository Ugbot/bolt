#!/usr/bin/env python3
"""Generate independent S3-reference OCF fixtures using PyIceberg.

Run with the existing UV environment: build/arrow-interop-tests/bin/python
extern/bolt/tests/data/make_iceberg_s3_graph.py. No network or S3 emulation:
PyIceberg's OutputFile carries the real S3 identity and writes to a local path.
No Bolt encoder/decoder participates and emitted bytes are never rewritten.
Requires pyiceberg 0.11.0 and pyarrow 21.0.0 (existing qualification environment).
"""
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID


def generate_at(here: Path) -> None:
    import pyarrow.fs as fs
    import pyiceberg
    import pyarrow
    from pyiceberg.io.pyarrow import PyArrowFile
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat, ManifestEntry, ManifestEntryStatus
    from pyiceberg.manifest import write_manifest, write_manifest_list
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.table.metadata import new_table_metadata
    from pyiceberg.table.snapshots import Snapshot, Summary, Operation
    from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
    from pyiceberg.typedef import Record
    from pyiceberg.types import NestedField, LongType

    root="s3://graph-bucket/root%20space"
    table=f"{root}/tables/orders"
    paths={name:here/f"golden_iceberg_s3_graph_{name}" for name in
           ("metadata.json", "manifest.avro", "manifest_list.avro")}
    manifest_uri=f"{root}/external/manifest.avro"
    list_uri=f"{root}/external/manifest-list.avro"
    localfs=fs.LocalFileSystem()
    def output(uri: str, path: Path):
        return PyArrowFile(location=uri,path=str(path),fs=localfs)
    schema=Schema(NestedField(1,"value",LongType(),required=True))
    spec=PartitionSpec()
    data=DataFile.from_args(content=DataFileContent.DATA,file_path=f"{root}/external/data.parquet",
        file_format=FileFormat.PARQUET,partition=Record(),record_count=3,file_size_in_bytes=2048,
        column_sizes={},value_counts={1:3},null_value_counts={1:0},nan_value_counts={},
        lower_bounds={1:(11).to_bytes(8,"little",signed=True)},
        upper_bounds={1:(33).to_bytes(8,"little",signed=True)},key_metadata=None,
        split_offsets=[4],equality_ids=None,sort_order_id=0)
    with write_manifest(2,spec,schema,output(manifest_uri,paths["manifest.avro"]),19,"deflate") as writer:
        writer.add(ManifestEntry.from_args(status=ManifestEntryStatus.ADDED,snapshot_id=19,
            sequence_number=None,file_sequence_number=None,data_file=data))
    manifest=writer.to_manifest_file()
    with write_manifest_list(2,output(list_uri,paths["manifest_list.avro"]),19,None,1,"deflate") as writer:
        writer.add_manifests([manifest])
    metadata=new_table_metadata(schema,spec,UNSORTED_SORT_ORDER,table,
        table_uuid=UUID("12345678-1234-1234-1234-123456789abc"))
    snapshot=Snapshot(snapshot_id=19,sequence_number=1,timestamp_ms=1000,
        manifest_list=list_uri,schema_id=0,summary=Summary(operation=Operation.APPEND))
    metadata=metadata.model_copy(update={"current_snapshot_id":19,"snapshots":[snapshot],
        "last_sequence_number":1,"last_updated_ms":1000})
    paths["metadata.json"].write_text(metadata.model_dump_json(by_alias=True),encoding="utf-8")
    receipt={"producer":"PyIceberg", "pyiceberg":pyiceberg.__version__,"pyarrow":pyarrow.__version__,
        "root":root,"table":table,"snapshot_id":19,"manifest_count":1,"data_files":1,
        "data_file_bytes":2048,"data_row_bodies":"not generated or read by graph admission",
        "files":{p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in paths.values()}}
    (here/"golden_iceberg_s3_graph_provenance.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


def generate() -> None:
    # OutputFile defaults to exclusive creation. Generate a fresh complete set
    # before replacing committed fixtures; repeated deliberate regeneration
    # therefore works without teaching the writer to overwrite existing files.
    here=Path(__file__).resolve().parent
    with TemporaryDirectory(prefix="iceberg-s3-graph-") as raw:
        scratch=Path(raw)
        generate_at(scratch)
        for suffix in ("metadata.json","manifest.avro","manifest_list.avro","provenance.json"):
            name=f"golden_iceberg_s3_graph_{suffix}"
            (here/name).write_bytes((scratch/name).read_bytes())


async def main() -> None:
    await asyncio.to_thread(generate)


if __name__=="__main__":
    asyncio.run(main())
