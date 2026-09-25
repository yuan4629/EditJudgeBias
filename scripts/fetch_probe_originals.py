"""Fetch just enough ORIGINAL images from an HF editing dataset to run the suitability probe.

★ WHY THIS EXISTS RATHER THAN `build_from_hf`
`build_from_hf` downloads whole parquet shards, which is right for ingestion and wrong for a
probe. Measured 2026-07-30 on OmniEdit's dev shard:

    full shard                   1,547 MB   ~94 min at the measured 0.258 MB/s
    src_img column, 2 row groups   244 MB   ~16 min          (200 originals)

The probe screens ORIGINALS -- person mask, face box, skin, single-subject. It never opens the
edited image (alignment checking is off for the deterministic arm). So pulling `edited_img` at
all is pure waste, and pulling all 700 rows is waste on top of that: 200 originals already
separates a usable corpus (yield >=5%) from an unusable one (the incumbent pools measure
0.43%), which is the only question the probe has to answer.

Parquet is columnar and row-group addressable, so this reads exactly the wanted column chunks
over HTTP and nothing else.

⚠️ THE MANIFEST IT WRITES IS PROBE-ONLY AND ITS `edited_image_path` DOES NOT EXIST ON DISK.
Every record is stamped `metadata.probe_only: true` and `metadata.edited_image_fetched: false`
so a later reader cannot mistake it for an ingested source. Do not feed it to the judges.

    PYTHONPATH=src python scripts/fetch_probe_originals.py \\
        --repo TIGER-Lab/OmniEdit-Filtered-1.2M \\
        --file data/dev-00000-of-00001.parquet \\
        --row-groups 2 --label omniedit \\
        --id-col omni_edit_id --instruction-col edited_prompt_list --image-col src_img
"""

from __future__ import annotations

import argparse
import io as _io
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

from edit_judge_bias.data import io
from edit_judge_bias.data.build_from_hf import instruction_text
from edit_judge_bias.data.edit_type_rules import (
    classify_content_category,
    classify_edit_type,
)
from edit_judge_bias.data.manifest_utils import default_root, slug
from edit_judge_bias.data.schema import SampleMetadata, SampleRecord


def fetch(
    *,
    label: str,
    repo: Optional[str] = None,
    file: Optional[str] = None,
    local_parquet: Optional[str] = None,
    row_groups: int = 2,
    id_col: str = "omni_edit_id",
    instruction_col: str = "edited_prompt_list",
    image_col: str = "src_img",
    task_col: Optional[str] = "task",
    root: Optional[Path] = None,
    out_manifest: Optional[str] = None,
    out_dir: Optional[str] = None,
    default_edit_type: str = "replace",
    append: bool = False,
) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    root = Path(root) if root is not None else default_root()
    out_dir = out_dir or f"data/images/original/{slug(label)}_probe"
    out_manifest = out_manifest or f"data/manifests/samples_{slug(label)}_probe.jsonl"
    image_root = root / out_dir
    image_root.mkdir(parents=True, exist_ok=True)

    # Multiple local shards: accept a glob or a comma-separated list. Row groups are counted
    # PER SHARD, so `--row-groups 0` (all) is the sane setting for a multi-shard run.
    if local_parquet and ("," in str(local_parquet) or "*" in str(local_parquet)):
        import glob as _glob
        parts = ([p for chunk in str(local_parquet).split(",") for p in _glob.glob(chunk.strip())]
                 if "*" in str(local_parquet)
                 else [p.strip() for p in str(local_parquet).split(",")])
        parts = sorted(set(parts))
        merged = {"source": f"{len(parts)} local shards", "row_groups_read": 0,
                  "of_total_row_groups": 0, "rows_available": 0, "originals_written": 0,
                  "manifest_rows_skipped_as_existing": 0,
                  "planned_bytes_mb": 0.0, "elapsed_s": 0.0, "shards": []}
        for part in parts:
            r = fetch(label=label, local_parquet=part, row_groups=row_groups,
                      id_col=id_col, instruction_col=instruction_col, image_col=image_col,
                      task_col=task_col, root=root, out_manifest=out_manifest,
                      out_dir=out_dir, default_edit_type=default_edit_type, append=True)
            for k in ("row_groups_read","of_total_row_groups","rows_available",
                      "originals_written","manifest_rows_skipped_as_existing",
                      "planned_bytes_mb","elapsed_s"):
                merged[k] += r[k]
            merged["shards"].append({"file": Path(part).name,
                                     "originals": r["originals_written"]})
        merged["out_manifest"] = out_manifest or f"data/manifests/samples_{slug(label)}_probe.jsonl"
        merged["out_dir"] = out_dir or f"data/images/original/{slug(label)}_probe"
        merged["note"] = "PROBE ONLY -- edited images deliberately not fetched"
        return merged

    if local_parquet:
        # ★ A manually-downloaded shard. Measured 2026-07-30, HF pulls at 0.093-0.258 MB/s from
        # here (hf-mirror.com was slower still, so it is general bandwidth, not HF throttling),
        # which put the 1,547 MB OmniEdit shard at ~94-261 min. Downloading it out-of-band
        # through a fast proxy and pointing at the file is strictly better than any clever
        # bounded-read trick, and it also gets the `edited_img` column that a bounded read skips.
        source = str(Path(local_parquet))
        pf = pq.ParquetFile(source)
    else:
        if not (repo and file):
            raise ValueError("give either --local-parquet, or both --repo and --file")
        fs = HfFileSystem()
        source = f"hf://datasets/{repo}/{file}"
        pf = pq.ParquetFile(fs.open(f"datasets/{repo}/{file}", "rb"))
    n_groups = (pf.metadata.num_row_groups if not row_groups
                else min(row_groups, pf.metadata.num_row_groups))

    wanted = [c for c in (id_col, instruction_col, image_col, task_col) if c]
    planned_bytes = sum(
        pf.metadata.row_group(i).column(j).total_compressed_size
        for i in range(n_groups)
        for j in range(pf.metadata.row_group(i).num_columns)
        if pf.metadata.row_group(i).column(j).path_in_schema.split(".")[0] in wanted
    )

    records: List[SampleRecord] = []
    started = time.time()
    for gi in range(n_groups):
        table = pf.read_row_group(gi, columns=wanted)
        rows = table.to_pylist()
        for row in rows:
            raw_id = str(row.get(id_col) or "").strip()
            instruction = instruction_text(row.get(instruction_col))
            cell = row.get(image_col)
            if not raw_id or not instruction or not cell:
                continue
            data = cell.get("bytes") if isinstance(cell, dict) else None
            if not data:
                continue
            sample_id = f"{slug(label)}_{slug(raw_id)}"
            path = image_root / f"{sample_id}.jpg"
            if not path.exists():
                with Image.open(_io.BytesIO(data)) as im:
                    im.convert("RGB").save(path, quality=95)

            edit_type = classify_edit_type(instruction) or default_edit_type
            records.append(SampleRecord(
                sample_id=sample_id,
                source_dataset=label,
                edit_type=edit_type,
                content_category=classify_content_category(instruction),
                original_image_path=path.relative_to(root).as_posix(),
                instruction=instruction,
                edit_model=f"{slug(label)}_probe",
                # ⚠️ NOT FETCHED. Recorded so the record is well-formed; the flags below are
                # what a reader must check before trusting this path.
                edited_image_path=f"data/images/edited/{slug(label)}_probe/{sample_id}.jpg",
                metadata=SampleMetadata(
                    probe_only=True,
                    edited_image_fetched=False,
                    source_repo=repo or "local",
                    # POSIX-normalised on purpose: `fetch_edited_images` groups the pool by
                    # `Path(source_file).name` to open each shard once. A Windows-separator
                    # string round-trips through a POSIX `Path` as ONE segment, so every shard
                    # lookup would miss and the whole pool would report as `missing`. Silent on
                    # Windows, fatal on Linux/CI -- so the manifest is written separator-free.
                    source_file=file or Path(str(local_parquet)).as_posix(),
                    source_row_groups=n_groups,
                    raw_task=str(row.get(task_col)) if task_col else None,
                ),
            ))

    skipped_existing = 0
    if append:
        # ★ IDEMPOTENT APPEND. A multi-shard run appends once per shard, so re-running the same
        # command -- the natural thing to do after adding a shard, or after an interrupted run --
        # would otherwise write every earlier row a second time. The images are already skipped
        # by the `path.exists()` check above, so the duplication would be manifest-only and
        # therefore invisible on disk: the pool build dedupes by `sample_id` and would report the
        # same counts while every downstream row count silently doubled.
        manifest_path = root / out_manifest
        existing = ({r.sample_id for r in io.iter_jsonl(manifest_path, SampleRecord)}
                    if manifest_path.exists() else set())
        fresh = [r for r in records if r.sample_id not in existing]
        skipped_existing = len(records) - len(fresh)
        for rec in fresh:
            io.append_jsonl(manifest_path, rec)
    else:
        io.write_jsonl(root / out_manifest, records)
    elapsed = time.time() - started
    return {
        "source": source,
        "row_groups_read": n_groups,
        "of_total_row_groups": pf.metadata.num_row_groups,
        "rows_available": sum(pf.metadata.row_group(i).num_rows for i in range(n_groups)),
        "originals_written": len(records),
        "manifest_rows_skipped_as_existing": skipped_existing,
        "planned_bytes_mb": round(planned_bytes / 1e6, 1),
        "elapsed_s": round(elapsed, 1),
        "effective_mb_s": round(planned_bytes / 1e6 / elapsed, 3) if elapsed else None,
        "out_manifest": out_manifest,
        "out_dir": out_dir,
        "note": "PROBE ONLY -- edited images deliberately not fetched",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--local-parquet", default=None,
                    help="path to a manually-downloaded shard (skips HF entirely)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--row-groups", type=int, default=2)
    ap.add_argument("--id-col", default="omni_edit_id")
    ap.add_argument("--instruction-col", default="edited_prompt_list")
    ap.add_argument("--image-col", default="src_img")
    ap.add_argument("--task-col", default="task")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    report = fetch(
        repo=a.repo, file=a.file, local_parquet=a.local_parquet,
        label=a.label, row_groups=a.row_groups,
        id_col=a.id_col, instruction_col=a.instruction_col, image_col=a.image_col,
        task_col=a.task_col, root=Path(a.root) if a.root else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
