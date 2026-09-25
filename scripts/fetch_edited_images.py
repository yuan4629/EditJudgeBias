"""Fetch the EDITED member for an already-built pool. Local parquet only, zero API cost.

★ WHY THIS IS A SEPARATE STEP FROM `fetch_probe_originals`
The probe screens ORIGINALS -- person mask, face box, skin, single subject -- and never opens
the edited image. Pulling `edited_img` for all 18,958 rows to answer a question about 15,613
originals would have roughly doubled a multi-GB read for nothing, so the probe manifest stamps
`edited_image_fetched: false` and leaves `edited_image_path` pointing at a file that does not
exist. This script fills that in for the scenes a pool actually selected -- 611 of 988, not
18,958 -- after the selection is known.

★ WHY THE EDITED MEMBER IS NEEDED AT ALL
The D-S pair the judge sees is (original_variant, edited_variant), and the manipulation is
applied to BOTH members of an already-existing edit. That is the whole reason this arm is
sound where the 2026-07-30 design was not: the edit happens ONCE -- it is the dataset's own
edited image, shared by both arms -- so generative churn cannot correlate with the arm label.

★ IT VERIFIES THE JOIN RATHER THAN ASSUMING IT
Two failure modes would otherwise be silent, and both produce a pool that looks complete:
  * a sample_id that no shard contains (wrong `--id-col`, wrong shard set) would just be
    missing from the output, shrinking the pool without saying so. `missing` is returned and
    the exit code is non-zero.
  * a row whose `edited_img` is byte-identical to `src_img` is not an edit. Reported as
    `identical_to_original` -- see `--verify-differs`.

    PYTHONPATH=src python scripts/fetch_edited_images.py \\
        --pool data/manifests/samples_fairness_ds_v4.jsonl \\
        --parquet-dir tmp_data/downloads --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io as _io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve, slug
from edit_judge_bias.data.schema import SampleRecord


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_edited(
    *,
    pool_path: str,
    root: Optional[Path] = None,
    parquet_dir: str = "tmp_data/downloads",
    label: str = "omniedit_train",
    id_col: str = "omni_edit_id",
    image_col: str = "edited_img",
    src_col: str = "src_img",
    quality: int = 95,
    dry_run: bool = False,
    verify_differs: bool = True,
    update_manifest: bool = True,
) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image

    root = Path(root) if root is not None else default_root()
    records = io.read_jsonl(root / pool_path, SampleRecord)

    # Group the wanted ids by the shard they came from, so each shard is opened at most once.
    # `metadata.source_file` was stamped at ingest and rides through the pool build, which is
    # what makes a targeted read possible at all.
    by_shard: Dict[str, List[SampleRecord]] = defaultdict(list)
    no_source: List[str] = []
    for rec in records:
        src = getattr(rec.metadata, "source_file", None)
        if not src:
            no_source.append(rec.sample_id)
            continue
        # Separator-agnostic: manifests written on Windows before 2026-08-01 carry a
        # backslash path, which a POSIX `Path` treats as a single segment -- `.name` would
        # then return the whole `dir\shard.parquet` string, miss every shard, and report the
        # entire pool as missing. Normalise instead of trusting the platform.
        by_shard[Path(str(src).replace("\\", "/")).name].append(rec)

    wanted_by_id = {r.sample_id: r for r in records}
    report: dict = {
        "pool": pool_path,
        "scenes": len(records),
        "shards": len(by_shard),
        "records_without_source_file": len(no_source),
    }
    if dry_run:
        report["per_shard"] = {k: len(v) for k, v in sorted(by_shard.items())}
        report["note"] = "dry run: nothing read, nothing written"
        return report

    stats = Counter()
    found: set = set()
    sizes = Counter()
    for shard_name, wanted in sorted(by_shard.items()):
        shard = root / parquet_dir / shard_name
        if not shard.exists():
            stats["shard_missing"] += 1
            report.setdefault("missing_shards", []).append(shard_name)
            continue
        want_ids = {r.sample_id for r in wanted}
        # Every one already on disk -> skip the shard entirely. Resumability without a
        # checkpoint file: the images themselves are the checkpoint.
        if all(resolve(wanted_by_id[i].edited_image_path, root).exists() for i in want_ids):
            stats["shards_skipped_complete"] += 1
            found |= want_ids
            continue

        pf = pq.ParquetFile(shard)
        # Parquet is columnar, so naming only what is used keeps the read proportional to the
        # work. `src_img` is read solely for the identical-bytes check and skipped without it.
        cols = [id_col, image_col] + ([src_col] if verify_differs and src_col else [])
        cols = list(dict.fromkeys(cols))
        for gi in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(gi, columns=cols)
            for row in table.to_pylist():
                raw_id = str(row.get(id_col) or "").strip()
                if not raw_id:
                    continue
                sample_id = f"{slug(label)}_{slug(raw_id)}"
                if sample_id not in want_ids:
                    continue
                found.add(sample_id)
                rec = wanted_by_id[sample_id]
                out = resolve(rec.edited_image_path, root)
                if out.exists():
                    stats["already_on_disk"] += 1
                    continue
                cell = row.get(image_col)
                data = cell.get("bytes") if isinstance(cell, dict) else None
                if not data:
                    stats["no_edited_bytes"] += 1
                    continue
                if verify_differs:
                    src_cell = row.get(src_col)
                    src_data = src_cell.get("bytes") if isinstance(src_cell, dict) else None
                    if src_data and _sha(src_data) == _sha(data):
                        # Not an edit. Recorded rather than written: an unedited "edited"
                        # member would give every judge an identical pair and a meaningless
                        # score, and it would be invisible downstream.
                        stats["identical_to_original"] += 1
                        report.setdefault("identical_ids", []).append(sample_id)
                        continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(_io.BytesIO(data)) as im:
                    edited = im.convert("RGB")
                    with Image.open(resolve(rec.original_image_path, root)) as orig:
                        sizes["same_size" if orig.size == edited.size
                              else "size_mismatch"] += 1
                    edited.save(out, quality=quality)
                stats["written"] += 1

    missing = sorted(set(wanted_by_id) - found)
    report.update({
        **dict(stats),
        "size_agreement": dict(sizes),
        "missing": len(missing),
        "missing_ids": missing[:20],
    })

    if update_manifest and not missing:
        for rec in records:
            if resolve(rec.edited_image_path, root).exists():
                rec.metadata.edited_image_fetched = True
        io.write_jsonl(root / pool_path, records)
        report["manifest_updated"] = pool_path
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--parquet-dir", default="tmp_data/downloads")
    ap.add_argument("--label", default="omniedit_train")
    ap.add_argument("--id-col", default="omni_edit_id")
    ap.add_argument("--image-col", default="edited_img")
    ap.add_argument("--src-col", default="src_img")
    ap.add_argument("--no-verify-differs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    report = fetch_edited(
        pool_path=a.pool, root=Path(a.root) if a.root else None,
        parquet_dir=a.parquet_dir, label=a.label, id_col=a.id_col,
        image_col=a.image_col, src_col=a.src_col,
        verify_differs=not a.no_verify_differs, dry_run=a.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # A short pool is a defect, not a result: fail loudly rather than let the next stage
    # quietly run on fewer scenes than the manifest claims.
    return 1 if report.get("missing") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
