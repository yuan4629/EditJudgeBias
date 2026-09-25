"""Collect the samples referenced by a pairs manifest into a samples manifest.

The pairwise arm needs *one-sided* bias conditions: show the judge pair (A, B)
where only A carries, say, a padding border, and see whether its verdict moves.
That needs a biased image for each pair member — but a PairRecord is
self-contained (it carries its own image paths and is drawn independently of the
scoring subset), so most pair members are not in `samples_judge_v2.jsonl` and
were never injected.

This writes the missing manifest so `run_bias_injection` can consume it with no
special-casing:

    python scripts/make_pair_member_manifest.py \
        --pairs data/manifests/pairs_judge_v2.jsonl \
        --pool  data/manifests/samples_full_v2.jsonl \
        --out   data/manifests/samples_pair_members_v2.jsonl

Members already present in the scoring subset are kept (the injector skips images
that exist, so listing them costs nothing and keeps the manifest self-describing).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edit_judge_bias.data.io import read_jsonl, write_jsonl  # noqa: E402
from edit_judge_bias.data.schema import PairRecord, SampleRecord  # noqa: E402


def collect(pairs_path: Path, pool_path: Path) -> tuple[list[SampleRecord], Counter]:
    pairs = read_jsonl(pairs_path, PairRecord)
    wanted: list[str] = []
    seen: set[str] = set()
    for p in pairs:  # stable order: A then B, in pair order
        for sid in (p.sample_id_a, p.sample_id_b):
            if sid not in seen:
                seen.add(sid)
                wanted.append(sid)

    by_id = {s.sample_id: s for s in read_jsonl(pool_path, SampleRecord)}
    stats = Counter(n_pairs=len(pairs), n_members=len(wanted))
    out: list[SampleRecord] = []
    for sid in wanted:
        rec = by_id.get(sid)
        if rec is None:
            # A pair whose member is missing from the pool is a build error, not a
            # skip: the pairwise runner would fail mid-batch on the missing image.
            stats["missing"] += 1
            continue
        out.append(rec)
    stats["written"] = len(out)
    return out, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    samples, stats = collect(args.pairs, args.pool)
    if not args.dry_run:
        write_jsonl(args.out, samples)
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"{verb} {stats['written']} pair-member samples from {stats['n_pairs']} pairs "
        f"({stats['n_members']} distinct members) -> {args.out}"
    )
    if stats["missing"]:
        print(f"ERROR: {stats['missing']} members are absent from the pool")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
