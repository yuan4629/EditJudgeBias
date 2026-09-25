"""Construct-validity screen over D-S pool candidates. One cheap call per ORIGINAL image.

★ WHY A PAID SCREEN IS THE ONLY OPTION HERE
The geometric pool screens are now good: the framing band selects head-and-shoulders single
subjects where the old pool selected gulls and a guitar body. But a contact sheet of a seeded
random sample of 72 in-band OmniEdit scenes found **30 (42%) unusable** on grounds no geometry
can reach -- multi-subject frames and side-by-side collages, non-photographic subjects (CGI,
anime, 3D renders, a statue, an ogre), theatrical face paint, and MINORS whose instructions
never mention a child. Two free screens were built for these and both were rejected on their
own controls:

    "exactly one detected face"   6/8 agreement with hand labels, and BOTH misses were
                                  two-subject scenes read as one -- Haar's failure is RECALL,
                                  so merging cannot fix it and the error direction is the
                                  harmful one (multi-subject scenes pass)
    "the skin region has chroma"  sound (B&W control 7.5 vs colour 22-33) but near-vacuous:
                                  2 of 723, because the skin mask is itself a chroma rule

★ IT MEASURES ITS OWN FALSE-REJECT FLOOR, AND THE FLOOR POINTS THE OTHER WAY FROM USUAL.
The pair validator's control is the same image twice, where the honest answer is "no flip", so
its floor measures rubber-stamping. Here the control is a set of images already judged usable
by eye, where the honest answer is "pass", so the floor measures how often a GOOD scene is
thrown away. Both directions matter and they are not interchangeable: this study has already
had a validator (glm-4v) whose 13.6% false-flag rate disqualified it from gating, and a screen
that rejects everything would look maximally safe while destroying n.

★ THE ETHICS QUESTION IS NOT SYMMETRIC WITH THE OTHERS.
`no_minor` is phrased positively and a missing or unparseable field is treated as NOT
satisfied, so a dropped field can only cost a scene. Over-rejecting shrinks n; under-rejecting
puts a minor into a dataset of deliberately manipulated skin tones. Only one of those is
recoverable, and the screen is built to fail in that direction.

    python -m edit_judge_bias.experiments.run_construct_screen \\
        --config configs/fairness/construct_screen.yaml --dry-run
    python -m edit_judge_bias.experiments.run_construct_screen \\
        --config configs/fairness/construct_screen.yaml --use-api
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve
from edit_judge_bias.data.schema import SampleRecord
from edit_judge_bias.experiments.judge_common import (
    adapter_for as _adapter_for,
    load_failure_log_path,
    log_failure,
    run_jobs,
    save_raw_response,
)
from edit_judge_bias.fairness.records import ConstructScreenResult
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.parser import parse_construct_screen
from edit_judge_bias.prompts.construct_screen_prompt import (
    SCREEN_FIELDS,
    build_construct_screen_prompt,
)

#: Above this false-reject rate on the hand-labelled controls, the screen is a random n-shredder
#: rather than a filter and `passed_sample_ids` REFUSES to hand back a pass set. Same number and
#: same reasoning as `passed_pair_keys()`: a validator too noisy to gate on removes scenes at
#: random, not bad ones. glm-4v's 13.6% is the measured example of failing this.
MAX_FALSE_REJECT = 0.10


@dataclass
class RunStats:
    screened: int = 0
    skipped: int = 0
    parse_failures: int = 0
    api_failures: int = 0

    def summary(self) -> str:
        return (f"screened={self.screened} skipped={self.skipped} "
                f"parse_failures={self.parse_failures} api_failures={self.api_failures}")


def _control_ids(cfg: dict, root: Path) -> List[str]:
    """Sample ids a human already judged USABLE, for the false-reject floor.

    Read from a file rather than inferred, because "a human looked at this and said yes" is
    not a property any code can derive.
    """
    rel = cfg.get("control_ids")
    if not rel:
        return []
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    return list(data if isinstance(data, list) else data.get("usable", []))


def run(
    config_path: str | Path,
    *,
    root: Optional[Path] = None,
    use_api: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    samples = io.read_jsonl(root / cfg["samples"], SampleRecord)
    out_manifest = root / cfg["out_manifest"]
    raw_dir = cfg.get("raw_dir", "results/v2_fairness/raw_responses_construct")
    summary_csv = root / cfg["summary_csv"]
    workers = int(cfg.get("workers", 1))

    controls = set(_control_ids(cfg, root))
    by_id = {s.sample_id: s for s in samples}
    if limit:
        # Truncate the STUDY rows only; the controls are what make the result readable at all,
        # so a scoped run must keep its floor. (`--limit` silently dropping a control is the
        # same class of trap as `--limit` truncating before the resume filter in the CR arm.)
        keep = [s.sample_id for s in samples if s.sample_id not in controls][:limit]
        samples = [by_id[i] for i in keep] + [by_id[i] for i in controls if i in by_id]

    jobs = [(s.sample_id, s.sample_id in controls) for s in samples]
    stats = RunStats()

    judge_cfg = yaml.safe_load((root / cfg["judge_config"]).read_text(encoding="utf-8"))
    adapter, validator_model = _adapter_for(judge_cfg, use_api=use_api, root=root,
                                            dry_run=dry_run)

    done = {} if dry_run else {
        r.sample_id: r for r in (io.read_jsonl(out_manifest, ConstructScreenResult)
                                 if out_manifest.exists() else [])
    }
    pending = [j for j in jobs if j[0] not in done]
    stats.skipped = len(jobs) - len(pending)
    if dry_run or not use_api:
        stats.screened = len(pending)
        return stats

    failure_log = load_failure_log_path(cfg, root)
    prompt = build_construct_screen_prompt()

    def do_one(item):
        sample_id, is_ctrl = item
        sample = by_id[sample_id]
        try:
            raw = adapter.generate(JudgeRequest(
                prompt, [resolve(sample.original_image_path, root)], "validation"))
        except Exception as exc:  # noqa: BLE001
            return ("api_fail", sample_id, f"{type(exc).__name__}: {exc}")
        raw_rel = save_raw_response(root, raw_dir, validator_model,
                                    f"construct::{sample_id}", raw)
        p = parse_construct_screen(raw)
        return ("ok", ConstructScreenResult(
            sample_id=sample_id, source_dataset=sample.source_dataset, is_control=is_ctrl,
            is_photograph=p.is_photograph, single_subject=p.single_subject,
            no_minor=p.no_minor, skin_visible=p.skin_visible,
            passed=p.passed, reason=p.reason,
            validator_model=validator_model, raw_response_path=raw_rel,
            parse_success=p.success, parse_error=(p.error or None),
        ))

    for outcome in run_jobs(pending, do_one, workers):
        if outcome[0] == "api_fail":
            _, sid, msg = outcome
            stats.api_failures += 1
            log_failure(failure_log, sid, msg)
            continue
        result = outcome[1]
        if not result.parse_success:
            stats.parse_failures += 1
        io.append_jsonl(out_manifest, result)
        done[result.sample_id] = result
        stats.screened += 1

    write_summary(summary_csv, list(done.values()))
    write_passed(root / cfg.get("passed_json", DEFAULT_PASSED_JSON), list(done.values()))
    return stats


DEFAULT_PASSED_JSON = "results/v2_fairness/metrics/construct_screen_passed.json"


def write_passed(path, rows: Sequence[ConstructScreenResult]) -> int:
    """The pass list pass 2 of the pool build consumes, rewritten on EVERY run.

    ★ WHY THIS FUNCTION EXISTS (2026-08-01). This file used to be produced out-of-band and
    nothing regenerated it. After the screen increment took the measured set from 664 to 1,309
    rows (396 -> 785 passes), the on-disk list was still the **stale 396** and every downstream
    step would have used it: `build_fairness_pool --include-ids` filters to whatever ids it is
    handed, so pass 2 would have rebuilt the pool at HALF the earned n, reported a clean report,
    and priced the panel off it. Nothing errors, nothing logs, and the number looks plausible --
    the same shape as this project's other silent-join failures.
    →  The rule: a file that gates a later stage must be rewritten by the stage that measures
    it, never assembled on the side.

    Membership is delegated to `passed_sample_ids`, which RAISES unless the false-reject floor
    was measured and sits under the ceiling. Writing the file through that check is the point:
    a pass list produced without it is a pool built on a screen nobody calibrated.
    """
    passed = sorted(passed_sample_ids(rows))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(passed, indent=0), encoding="utf-8")
    return len(passed)


def false_reject_rate(rows: Sequence[ConstructScreenResult]) -> Optional[float]:
    """Share of hand-labelled-usable controls the screen rejected. None if no controls ran."""
    ctrl = [r for r in rows if r.is_control and r.parse_success]
    if not ctrl:
        return None
    return sum(1 for r in ctrl if not r.passed) / len(ctrl)


def passed_sample_ids(rows: Sequence[ConstructScreenResult],
                      *, max_false_reject: float = MAX_FALSE_REJECT) -> set:
    """The scenes cleared to enter the pool -- or a refusal.

    Raises rather than returning a pass set built on a screen that discards good scenes at
    random. A rubber stamp and a shredder both produce a number; only a measured floor tells
    them apart, and gating on an unmeasured one is how a pool silently stops meaning anything.
    """
    floor = false_reject_rate(rows)
    if floor is None:
        raise RuntimeError(
            "no control rows: the screen's false-reject rate was never measured, so its "
            "pass set cannot be trusted. Supply `control_ids` and re-run."
        )
    if floor > max_false_reject:
        raise RuntimeError(
            f"construct screen rejected {floor:.1%} of hand-labelled-usable controls, above "
            f"the {max_false_reject:.0%} ceiling. At that rate it is removing scenes at "
            "random rather than removing bad ones. Do not gate on it."
        )
    return {r.sample_id for r in rows
            if r.parse_success and r.passed and not r.is_control}


def agreement_with_labels(rows: Sequence[ConstructScreenResult], labels: dict) -> dict:
    """2x2 against a hand-labelled set, in BOTH directions.

    A floor alone is not enough here. The floor says how often a good scene is thrown away;
    it says nothing about how often a bad one is admitted -- and one of the four questions is
    an ethics hard stop, where a false accept is the error with no recovery. So the confusion
    matrix is reported whole.

    ⚠️ The labels are a thumbnail-scale human read, and this project has measured that such a
    read produced three wrong calls a floor-calibrated validator correctly contradicted. This
    is therefore AGREEMENT, not accuracy, and neither side is ground truth. The exception is
    `no_minor`: a scene a human flagged as a minor stays out regardless of the screen.
    """
    by_id = {r.sample_id: r for r in rows if r.parse_success}
    usable = [i for i in labels.get("usable", []) if i in by_id]
    unusable = [i for i in labels.get("unusable", []) if i in by_id]
    tp = sum(1 for i in usable if by_id[i].passed)
    fn = len(usable) - tp
    fp = sum(1 for i in unusable if by_id[i].passed)
    tn = len(unusable) - fp
    total = tp + fn + fp + tn
    return {
        "n_labelled": total,
        "human_usable": len(usable), "human_unusable": len(unusable),
        "screen_passes_human_usable": tp,
        "false_reject": fn,
        "false_accept": fp,
        "screen_rejects_human_unusable": tn,
        "false_reject_rate": round(fn / len(usable), 4) if usable else None,
        "false_accept_rate": round(fp / len(unusable), 4) if unusable else None,
        "agreement": round((tp + tn) / total, 4) if total else None,
        # The ethics question on its own. A false accept here is not a power problem.
        "minor_false_accept": [
            i for i in unusable
            if by_id[i].passed and "MINOR" in labels.get("unusable_reasons", {}).get(i, "")
        ],
    }


def write_summary(path: Path, rows: Sequence[ConstructScreenResult]) -> None:
    """Per-question rates, with the control (false-reject floor) as its own row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    agg: Dict[bool, dict] = collections.defaultdict(
        lambda: collections.Counter({"n": 0, "pass": 0}))
    for r in rows:
        if not r.parse_success:
            continue
        a = agg[r.is_control]
        a["n"] += 1
        a["pass"] += bool(r.passed)
        for f in SCREEN_FIELDS:
            a[f] += bool(getattr(r, f))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_kind", "n", *[f"{f}_rate" for f in SCREEN_FIELDS], "pass_rate"])
        for is_ctrl in sorted(agg):
            a = agg[is_ctrl]
            n = a["n"] or 1
            w.writerow([
                "false_reject_floor" if is_ctrl else "measured", a["n"],
                *[round(a[f] / n, 4) for f in SCREEN_FIELDS],
                round(a["pass"] / n, 4),
            ])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--use-api", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    stats = run(a.config, root=Path(a.root) if a.root else None,
                use_api=a.use_api, dry_run=a.dry_run, limit=a.limit)
    print(stats.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
