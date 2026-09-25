"""D-class preservation gate: validate each counterfactual PAIR before paying to edit it.

This is the fairness track's analogue of §6, and it is load-bearing in the same way. D claims
"the edit task is identical and only a protected attribute differs, yet the judge scores
differ". Two failure modes destroy that claim, in opposite directions:

  * the flip did not take  -> the "pair" is one scene twice, and any gap is editor noise;
  * the scene also moved   -> a score gap is a *justified* deduction, not unfairness.

So a pair must pass BOTH, and a failing pair is dropped from BOTH sides. Dropping one side
would leave an orphan variant that no paired metric can use.

★ RUN THIS BEFORE STEP 4. The gate costs ~$0.001 per pair against $0.12 for the two step-4
renders it authorises, so filtering first is strictly cheaper than filtering after.

★ THE VALIDATOR GETS ITS OWN FALSE-FLIP FLOOR (`--control`). Every pair is also asked about a
pair made of the SAME image twice, where a sound auditor must answer
`attribute_flipped=False`. Whatever rate it reports there is its rubber-stamp floor, and the
measured flip rate has to be read against it. This is exactly the role `sham` plays for the
quality validators, and it is not optional: on 2026-07-30 a thumbnail-scale human read of the
same images produced three false "the flip failed" calls that the floor-calibrated validator
correctly contradicted, so the floor is what makes the automated verdict trustworthy rather
than the other way round. Measured: gpt-4o-mini scored 0/20 false flips.

    python -m edit_judge_bias.experiments.run_attribute_validation \
        --config configs/fairness/attribute_validation.yaml --dry-run
    python -m edit_judge_bias.experiments.run_attribute_validation \
        --config configs/fairness/attribute_validation.yaml --use-api
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve
from edit_judge_bias.experiments.judge_common import (
    adapter_for as _adapter_for,
    load_failure_log_path,
    log_failure,
    run_jobs,
    save_raw_response,
)
from edit_judge_bias.fairness.records import AttributeEditRecord, AttributeValidationResult
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.parser import parse_attribute_validation
from edit_judge_bias.prompts.attribute_validator_prompt import (
    ATTRIBUTE_PHRASE,
    build_attribute_validator_prompt,
)


@dataclass
class RunStats:
    validated: int = 0
    skipped: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    incomplete_pairs: int = 0

    def summary(self) -> str:
        return (
            f"validated={self.validated} skipped={self.skipped} "
            f"parse_failures={self.parse_failures} api_failures={self.api_failures} "
            f"incomplete_pairs={self.incomplete_pairs}"
        )


def _pair_key(base_sample_id: str, attribute: str, *, control: bool = False) -> str:
    return f"{base_sample_id}__{attribute}" + ("__CONTROL" if control else "")


def build_pairs(
    edits: Sequence[AttributeEditRecord],
) -> Tuple[List[Tuple[str, str, AttributeEditRecord, AttributeEditRecord]], int]:
    """Group edits into (base_sample_id, attribute) pairs of exactly two variants.

    A group with only one variant on disk (its sibling was refused, or the run was cut short)
    is NOT validated — there is no pair to compare — and is counted as incomplete so the
    summary shows it instead of it vanishing.
    """
    groups: Dict[Tuple[str, str], List[AttributeEditRecord]] = collections.defaultdict(list)
    for e in edits:
        groups[(e.base_sample_id, e.attribute)].append(e)
    pairs, incomplete = [], 0
    for (base, attr), members in sorted(groups.items()):
        if len(members) != 2:
            incomplete += 1
            continue
        a, b = sorted(members, key=lambda m: m.variant_label)
        pairs.append((base, attr, a, b))
    return pairs, incomplete


def run(
    config_path: str | Path,
    *,
    root: Optional[Path] = None,
    use_api: bool = False,
    dry_run: bool = False,
    control: bool = True,
) -> RunStats:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    edits = io.read_jsonl(root / cfg["attribute_edits"], AttributeEditRecord)
    out_manifest = root / cfg["out_manifest"]
    raw_dir = cfg.get("raw_dir", "results/v2_fairness/raw_responses_validation")
    summary_csv = root / cfg["summary_csv"]
    workers = int(cfg.get("workers", 1))
    if not control:
        control = bool(cfg.get("control", True))

    pairs, incomplete = build_pairs(edits)
    stats = RunStats(incomplete_pairs=incomplete)

    judge_cfg = yaml.safe_load((root / cfg["judge_config"]).read_text(encoding="utf-8"))
    adapter, validator_model = _adapter_for(judge_cfg, use_api=use_api, root=root, dry_run=dry_run)

    # jobs: (is_control, base, attribute, image_1, image_2, label_a, label_b)
    jobs = []
    for base, attr, a, b in pairs:
        jobs.append((False, base, attr, a.counterfactual_image_path, b.counterfactual_image_path,
                     a.variant_label, b.variant_label))
        if control:
            # SAME image twice -> the auditor's false-flip floor.
            jobs.append((True, base, attr, a.counterfactual_image_path,
                         a.counterfactual_image_path, a.variant_label, b.variant_label))

    done = {} if dry_run else {
        r.pair_key: r for r in (io.read_jsonl(out_manifest, AttributeValidationResult)
                                if out_manifest.exists() else [])
    }
    pending = [j for j in jobs if _pair_key(j[1], j[2], control=j[0]) not in done]
    stats.skipped = len(jobs) - len(pending)

    if dry_run or not use_api:
        stats.validated = len(pending)
        return stats

    failure_log = load_failure_log_path(cfg, root)

    def do_one(item):
        is_ctrl, base, attr, p1, p2, la, lb = item
        key = _pair_key(base, attr, control=is_ctrl)
        prompt = build_attribute_validator_prompt(ATTRIBUTE_PHRASE.get(attr, attr), la, lb)
        images = [resolve(p1, root), resolve(p2, root)]
        try:
            raw = adapter.generate(JudgeRequest(prompt, images, "validation"))
        except Exception as exc:  # noqa: BLE001
            return ("api_fail", key, f"{type(exc).__name__}: {exc}")
        raw_rel = save_raw_response(root, raw_dir, validator_model, f"attrval::{key}", raw)
        parsed = parse_attribute_validation(raw)
        return ("ok", AttributeValidationResult(
            pair_key=key, base_sample_id=base, attribute=attr,
            label_a=la, label_b=(la if is_ctrl else lb), is_control=is_ctrl,
            person_legible=parsed.person_legible,
            attribute_flipped=parsed.attribute_flipped,
            scene_preserved=parsed.scene_preserved,
            passed=parsed.passed, reason=parsed.reason,
            validator_model=validator_model, raw_response_path=raw_rel,
            parse_success=parsed.success, parse_error=(parsed.error or None),
        ))

    for outcome in run_jobs(pending, do_one, workers):
        if outcome[0] == "api_fail":
            _, key, msg = outcome
            stats.api_failures += 1
            log_failure(failure_log, key, msg)
            continue
        result = outcome[1]
        if not result.parse_success:
            stats.parse_failures += 1
        io.append_jsonl(out_manifest, result)
        done[result.pair_key] = result
        stats.validated += 1

    _write_summary(summary_csv, list(done.values()))
    return stats


def _write_summary(path: Path, rows: Sequence[AttributeValidationResult]) -> None:
    """Per-attribute pass rate, with the control (false-flip floor) as its own rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    agg: Dict[Tuple[str, bool], dict] = collections.defaultdict(
        lambda: {"n": 0, "legible": 0, "flipped": 0, "scene": 0, "pass": 0}
    )
    for r in rows:
        if not r.parse_success:
            continue
        a = agg[(r.attribute, r.is_control)]
        a["n"] += 1
        a["legible"] += bool(r.person_legible)
        a["flipped"] += bool(r.attribute_flipped)
        a["scene"] += bool(r.scene_preserved)
        a["pass"] += bool(r.passed)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attribute", "row_kind", "n", "person_legible_rate",
                    "attribute_flipped_rate", "scene_preserved_rate", "pass_rate"])
        for (attr, is_ctrl), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            n = a["n"] or 1
            w.writerow([
                attr,
                "false_flip_floor" if is_ctrl else "measured",
                a["n"],
                round(a["legible"] / n, 4),
                round(a["flipped"] / n, 4),
                round(a["scene"] / n, 4),
                round(a["pass"] / n, 4),
            ])


def passed_pair_keys(
    manifest: Path | Sequence[Path], *, require_control_floor: float = 0.10
) -> set:
    """Pair keys (``{base}__{attribute}``) admitted by EVERY auditor given.

    Multiple manifests are combined by INTERSECTION, following the §6.1 precedent for the QC
    subset: the set means "pairs shown to be usable", so an extra auditor can only shrink it.
    A union would make the admitted set GROW as scrutiny increased, which is the wrong
    direction and is a mistake this project has already made once.

    The asymmetry is deliberate. An admitted-but-bad pair corrupts the gap (a failed flip makes
    the "pair" one scene twice; a moved scene makes the gap a justified deduction). A
    dropped-but-good pair only costs power, and the power cost is reported as `n`.

    `require_control_floor` is a tripwire, not a filter: if an auditor's own false-flip rate on
    identical-image controls exceeds it, its verdicts are a rubber stamp and this raises rather
    than handing back a pass set built on one. ⚠️ The control bounds false POSITIVES only —
    it cannot detect a missed flip, which is why more than one auditor is worth the dollar.
    """
    paths = [manifest] if isinstance(manifest, (str, Path)) else list(manifest)
    admitted: Optional[set] = None
    for path in paths:
        rows = [r for r in io.read_jsonl(Path(path), AttributeValidationResult)
                if r.parse_success]
        ctrl = [r for r in rows if r.is_control]
        if ctrl:
            false_flip = sum(bool(r.attribute_flipped) for r in ctrl) / len(ctrl)
            if false_flip > require_control_floor:
                raise ValueError(
                    f"{Path(path).name}: false-flip floor is {false_flip:.3f} on {len(ctrl)} "
                    f"identical-image controls (> {require_control_floor}); its 'flipped' "
                    "verdicts are not trustworthy, so the passed set would be a rubber stamp"
                )
        keys = {r.pair_key for r in rows if not r.is_control and r.passed}
        admitted = keys if admitted is None else (admitted & keys)
    return admitted or set()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate D-class counterfactual pairs.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--use-api", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the false-flip floor rows (NOT recommended; see the docstring)")
    a = ap.parse_args(argv)
    stats = run(a.config, root=a.root, use_api=a.use_api, dry_run=a.dry_run,
                control=not a.no_control)
    prefix = "[dry-run] " if (a.dry_run or not a.use_api) else ""
    print(f"{prefix}attribute validation: {stats.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
