"""WP-A4c — the C-class validator-prompt ablation table.

THE QUESTION.  The published validator template names four artefacts as "cosmetic" and
all four are B-CLASS (global brightness/saturation, borders, watermarks, edge text bands).
The four C-class overlays are never named, so the template is harsher on C-class BY
CONSTRUCTION.  That was a footnote until WP-A3's matched floor made `zoom_inset`
significantly below its floor on two validators -- at which point "the validator saw the
inset and called the content changed" and "the edit was destroyed" became observationally
equivalent, and §7.1's provability axis had to be written as UNINTERPRETABLE.

THE DESIGN.  Same images, same validator, same everything -- only `prompt_style` differs.
So the contrast is paired WITHIN each cue on the same 110 pictures, and McNemar is the
right test: `b` = the published prompt failed the image and the ablation passed it (the
direction predicted if the template asymmetry drove the result), `c` = the reverse.

★★ THE CEILING CHECK, WHICH IS WHY THIS MODULE EXISTS RATHER THAN A ONE-LINER.
Two of the four cues sit at a published pass rate of 1.000.  They CANNOT move up: `b` is
zero by construction, not by evidence.  Reading "no change on region_annotation" as
"the template asymmetry does not matter" would be reading a ceiling as a null.  Every
row therefore carries
`headroom` (how many images could possibly flip toward pass) and `interpretable`, and a
cue with no headroom is reported as UNINFORMATIVE rather than as a null.

Pre-registered before the arm ran:
  * `zoom_inset` moves up  -> the 0.918 is a template artefact; §5.7 must be rewritten.
  * `zoom_inset` does not  -> §5.7's finding is independently reinforced.
  * the two ceiling cues move a lot -> the ablation changed something other than what it
    was meant to change, and NOTHING here is readable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from edit_judge_bias.metrics.stats import wilson_interval
from edit_judge_bias.metrics.stats import benjamini_hochberg, mcnemar_pvalue

C_CLASS = ("region_annotation", "zoom_inset", "detail_caption", "distraction")

FIELDS = (
    "validator_model", "bias_type", "n_paired",
    "published_pass_rate", "ablation_pass_rate", "delta",
    "published_ci_low", "published_ci_high", "ablation_ci_low", "ablation_ci_high",
    "mcnemar_b", "mcnemar_c", "mcnemar_p", "mcnemar_q",
    "headroom", "interpretable", "moved_toward_pass",
)


def _verdicts(path: Path) -> Dict[str, bool]:
    """biased_id -> passed, dropping parse failures (a non-answer is not a verdict)."""
    out: Dict[str, bool] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("parse_success") is False or row.get("pass") is None:
            continue
        out[row["biased_id"]] = bool(row["pass"])
    return out


def _cue_of(manifest: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["biased_id"]] = row["bias_type"]
    return out


def build_prompt_ablation(
    ablation_manifest: Path,
    published_results: Path,
    ablation_results: Path,
    *,
    validator_model: str = "gemini-3.5-flash",
) -> List[dict]:
    cue = _cue_of(Path(ablation_manifest))
    pub = _verdicts(Path(published_results))
    abl = _verdicts(Path(ablation_results))

    by_cue: Dict[str, List[str]] = defaultdict(list)
    for bid, c in cue.items():
        if bid in pub and bid in abl:
            by_cue[c].append(bid)

    rows: List[dict] = []
    for c in sorted(by_cue):
        ids = sorted(by_cue[c])
        n = len(ids)
        p_pass = sum(pub[i] for i in ids)
        a_pass = sum(abl[i] for i in ids)
        # b: published FAILED, ablation PASSED -- the predicted direction if naming the
        # C-class overlays as cosmetic is what changes the verdict.
        b = sum(1 for i in ids if not pub[i] and abl[i])
        cc = sum(1 for i in ids if pub[i] and not abl[i])
        p = mcnemar_pvalue(b, cc) if (b + cc) else None
        headroom = n - p_pass  # images that COULD flip toward pass
        rows.append({
            "validator_model": validator_model,
            "bias_type": c,
            "n_paired": n,
            "published_pass_rate": round(p_pass / n, 4) if n else "",
            "ablation_pass_rate": round(a_pass / n, 4) if n else "",
            "delta": round((a_pass - p_pass) / n, 4) if n else "",
            **dict(zip(("published_ci_low", "published_ci_high"),
                       [round(v, 4) for v in wilson_interval(p_pass, n)] if n else ("", ""))),
            **dict(zip(("ablation_ci_low", "ablation_ci_high"),
                       [round(v, 4) for v in wilson_interval(a_pass, n)] if n else ("", ""))),
            "mcnemar_b": b, "mcnemar_c": cc, "mcnemar_p": p, "mcnemar_q": None,
            "headroom": headroom,
            # A cue already at 1.000 cannot move up.  Its `b=0` is arithmetic, not evidence.
            "interpretable": headroom > 0,
            "moved_toward_pass": b > cc,
        })

    live = [r for r in rows if r["mcnemar_p"] is not None]
    for r, q in zip(live, benjamini_hochberg([r["mcnemar_p"] for r in live])):
        r["mcnemar_q"] = q
    return rows


def write_prompt_ablation(rows: Sequence[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation-manifest",
                    default="data/manifests/biased_samples_cclass_ablation.jsonl")
    ap.add_argument("--published-results",
                    default="results/v2/quality/validation__gemini-3.5-flash.jsonl")
    ap.add_argument("--ablation-results",
                    default="results/v2_ablation/quality/validation__gemini-3.5-flash.jsonl")
    ap.add_argument("--out",
                    default="results/v2_ablation/metrics/validator_prompt_ablation.csv")
    args = ap.parse_args(argv)

    rows = build_prompt_ablation(Path(args.ablation_manifest),
                                 Path(args.published_results),
                                 Path(args.ablation_results))
    write_prompt_ablation(rows, Path(args.out))
    print(f"wrote -> {args.out} ({len(rows)} rows)")
    for r in rows:
        tag = "" if r["interpretable"] else "   ⚠️ AT CEILING — UNINFORMATIVE"
        print(f"  {r['bias_type']:<19} published {r['published_pass_rate']:<7} -> "
              f"ablation {r['ablation_pass_rate']:<7} delta {r['delta']:<+8} "
              f"b/c {r['mcnemar_b']}/{r['mcnemar_c']}  q={r['mcnemar_q']}"
              f"  headroom={r['headroom']}{tag}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
