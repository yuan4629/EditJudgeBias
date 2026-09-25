"""Counterfactual attribute-edit runner — fairness track scheme alpha (§5.4 D).

Selects human-category samples whose editing instruction is identity-irrelevant,
edits each *original* image with the gender / skin_tone templates via the
OpenAI-compatible image-edit endpoint, and writes the counterfactual originals +
an ``attribute_edits.jsonl`` manifest. Resumable (completed edit_ids are skipped),
fail-soft (a permanent API error is logged, the batch continues).

    # preview the plan, no network:
    python -m edit_judge_bias.experiments.run_attribute_edit --config \
        configs/fairness/qwen_image_edit.yaml --dry-run
    # real edits (spends on the user's Qwen-Image-Edit platform):
    export EDITJUDGE_IMAGE_EDIT_KEY=... ; python -m ... run_attribute_edit \
        --config configs/fairness/qwen_image_edit.yaml --use-api --limit 4

Ethics: identity-irrelevant instructions only; see docs/ETHICS_fairness_track.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve, to_rel_posix
from edit_judge_bias.data.schema import SampleRecord
from edit_judge_bias.experiments.judge_common import run_jobs
from edit_judge_bias.experiments.run_scoring_judge import _pick
from edit_judge_bias.fairness.attribute_templates import variants_for
from edit_judge_bias.fairness.image_edit_client import ImageEditError, QwenImageEditClient
from edit_judge_bias.fairness.records import AttributeEditRecord

# Instructions mentioning a person attribute are excluded: editing them would
# confound the attribute flip. Conservative keyword screen (§5.4 step 1).
#
# ★ FACIAL-FEATURE TERMS ADDED 2026-07-31, from a measured gap.
# The OmniEdit probe surfaced `Add eyes` and `Add glasses` among its usable scenes: both
# operate ON THE FACE, which is exactly where a skin-lightness manipulation acts, yet neither
# tripped the original screen because it named a facial *feature* rather than a demographic
# attribute. `face` alone was not enough. A judge asked "were eyes added correctly?" while we
# have recoloured the face is being asked two questions at once, and glasses additionally
# occlude the region the manipulation depends on.
#
# The screen stays deliberately over-broad: it costs pool size, and pool size is cheap
# relative to a confound that cannot be detected after the fact.
_ATTR_WORDS = re.compile(
    r"\b(man|woman|men|women|male|female|person|people|face|gender|guy|guys|"
    r"lady|ladies|gentleman|gentlemen|husband|wife|wives|mr|mrs|ms|"
    r"skin|hair|beard|mustache|old|elderly|young|age|he|she|his|her|hers|"
    # facial features and face-occluding wearables
    r"eye|eyes|eyebrow|eyebrows|eyelash|eyelashes|lip|lips|nose|cheek|cheeks|"
    r"chin|forehead|teeth|tooth|smile|freckle|freckles|wrinkle|wrinkles|"
    r"complexion|tan|tattoo|makeup|lipstick|glasses|sunglasses|spectacles|"
    r"mask|helmet|veil|hijab|beanie|"
    # ★ EXPRESSION AND EMOTION. Measured 2026-07-31: 293 of the 1,022 in-band OmniEdit scenes
    # are its `attr_mod_facial` sub-task -- "make him be angry", "let him be excited", "make
    # the dryad feel amused". The instruction acts on the face, which is exactly where a
    # skin-lightness manipulation acts, so the judge is asked two questions at once. OmniEdit
    # happens to encode that sub-task in its id and the pool config screens it there too, but
    # a corpus that publishes no sub-task label leaves this list as the only defence -- and it
    # still catches 6 rows the id screen misses even where both apply.
    r"cry|crying|cried|smile|smiling|frown|frowning|angry|anger|sad|sadness|happy|"
    r"happiness|surprised|surprise|laugh|laughing|expression|grin|grinning|scowl|"
    r"weep|weeping|tear|tears|emotion|emotional|mood|wink|pout|amused|annoyed|"
    r"anxious|excited|excitement|worried|afraid|scared|disgusted|joyful)\b",
    re.I,
)
# Minors must never have their demographic attributes edited (ethics, §D). Any
# hint of a child in the instruction excludes the sample outright.
_MINOR_WORDS = re.compile(
    r"\b(kid|kids|child|children|childs|toddler|toddlers|baby|babies|infant|"
    r"boy|boys|girl|girls|teen|teens|teenager|teenagers|son|daughter)\b",
    re.I,
)


def instruction_ok(instruction: Optional[str]) -> bool:
    """Is this instruction safe to flip a demographic attribute under?

    Split out of `_eligible` so the detector-first pool builder can apply the *instruction*
    screen without inheriting the `content_category == "human"` screen, which is measurably
    unreliable in these pools: of the 139 originals with a face covering >2% of the frame,
    only 87 carried the `human` label -- the other 52 were tagged global / object / scenery.
    The pool also contains gulls, a squirrel, a zebra and dolls under `human`.

    ⚠️ Both screens stay. The attribute screen prevents flipping an attribute the instruction
    itself refers to, and `_MINOR_WORDS` is an **ethics hard stop that must never be relaxed
    to grow the pool** (`tests/test_fairness_pipeline.py` and `tests/test_pool_detector.py`
    both pin it).
    """
    instr = instruction or ""
    return not _ATTR_WORDS.search(instr) and not _MINOR_WORDS.search(instr)


def _eligible(sample: SampleRecord) -> bool:
    """Unchanged behaviour: human category AND a clean instruction.

    Kept byte-equivalent because `tests/test_fairness_pipeline.py:25` imports it and three
    tests assert its exact semantics, and the published 41-scene pool was built with it.
    """
    return sample.content_category.value == "human" and instruction_ok(sample.instruction)


def _edit_id(sample_id: str, attribute: str, label: str) -> str:
    return f"{sample_id}__{attribute}__{label}"


def _load_done(manifest_out: Path) -> set[str]:
    if not manifest_out.exists():
        return set()
    return {r.edit_id for r in io.iter_jsonl(manifest_out, AttributeEditRecord)}


def run(
    config_path: str | Path,
    *,
    root: Optional[Path] = None,
    use_api: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict:
    root = Path(root) if root is not None else default_root()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    attributes: List[str] = cfg.get("attributes", ["gender", "skin_tone"])
    n_samples = int(cfg.get("n_samples", 4))
    samples_path = root / cfg["samples"]
    output_dir = root / cfg["output_dir"]
    manifest_out = root / cfg["manifest_out"]

    samples = [s for s in io.read_jsonl(samples_path, SampleRecord) if _eligible(s)]
    # One sample per base image (dedupe on original path) to vary the scene, not the edit_model.
    # Dedupe the WHOLE pool before taking n, so `eligible_samples` reports the real pool size
    # rather than being capped by n_samples, and so `sample_pick` has something to spread over.
    seen_orig, eligible = set(), []
    for s in samples:
        key = str(s.original_image_path)
        if key in seen_orig:
            continue
        if not resolve(s.original_image_path, root).exists():
            continue
        seen_orig.add(key)
        eligible.append(s)

    # The manifest is written source by source, so a prefix of 20 is 20 ImagenHub scenes and the
    # refusal rate it measures would be one source's, not the pool's. `spread` walks all of it.
    # `n_samples: 0` means all, the same convention as `per_bias: 0` in the validator configs
    # (and it avoids _pick's step = len/0).
    picked = (
        list(eligible)
        if n_samples <= 0
        else _pick(eligible, n_samples, str(cfg.get("sample_pick", "prefix")))
    )

    jobs = [(s, v) for s in picked for a in attributes for v in variants_for(a)]
    if limit is not None:
        jobs = jobs[:limit]

    if dry_run or not use_api:
        plan = {
            "eligible_samples": len(eligible),
            "picked_samples": len(picked),
            "picked_by_source": dict(Counter(s.source_dataset for s in picked)),
            "attributes": attributes,
            "planned_edits": len(jobs),
            "use_api": use_api,
            "note": "dry-run" if dry_run else "no --use-api: nothing was called",
        }
        return plan

    client = QwenImageEditClient(
        base_url=cfg.get("base_url") or os.environ.get("EDITJUDGE_BASE_URL", ""),
        model=cfg.get("model", "qwen-image-edit"),
        api_key_env=cfg.get("api_key_env", "EDITJUDGE_IMAGE_EDIT_KEY"),
        size=cfg.get("size"),
        response_format=cfg.get("response_format", "b64_json"),
        timeout=int(cfg.get("timeout", 120)),
        max_retries=int(cfg.get("max_retries", 4)),
        cache_dir=root / cfg["cache_dir"] if cfg.get("cache_dir") else None,
    )
    done = _load_done(manifest_out)
    stats = {"written": 0, "skipped": 0, "failed": 0}
    pending = [
        (s, v) for s, v in jobs
        if _edit_id(s.sample_id, v.attribute, v.label) not in done
    ]
    stats["skipped"] = len(jobs) - len(pending)

    def do_one(item):
        sample, variant = item
        eid = _edit_id(sample.sample_id, variant.attribute, variant.label)
        src = resolve(sample.original_image_path, root)
        out_path = output_dir / variant.attribute / f"{eid}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Only the network call and the (distinctly-named) PNG write happen in the
            # worker; the manifest append is done by the consumer on the main thread.
            img = client.edit(src, variant.prompt)
            img.save(out_path)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad, and it is the second layer of a fix, not a substitute for
            # the first: the client now retries connection-level errors, but ONE item must
            # never be able to abort the batch. A bare RemoteDisconnected slipping past
            # `except ImageEditError` killed a 164-render run at render 61 on 2026-07-30.
            # Resume by edit_id makes a logged failure cheap; a dead batch is not.
            return ("fail", eid, f"{type(exc).__name__}: {exc}")
        return ("ok", AttributeEditRecord(
            edit_id=eid,
            base_sample_id=sample.sample_id,
            attribute=variant.attribute,
            variant_label=variant.label,
            prompt=variant.prompt,
            original_image_path=to_rel_posix(src, root),
            counterfactual_image_path=to_rel_posix(out_path, root),
            edit_model=sample.edit_model,
            content_category=sample.content_category.value,
            instruction=sample.instruction,
            editor_model=cfg.get("model", "qwen-image-edit"),
        ))

    # gpt-image-2 runs ~70s per call, so a serial 164-render pass is over three hours.
    # `run_jobs` overlaps the waiting exactly as the judge arms do.
    for outcome in run_jobs(pending, do_one, int(cfg.get("workers", 1))):
        if outcome[0] == "fail":
            _, eid, msg = outcome
            stats["failed"] += 1
            _log_failure(root / cfg.get("failure_log", "results/logs/attribute_edit_fail.jsonl"),
                         eid, msg)
            continue
        io.append_jsonl(manifest_out, outcome[1])
        stats["written"] += 1
    return stats


def _log_failure(path: Path, edit_id: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"edit_id": edit_id, "message": message}, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Counterfactual attribute-edit runner (fairness track).")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--use-api", action="store_true", help="actually call the edit endpoint (spends)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, call nothing")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    result = run(args.config, root=args.root, use_api=args.use_api,
                 dry_run=args.dry_run, limit=args.limit)
    prefix = "[dry-run] " if (args.dry_run or not args.use_api) else ""
    print(f"{prefix}attribute edit: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
