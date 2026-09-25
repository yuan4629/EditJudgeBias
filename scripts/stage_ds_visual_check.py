"""Stage the D-S injections as side-by-side JPGs for a human eye. Free, no API.

★ WHY A HUMAN STILL HAS TO LOOK
Everything about this arm is verified arithmetically -- locality is `max_outside_diff == 0`,
the dose is a measured ITA delta, the two arms share one mask digest. None of that answers the
question a reviewer will ask first: **does it look like a photograph of a person with a
different skin tone, or does it look like a recoloured patch?** That is a perceptual question,
and the project has already learned the hard way that a floor-calibrated instrument beats an
eyeball on some questions and loses on others -- so both get a turn.

Each sheet shows, for one scene:  dark | ORIGINAL | light   (top row, the original member)
                                  dark | EDITED   | light   (bottom row, the edited member)
plus the measured dose, so a suspicious-looking cell can be traced to a number.

    PYTHONPATH=src python scripts/stage_ds_visual_check.py
    PYTHONPATH=src python scripts/stage_ds_visual_check.py --limit 8 --only-passed
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root, resolve
from edit_judge_bias.fairness.records import AttributeInjectionRecord

TILE = 300
LABEL_H = 22


def _tile(path: Path, size: int) -> Image.Image:
    """Fit into a square cell WITHOUT distorting the frame.

    ⚠️ This used to force `resize((size, size))`. That was harmless while the corpora were
    500/512/1024-px squares and wrong the moment the pool moved to OmniEdit, whose frames are
    768x1344 and 1344x768: the reviewer at gate P4 is asked whether a person looks like a
    person with a different skin tone, and a 1.75x horizontal stretch is exactly the kind of
    artefact that makes an honest answer impossible. Letterboxed instead, so the pixels the
    gate is judged on are the pixels that were injected.
    """
    with Image.open(path) as im:
        frame = im.convert("RGB")
        frame.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (24, 24, 24))
        canvas.paste(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
        return canvas


def _label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.rectangle([x, y, x + TILE, y + LABEL_H], fill=(20, 20, 20))
    draw.text((x + 4, y + 5), text[:52], fill=(240, 240, 240))


def build(*, root: Optional[Path] = None,
          injections: str = "data/manifests/attribute_injections.jsonl",
          gate_json: Optional[str] = "results/v2_fairness/metrics/construction_gate.json",
          out_dir: str = "data/human_validation_ds",
          limit: Optional[int] = None,
          only_passed: bool = False) -> dict:
    root = Path(root) if root is not None else default_root()
    rows = io.read_jsonl(root / injections, AttributeInjectionRecord)

    passed = None
    if gate_json and (root / gate_json).exists():
        data = json.loads((root / gate_json).read_text(encoding="utf-8"))
        passed = set(data.get("passed_pair_keys") or ())

    # (scene, member) -> {label: row}
    grouped: Dict[str, Dict[str, Dict[str, AttributeInjectionRecord]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row.role != "study":
            continue
        grouped[row.base_sample_id][row.applied_to][row.variant_label] = row

    out_root = root / out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    index: List[dict] = []

    scenes = sorted(grouped)
    if only_passed and passed is not None:
        scenes = [s for s in scenes if f"{s}__skin_tone" in passed]
    if limit:
        scenes = scenes[:limit]

    for scene in scenes:
        members = grouped[scene]
        rows_to_draw = [m for m in ("original", "edited") if m in members]
        if not rows_to_draw:
            continue
        width = TILE * 3
        height = len(rows_to_draw) * (TILE + LABEL_H)
        sheet = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)

        for r, member in enumerate(rows_to_draw):
            y = r * (TILE + LABEL_H)
            by_label = members[member]
            dark, light = by_label.get("dark"), by_label.get("light")
            if dark is None or light is None:
                continue
            source = resolve(dark.source_image_path, root)
            cells = [
                (resolve(dark.injected_image_path, root),
                 f"dark  dITA {dark.achieved_delta_ita:+.1f}  dE00 {dark.delta_e00_mean:.1f}"),
                (source, f"{member.upper()}  (untouched)"),
                (resolve(light.injected_image_path, root),
                 f"light dITA {light.achieved_delta_ita:+.1f}  dE00 {light.delta_e00_mean:.1f}"),
            ]
            for c, (path, text) in enumerate(cells):
                x = c * TILE
                sheet.paste(_tile(path, TILE), (x, y + LABEL_H))
                _label(draw, x, y, text)

        name = f"{scene}.jpg"
        sheet.save(out_root / name, quality=88)
        first = members[rows_to_draw[0]]
        index.append({
            "scene": scene,
            "sheet": f"{out_dir}/{name}",
            "gate_passed": None if passed is None else (f"{scene}__skin_tone" in passed),
            "instruction": first.get("dark").instruction if first.get("dark") else "",
            "person_frac": first.get("dark").person_frac if first.get("dark") else None,
            "skin_frac": first.get("dark").mask_frac if first.get("dark") else None,
            "human_verdict": "",
        })

    with (out_root / "index.jsonl").open("w", encoding="utf-8") as fh:
        for entry in index:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    (out_root / "README.md").write_text(
        "# D-S visual check\n\n"
        "Each sheet: `dark | untouched | light`, top row the original member, bottom row the\n"
        "dataset's edited member. Labels carry the measured ITA delta and mean CIEDE2000.\n\n"
        "Fill `human_verdict` in `index.jsonl` with one of:\n\n"
        "- `plausible`  - reads as a photograph of a person with a different skin tone\n"
        "- `recoloured` - reads as a recoloured patch (the failure mode that would make this\n"
        "  arm an artefact measurement rather than a fairness one)\n"
        "- `leaked`     - something outside the person changed (should be impossible: the gate\n"
        "  asserts `max_outside_diff == 0`, so a `leaked` verdict means the gate is wrong)\n"
        "- `no_person`  - no legible person (the detector screen should have caught it)\n\n"
        "★ A `recoloured` majority does not invalidate the arm, but it does change what may be\n"
        "claimed: the write-up would have to lead with the `sham_nonskin` contrast rather than\n"
        "with the attribute gap.\n",
        encoding="utf-8",
    )
    return {"scenes": len(index), "out_dir": out_dir,
            "gate_passed": sum(1 for e in index if e["gate_passed"]),
            "index": f"{out_dir}/index.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-passed", action="store_true")
    ap.add_argument("--out-dir", default="data/human_validation_ds")
    # v3 and v4 injections live in separate manifests on purpose -- same discipline as
    # `results/v2` vs `results/`. Pointing the staging at the wrong one would silently show
    # the OLD pool's sheets and invite a verdict about images nobody is going to judge.
    ap.add_argument("--injections", default="data/manifests/attribute_injections.jsonl")
    ap.add_argument("--gate-json",
                    default="results/v2_fairness/metrics/construction_gate.json")
    a = ap.parse_args()
    print(json.dumps(build(root=Path(a.root) if a.root else None, out_dir=a.out_dir,
                           injections=a.injections, gate_json=a.gate_json,
                           limit=a.limit, only_passed=a.only_passed), indent=2))


if __name__ == "__main__":
    main()
