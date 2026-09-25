"""A-class (prompt-level) bias specs shared by both judge runners (§4.A / §5.4).

Unlike the B/C-class biases, which are pixels on disk referenced by a
``BiasedRecord``, an A-class bias lives entirely in the judge *prompt*: the image
shown is the ordinary edited image and only the surrounding text changes. So each
entry under ``prompt_biases:`` expands to one extra judged condition per sample (or
per pair), carrying its own ``bias_type`` so the verdict routes to
``results/biased_judgments/`` like any other bias.

Config shape (both runners)::

    prompt_biases:
      - bias_type: bandwagon              # A2 — fabricated social proof
        bandwagon: true                   #   scoring
        bandwagon_target: seeded          #   pairwise: seeded | A | B
      - bias_type: model_name             # A3 — the true generator is named
        model_name_source: true_name
      - bias_type: model_name_permuted    # A3 counterfactual control
        model_name_source: permuted
      - bias_type: model_name_anon        # A3 anonymous control (no name line)
        model_name_source: none
      - bias_type: model_name_literal
        model_name_source: literal
        model_name: "GPT-Image"

``bias_type`` is free text on purpose: it is the analysis group key, so a run can
carry several A3 variants side by side.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence


def permuted_name_map(names: Sequence[str], seed: int) -> Dict[str, str]:
    """Map each model name to a *different* one, deterministically.

    Used by the A3 counterfactual control: the prompt names a model that did not
    produce the image, so any score change is attributable to the name alone
    rather than to naming-in-general. Seeded-shuffle-then-rotate guarantees a
    derangement (no name maps to itself) for two or more names; a single-name
    roster has no derangement and is returned unchanged.
    """
    uniq = sorted(set(names))
    if len(uniq) < 2:
        return {n: n for n in uniq}
    shuffled = list(uniq)
    random.Random(f"{seed}:model_name_permutation").shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    return dict(zip(shuffled, rotated))


def resolve_model_name(
    spec: dict, actual: Optional[str], name_map: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Resolve the model name an A3 spec wants injected, or None for no name line."""
    source = spec.get("model_name_source", "none")
    if source == "true_name":
        return actual
    if source == "permuted":
        if not actual:
            return None
        return (name_map or {}).get(actual, actual)
    if source == "literal":
        return spec.get("model_name")
    return None


def bandwagon_side(spec: dict, key: str, seed: int) -> Optional[str]:
    """Which side an A2 pairwise spec endorses: an explicit "A"/"B", or seeded.

    Seeded means the endorsed side is a deterministic function of (seed, pair id),
    so the fabricated majority does not systematically favour slot A -- otherwise
    A2 would be confounded with the A1 position bias.
    """
    target = spec.get("bandwagon_target")
    if target in ("A", "B"):
        return target
    if target in (None, False):
        return None
    if str(target).lower() != "seeded":
        raise ValueError(f"bandwagon_target must be 'A', 'B' or 'seeded'; got {target!r}")
    return random.Random(f"{seed}:bandwagon:{key}").choice(("A", "B"))


def spec_bias_type(spec: dict) -> str:
    """The analysis group key for one prompt-bias spec (required)."""
    bias_type = spec.get("bias_type")
    if not bias_type:
        raise ValueError(f"prompt_biases entry needs a bias_type: {spec!r}")
    return str(bias_type)


def load_specs(cfg: dict) -> List[dict]:
    """Read and validate the `prompt_biases:` block of an experiment config."""
    specs = cfg.get("prompt_biases") or []
    if not isinstance(specs, list):
        raise ValueError("prompt_biases must be a list of specs")
    seen = set()
    for spec in specs:
        bias_type = spec_bias_type(spec)
        if bias_type in seen:
            raise ValueError(f"duplicate prompt_biases bias_type: {bias_type}")
        seen.add(bias_type)
    return specs
