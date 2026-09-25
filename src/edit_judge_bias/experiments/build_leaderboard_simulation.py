"""WP-A1e: what a quality-preserving cue does to an editor LEADERBOARD.

Claim A says the judge's score moves. Claim B says the item-level ranking moves for one
cue and survives for the others. Neither is the quantity a benchmark is actually used
for: benchmarks publish **leaderboards over editors**, and a reader is entitled to ask
what a 2-point deflation does to one. This table answers that directly, on the two
anchor blocks — the only parts of the grid drawn as whole turns with many editors each,
which is exactly what a leaderboard needs.

Two scenarios, and they are different threat models rather than two views of one:

``all_editors``
    Every editor's outputs carry the cue. Nobody is singled out, so a *uniform*
    deflation should leave the ranking alone. What survives here is claim B's
    "order-preserving deflation" restated at the level the field publishes, and what
    does not survive is a cue whose damage is heterogeneous across editors.

``single_editor``
    One editor's submissions carry the cue and everyone else's do not. This is the
    realistic case and it needs no adversary to be worrying: an editor that watermarks
    its outputs, pads them to a fixed aspect ratio, or ships them with a caption has
    done exactly this to itself. The row reports how far editors move.

**Scores are turn-centred before being averaged.** The EBench-18K anchor runs 17
editors over 48 turns with 8 editors per turn, so a raw per-editor mean would confound
"this editor is better" with "this editor drew easier turns". Subtracting each turn's
own mean removes it. On a complete design (ImagenHub: 8 editors x 30 turns) the
centring shifts every editor by the same constant and cannot change the ranking, so the
correction costs nothing where it is not needed.

Three numbers per (judge, source, cue), all in the output:

``kendall_tau_clean_vs_biased``  does the cue move the board at all
``delta_tau_human``             does it move the board AWAY from the human ranking —
                                the leaderboard-level analogue of claim B, with a
                                cluster bootstrap over turns
``rank change counts``          how many places, and whether the top spot changes

⚠️ The anchor arm asked four cues (`padding`, `text_overlay`, `brightness`,
`region_annotation`). `bandwagon` — the one cue that inflates — was never run there, so
this table can say nothing about an editor climbing. That asymmetry is a property of
the collected grid, not a finding, and it must travel with any leaderboard claim.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import PathLike, default_root
from edit_judge_bias.data.schema import JudgeResult, SampleRecord
from edit_judge_bias.experiments.build_claim_tables import PUBLISHED_ROSTER, _judges
from edit_judge_bias.experiments.build_claim_tables import ANCHOR_BIASES, write_csv
from edit_judge_bias.metrics.scoring_metrics import (
    check_single_scale,
    is_retest_repeat,
    resolve_score_field,
)

CUES: Tuple[str, ...] = ANCHOR_BIASES

#: Cluster-bootstrap resamples for the change in agreement with the human board. Turns
#: are the unit, for the same reason claim B resamples them: the 8 editors of one turn
#: share an image, an instruction and a difficulty.
DEFAULT_N_BOOT = 1000


@dataclass
class Board:
    """One (judge, anchor source) block, as the leaderboard inputs."""

    judge_model: str
    anchor_source: str
    #: turn key -> {editor: sample_id}
    turns: Dict[str, Dict[str, str]] = field(default_factory=dict)
    clean: Dict[str, float] = field(default_factory=dict)
    biased: Dict[str, Dict[str, float]] = field(default_factory=dict)
    human: Dict[str, float] = field(default_factory=dict)

    @property
    def editors(self) -> List[str]:
        return sorted({e for members in self.turns.values() for e in members})


# --------------------------------------------------------------------------- #
# leaderboard arithmetic                                                       #
# --------------------------------------------------------------------------- #
def turn_centred_means(
    turns: Dict[str, Dict[str, str]], scores: Dict[str, float]
) -> Dict[str, float]:
    """Editor -> mean of its turn-centred scores.

    A turn contributes nothing when fewer than two of its editors have a score: with
    one editor the centred value is identically zero, which would silently pull that
    editor toward the middle of the board.
    """
    collected: Dict[str, List[float]] = defaultdict(list)
    for members in turns.values():
        present = {e: scores[s] for e, s in members.items() if s in scores}
        if len(present) < 2:
            continue
        mean = statistics.fmean(present.values())
        for editor, value in present.items():
            collected[editor].append(value - mean)
    return {e: statistics.fmean(v) for e, v in collected.items() if v}


def ranks_from_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Rank 1 = best (highest score). Ties share the average rank."""
    from scipy.stats import rankdata

    editors = sorted(scores)
    if not editors:
        return {}
    values = [-scores[e] for e in editors]
    return dict(zip(editors, (float(r) for r in rankdata(values))))


def kendall_tau(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    """Kendall tau-b over the editors both boards score."""
    from scipy.stats import kendalltau

    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        return None
    stat = kendalltau([a[e] for e in shared], [b[e] for e in shared]).statistic
    return None if stat is None or stat != stat else float(stat)


def _swap_scores(board: Board, cue: str, editors: Sequence[str]) -> Dict[str, float]:
    """Clean scores everywhere, biased scores for `editors` only."""
    swapped = dict(board.clean)
    targets = set(editors)
    cued = board.biased.get(cue, {})
    for members in board.turns.values():
        for editor, sample_id in members.items():
            if editor in targets and sample_id in cued:
                swapped[sample_id] = cued[sample_id]
            elif editor in targets:
                swapped.pop(sample_id, None)  # no biased score: drop, never reuse clean
    return swapped


# --------------------------------------------------------------------------- #
# assembling the boards                                                        #
# --------------------------------------------------------------------------- #
def build_boards(
    samples: Sequence[SampleRecord],
    results_dir: PathLike,
    *,
    cues: Sequence[str] = CUES,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[Board]:
    results_dir = Path(results_dir)
    anchor = {
        s.sample_id: s for s in samples
        if getattr(s.metadata, "anchor_source", None)
    }
    # Roster, not bare discovery.  Coverage alone stops admitting a judge once it is
    # complete, and this table is published: WP-A5's sixth judge took it from 40 rows to
    # 48 the moment its anchor arm passed the coverage threshold (caught by
    # `test_frozen_grid_covers_five_judges_two_anchors_four_cues`, 2026-08-18).
    models = _judges(results_dir, "scoring", roster)
    wanted = set(cues)

    boards: List[Board] = []
    for model in models:
        base = io.read_jsonl(results_dir / "raw_judgments" / f"scoring__{model}.jsonl",
                             JudgeResult)
        cued = io.read_jsonl(results_dir / "biased_judgments" / f"scoring__{model}.jsonl",
                             JudgeResult)
        check_single_scale(base + cued, label=f"leaderboard inputs ({model})")
        field_name = resolve_score_field(base + cued)
        per_source: Dict[str, Board] = {}
        for r in base:
            value = getattr(r, field_name, None)
            if not (r.parse_success and value is not None and not r.bias_type
                    and not is_retest_repeat(r) and r.sample_id in anchor):
                continue
            sample = anchor[r.sample_id]
            board = per_source.setdefault(
                sample.metadata.anchor_source,
                Board(judge_model=model, anchor_source=sample.metadata.anchor_source),
            )
            turn = f"{sample.original_image_path.as_posix()}|{sample.instruction}"
            board.turns.setdefault(turn, {})[sample.edit_model] = r.sample_id
            board.clean[r.sample_id] = float(value)
            if sample.human_score is not None:
                board.human[r.sample_id] = float(sample.human_score)
        for r in cued:
            value = getattr(r, field_name, None)
            if not (r.parse_success and value is not None and r.bias_type in wanted
                    and r.sample_id in anchor):
                continue
            source = anchor[r.sample_id].metadata.anchor_source
            if source in per_source:
                per_source[source].biased.setdefault(r.bias_type, {})[r.sample_id] = (
                    float(value)
                )
        boards.extend(per_source[k] for k in sorted(per_source))
    return boards


# --------------------------------------------------------------------------- #
# the two scenarios                                                            #
# --------------------------------------------------------------------------- #
def _rank_move_summary(before: Dict[str, float], after: Dict[str, float]) -> dict:
    shared = sorted(set(before) & set(after))
    moves = {e: after[e] - before[e] for e in shared}
    changed = [e for e, d in moves.items() if abs(d) > 1e-9]
    top_before = min(shared, key=lambda e: before[e]) if shared else None
    top_after = min(shared, key=lambda e: after[e]) if shared else None
    return {
        "n_editors": len(shared),
        "n_rank_changes": len(changed),
        "max_rank_change": round(max((abs(d) for d in moves.values()), default=0.0), 2),
        "top1_changed": None if top_before is None else bool(top_before != top_after),
        "top1_clean": top_before,
        "top1_biased": top_after,
    }


def _bootstrap_delta_tau(
    board: Board, cue: str, *, n_boot: int, seed: int
) -> Tuple[Optional[float], Optional[float]]:
    """Cluster-bootstrap CI for the change in agreement with the human board."""
    import numpy as np

    keys = sorted(board.turns)
    if len(keys) < 2 or not board.human:
        return (None, None)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    biased_scores = _swap_scores(board, cue, board.editors)
    for _ in range(n_boot):
        picked = [keys[i] for i in rng.integers(0, len(keys), size=len(keys))]
        turns = {f"{k}#{i}": board.turns[k] for i, k in enumerate(picked)}
        human = ranks_from_scores(turn_centred_means(turns, board.human))
        clean = ranks_from_scores(turn_centred_means(turns, board.clean))
        after = ranks_from_scores(turn_centred_means(turns, biased_scores))
        a, b = kendall_tau(human, clean), kendall_tau(human, after)
        if a is not None and b is not None:
            values.append(b - a)
    if len(values) < 2:
        return (None, None)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return (round(float(lo), 4), round(float(hi), 4))


def build_leaderboard_rows(
    samples: Sequence[SampleRecord],
    results_dir: PathLike,
    *,
    cues: Sequence[str] = CUES,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    roster: Optional[Sequence[str]] = PUBLISHED_ROSTER,
) -> List[dict]:
    rows: List[dict] = []
    for board in build_boards(samples, results_dir, cues=cues, roster=roster):
        clean_means = turn_centred_means(board.turns, board.clean)
        clean_ranks = ranks_from_scores(clean_means)
        human_ranks = ranks_from_scores(turn_centred_means(board.turns, board.human))
        tau_human_clean = kendall_tau(human_ranks, clean_ranks)
        per_editor_turns = defaultdict(int)
        for members in board.turns.values():
            for editor in members:
                per_editor_turns[editor] += 1

        for cue in cues:
            if cue not in board.biased:
                continue
            # --- scenario 1: everybody carries the cue -------------------------- #
            all_ranks = ranks_from_scores(
                turn_centred_means(board.turns, _swap_scores(board, cue, board.editors))
            )
            tau_human_biased = kendall_tau(human_ranks, all_ranks)
            lo, hi = _bootstrap_delta_tau(board, cue, n_boot=n_boot, seed=seed)
            summary = _rank_move_summary(clean_ranks, all_ranks)
            delta_tau = (None if tau_human_clean is None or tau_human_biased is None
                         else round(tau_human_biased - tau_human_clean, 4))
            rows.append({
                "scenario": "all_editors",
                "judge_model": board.judge_model,
                "anchor_source": board.anchor_source,
                "bias_type": cue,
                "n_turns": len(board.turns),
                "n_items": len(board.clean),
                "turns_per_editor_min": min(per_editor_turns.values()),
                "turns_per_editor_max": max(per_editor_turns.values()),
                **summary,
                "kendall_tau_clean_vs_biased": (
                    None if (t := kendall_tau(clean_ranks, all_ranks)) is None
                    else round(t, 4)
                ),
                "kendall_tau_human_vs_clean": (
                    None if tau_human_clean is None else round(tau_human_clean, 4)
                ),
                "kendall_tau_human_vs_biased": (
                    None if tau_human_biased is None else round(tau_human_biased, 4)
                ),
                "delta_tau_human": delta_tau,
                "delta_tau_ci_low": lo,
                "delta_tau_ci_high": hi,
                "delta_tau_ci_excludes_zero": (
                    None if lo is None or hi is None else bool(lo > 0 or hi < 0)
                ),
                "n_boot": n_boot,
                "family": "leaderboard_simulation",
            })

            # --- scenario 2: one editor carries it alone ------------------------ #
            drops: Dict[str, float] = {}
            lost_top1: List[str] = []
            for editor in board.editors:
                moved = ranks_from_scores(
                    turn_centred_means(board.turns, _swap_scores(board, cue, [editor]))
                )
                if editor not in moved or editor not in clean_ranks:
                    continue
                drops[editor] = moved[editor] - clean_ranks[editor]
                if clean_ranks[editor] == min(clean_ranks.values()) and moved[editor] > 1:
                    lost_top1.append(editor)
            if not drops:
                continue
            worst = max(drops, key=lambda e: drops[e])
            rows.append({
                "scenario": "single_editor",
                "judge_model": board.judge_model,
                "anchor_source": board.anchor_source,
                "bias_type": cue,
                "n_turns": len(board.turns),
                "n_items": len(board.clean),
                "n_editors": len(drops),
                # Positive = the editor FELL when its own outputs carried the cue.
                "mean_rank_drop": round(statistics.fmean(drops.values()), 3),
                "median_rank_drop": round(statistics.median(drops.values()), 3),
                "max_rank_drop": round(max(drops.values()), 2),
                "n_editors_falling": sum(1 for d in drops.values() if d > 0),
                "n_editors_falling_3_or_more": sum(1 for d in drops.values() if d >= 3),
                "n_editors_rising": sum(1 for d in drops.values() if d < 0),
                "worst_editor": worst,
                "worst_editor_rank_clean": clean_ranks.get(worst),
                # Does the board's leader lose first place by cueing only itself?
                "leader_loses_top1": bool(lost_top1),
                "family": "leaderboard_simulation",
            })
    return rows


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def _summarise(rows: Sequence[dict]) -> None:
    uniform = [r for r in rows if r["scenario"] == "all_editors"]
    single = [r for r in rows if r["scenario"] == "single_editor"]
    print("=== A1e: leaderboard simulation ===")
    print("\n  --- every editor carries the cue (uniform) ---")
    print(f"    {'judge':<18}{'source':<12}{'cue':<19}{'tau(clean,biased)':>19}"
          f"{'rank moves':>12}{'top1':>7}{'d tau vs human':>16}{'CI':>20}")
    for r in sorted(uniform, key=lambda r: (r["bias_type"], r["anchor_source"], r["judge_model"])):
        ci = (f"[{r['delta_tau_ci_low']}, {r['delta_tau_ci_high']}]"
              if r["delta_tau_ci_low"] is not None else "-")
        star = "*" if r["delta_tau_ci_excludes_zero"] else " "
        print(f"  {star} {r['judge_model']:<18}{r['anchor_source']:<12}{r['bias_type']:<19}"
              f"{str(r['kendall_tau_clean_vs_biased']):>19}"
              f"{str(r['n_rank_changes']) + '/' + str(r['n_editors']):>12}"
              f"{'CHANGED' if r['top1_changed'] else '-':>7}"
              f"{str(r['delta_tau_human']):>16}{ci:>20}")
    print("\n  --- one editor carries the cue alone ---")
    print(f"    {'judge':<18}{'source':<12}{'cue':<19}{'mean drop':>10}{'max drop':>9}"
          f"{'fell':>8}{'>=3':>5}{'leader loses #1':>17}")
    for r in sorted(single, key=lambda r: (r["bias_type"], r["anchor_source"], r["judge_model"])):
        print(f"    {r['judge_model']:<18}{r['anchor_source']:<12}{r['bias_type']:<19}"
              f"{r['mean_rank_drop']:>10}{r['max_rank_drop']:>9}"
              f"{str(r['n_editors_falling']) + '/' + str(r['n_editors']):>8}"
              f"{r['n_editors_falling_3_or_more']:>5}"
              f"{'YES' if r['leader_loses_top1'] else '-':>17}")


def main(argv: list[str] | None = None) -> int:
    root = default_root()
    manifests = root / "data" / "manifests"
    ap = argparse.ArgumentParser(description="WP-A1e editor-leaderboard simulation.")
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "v2")
    ap.add_argument("--samples", type=Path, default=manifests / "samples_judge_v2.jsonl")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    samples = io.read_jsonl(args.samples, SampleRecord)
    rows = build_leaderboard_rows(
        samples, args.results_dir, n_boot=args.n_boot, seed=args.seed
    )
    out_dir = Path(args.out_dir) if args.out_dir else args.results_dir / "metrics"
    write_csv(out_dir / "leaderboard_simulation.csv", rows)
    _summarise(rows)
    print(f"\nwrote -> {out_dir / 'leaderboard_simulation.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
