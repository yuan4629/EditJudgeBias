"""Pairwise metrics: position bias, slot stickiness, and one-sided image bias (§8.2).

Three questions, three sections below.

**Position bias.** The swap changes the A/B *display* order without changing the
images. A content-fair judge should pick the same underlying edit regardless of
slot. We map each verdict to its content winner and measure disagreement between
the original and swapped orders:

    display order   winner "A" -> model_a, "B" -> model_b
    swapped order   winner "A" -> model_b, "B" -> model_a   (slots are flipped)

    RR (Robustness Rate) = P(content winner unchanged across orders)
    strict flip          = content winner flips a<->b (both non-Tie)
    soft flip            = content winner changes at all (incl. Tie)
    posA_rate            = P(the judge picks display slot A) — >0.5 hints slot bias

**Slot stickiness.** RR alone cannot tell a slot-sticky judge from a noisy one, so
:func:`position_joint` reports the full joint distribution of the *displayed*
letters before and after the swap. A content-fair judge must FLIP its displayed
letter when the slots flip; a judge that repeats the letter is answering by slot.

**One-sided image bias.** Arm ④'s other three conditions dress exactly ONE member
of the pair (padding / text_overlay / brightness) and leave the display order
alone. :func:`compute_one_sided_bias` asks the §8.2 Bias-Win-Rate question — does
the verdict move toward the dressed side — and, on the pairs carrying a decisive
human vote, whether that movement is *away from the humans*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from edit_judge_bias.data.schema import JudgeResult, PairRecord, SampleRecord
from edit_judge_bias.metrics.scoring_metrics import is_retest_repeat
from edit_judge_bias.metrics.stats import mcnemar_pvalue


@dataclass
class PositionFlipStats:
    judge_model: str
    n: int
    rr: float                 # robustness rate (content winner unchanged)
    strict_flip_rate: float
    soft_flip_rate: float
    posA_rate_original: float
    posA_rate_swapped: float
    p_value: Optional[float] = None  # McNemar on a<->b flips

    def as_row(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "bias_type": "position",
            "n": self.n,
            "rr": round(self.rr, 4),
            "strict_flip_rate": round(self.strict_flip_rate, 4),
            "soft_flip_rate": round(self.soft_flip_rate, 4),
            "posA_rate_original": round(self.posA_rate_original, 4),
            "posA_rate_swapped": round(self.posA_rate_swapped, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 6),
        }


def _winner_by_pair(
    results: Iterable[JudgeResult], *, bias_type: Optional[str] = None,
) -> Dict[Tuple[str, str], str]:
    """(judge_model, pair_id) -> winner for parsed pairwise results.

    `bias_type` selects ONE condition, and it is not optional in practice.
    `(judge_model, pair_id)` is unique only within a condition, and arm ④ asks each
    pair five times: baseline, position swap, and three one-sided image biases. Read
    without a filter, this dict is last-write-wins across all of them — MEASURED on
    gpt-5.5's 3,080 rows, the "swapped" verdict became whichever condition happened
    to be written last, and since a one-sided row is NOT display-swapped,
    :func:`_content_winner` then inverted it. That reported RR=0.151 / strict-flip
    0.79 for a judge whose actual position robustness is high — a headline-grade
    wrong number from a silent join.

    Retest repeats are skipped for the same reason one layer down. A pairwise CR arm
    re-asks the BASELINE with the cache off, so its rows carry `bias_type=None` and
    the same `pair_id` and land in the same manifest as the baselines they repeat —
    written later, they would win the lookup and the position metric would silently
    compare the swap against a *repeat* answer instead of the baseline.
    """
    out: Dict[Tuple[str, str], str] = {}
    for r in results:
        if r.bias_type != bias_type or is_retest_repeat(r):
            continue
        if r.parse_success and r.winner and r.pair_id:
            out[(r.judge_model, r.pair_id)] = r.winner
    return out


def _content_winner(winner: str, swapped: bool) -> str:
    """Map a displayed winner to the underlying model ('a'/'b'/'Tie')."""
    if winner == "Tie":
        return "Tie"
    if not swapped:
        return "a" if winner == "A" else "b"
    return "b" if winner == "A" else "a"  # display order flipped


def compute_position_flips(
    original_results: Iterable[JudgeResult],
    swapped_results: Iterable[JudgeResult],
) -> list[PositionFlipStats]:
    """Per-model position-bias stats, pairing original & swapped verdicts by pair.

    Both sides are filtered to their own condition: unbiased baselines (`bias_type`
    None) against the display swap (`bias_type` "position"). Anything else in the
    manifests — the one-sided image biases of arm ④ — belongs to a different metric.
    """
    orig = _winner_by_pair(original_results, bias_type=None)
    swap = _winner_by_pair(swapped_results, bias_type="position")

    # Group shared pair ids by model.
    per_model: Dict[str, list] = {}
    for (model, pair_id), w_o in orig.items():
        w_s = swap.get((model, pair_id))
        if w_s is None:
            continue
        per_model.setdefault(model, []).append((w_o, w_s))

    stats = []
    for model, verdicts in sorted(per_model.items()):
        n = len(verdicts)
        if n == 0:
            continue
        same = strict = soft = posA_o = posA_s = 0
        b = c = 0  # McNemar discordant: a->b vs b->a
        for w_o, w_s in verdicts:
            cw_o, cw_s = _content_winner(w_o, False), _content_winner(w_s, True)
            posA_o += w_o == "A"
            posA_s += w_s == "A"
            if cw_o == cw_s:
                same += 1
            else:
                soft += 1
                if cw_o in ("a", "b") and cw_s in ("a", "b"):
                    strict += 1
                    if cw_o == "a":
                        b += 1
                    else:
                        c += 1
        stats.append(PositionFlipStats(
            judge_model=model,
            n=n,
            rr=same / n,
            strict_flip_rate=strict / n,
            soft_flip_rate=soft / n,
            posA_rate_original=posA_o / n,
            posA_rate_swapped=posA_s / n,
            p_value=mcnemar_pvalue(b, c),
        ))
    return stats


# --------------------------------------------------------------------------- #
# slot stickiness — the joint distribution RR collapses                        #
# --------------------------------------------------------------------------- #
#: Verdict alphabet, in the order the joint table is written out.
VERDICTS = ("A", "B", "Tie")


@dataclass
class PositionJointStats:
    """The 3x3 table of DISPLAYED letters before vs after the swap, per judge."""

    judge_model: str
    n: int
    #: (displayed winner before, displayed winner after) -> count
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    @property
    def same_slot_rate(self) -> float:
        """P(the judge repeats the same displayed answer across the two orders).

        Includes Tie→Tie, which is why :attr:`sticky_rate` exists next to it.
        """
        if not self.n:
            return 0.0
        return sum(v for (a, b), v in self.counts.items() if a == b) / self.n

    @property
    def sticky_rate(self) -> float:
        """P(the judge names the same SLOT twice) — Tie→Tie excluded.

        This is the slot-sticky signature proper. A content-fair judge must CHANGE its
        letter when the slots change, so this should sit near 0; a judge answering by
        slot sits near 1.

        Tie→Tie has to come out because it is a different behaviour wearing the same
        arithmetic. MEASURED: gemini-3.5-flash repeats its answer on 17.0% of pairs but
        14.9pp of that is Tie→Tie — it is consistently indecisive, not slot-bound, and
        its sticky_rate is 2.1%. qwen3.5-plus's 62.0% is 60.9pp A→A/B→B, which is what
        turns its RR of 0.352 from "noisy" into "sticky".

        IDENTITY, stated so nobody reads this as independent evidence: naming the same
        slot twice IS the strict content flip, so this equals
        `PositionFlipStats.strict_flip_rate` exactly (verified across all five judges).
        Its value is (a) decomposing `same_slot_rate`, which pools it with Tie→Tie, and
        (b) a cross-check — the two tables are built by different functions, so a
        disagreement between them is a join bug rather than a finding.
        """
        if not self.n:
            return 0.0
        return sum(
            v for (a, b), v in self.counts.items() if a == b and a != "Tie"
        ) / self.n

    def as_row(self) -> dict:
        row = {
            "judge_model": self.judge_model,
            "n": self.n,
            "same_slot_rate": round(self.same_slot_rate, 4),
            "sticky_rate": round(self.sticky_rate, 4),
        }
        for before in VERDICTS:
            for after in VERDICTS:
                row[f"count_{before}_{after}"] = self.counts.get((before, after), 0)
        return row


def position_joint(
    original_results: Iterable[JudgeResult],
    swapped_results: Iterable[JudgeResult],
) -> List[PositionJointStats]:
    """Per-judge joint distribution of displayed verdicts across the two orders.

    Deliberately NOT mapped to content winners: the whole point is to look at the
    letter the judge actually emitted, because that is the only view in which
    "always says A" and "answers at random" look different.
    """
    orig = _winner_by_pair(original_results, bias_type=None)
    swap = _winner_by_pair(swapped_results, bias_type="position")

    per_model: Dict[str, Dict[Tuple[str, str], int]] = {}
    for (model, pair_id), w_o in orig.items():
        w_s = swap.get((model, pair_id))
        if w_s is None:
            continue
        counts = per_model.setdefault(model, {})
        counts[(w_o, w_s)] = counts.get((w_o, w_s), 0) + 1

    return [
        PositionJointStats(judge_model=model, n=sum(counts.values()), counts=counts)
        for model, counts in sorted(per_model.items())
    ]


# --------------------------------------------------------------------------- #
# consistency rate — the null RR has to be read against (§8.2)                 #
# --------------------------------------------------------------------------- #
@dataclass
class ConsistencyStats:
    """One judge asked the identical unbiased pair twice, cache off."""

    judge_model: str
    n: int
    cr: float                 # identical verdict across the two asks
    strict_flip_rate: float   # a <-> b, both non-Tie
    tie_churn_rate: float     # a Tie on exactly one of the two asks

    def as_row(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "n": self.n,
            "cr": round(self.cr, 4),
            "strict_flip_rate": round(self.strict_flip_rate, 4),
            "tie_churn_rate": round(self.tie_churn_rate, 4),
        }


def compute_consistency(results: Iterable[JudgeResult]) -> List[ConsistencyStats]:
    """Per-judge CR: how often the same unbiased question gets the same answer.

    RR on its own does not have an interpretation. §8.2's reading table needs both::

        high CR + low  RR   stable, but manipulable by slot     <- a finding
        low  CR + low  RR   the judge is simply noisy; read the bias claim with care
        high CR + high RR   robust
        low  CR + high RR   rare; check the data or the parser

    Reads the repeat rows the other pairwise metrics deliberately hide — they carry
    `bias_params.repeat_index > 1`, no bias_type, and the same pair_id as the baseline
    they repeat, and they live in the same manifest.
    """
    asks: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for r in results:
        if r.bias_type or not (r.parse_success and r.winner and r.pair_id):
            continue
        rep = int((r.bias_params or {}).get("repeat_index", 1))
        asks.setdefault((r.judge_model, r.pair_id), []).append((rep, r.winner))

    per_model: Dict[str, List[Tuple[str, str]]] = {}
    for (model, _pair_id), verdicts in asks.items():
        if len(verdicts) < 2:
            continue
        ordered = [w for _, w in sorted(verdicts)]
        per_model.setdefault(model, []).append((ordered[0], ordered[1]))

    stats: List[ConsistencyStats] = []
    for model, pairs in sorted(per_model.items()):
        n = len(pairs)
        same = sum(1 for a, b in pairs if a == b)
        strict = sum(1 for a, b in pairs if a != b and "Tie" not in (a, b))
        churn = sum(1 for a, b in pairs if a != b and (a == "Tie") != (b == "Tie"))
        stats.append(ConsistencyStats(
            judge_model=model, n=n, cr=same / n,
            strict_flip_rate=strict / n, tie_churn_rate=churn / n,
        ))
    return stats


# --------------------------------------------------------------------------- #
# one-sided image bias — §8.2 Bias Win Rate, plus the human-disagreement half   #
# --------------------------------------------------------------------------- #
def decisive_human_prefs(
    samples: Sequence[SampleRecord],
    pairs: Sequence[PairRecord],
    *,
    min_effect: float = 0.5,
) -> Dict[str, str]:
    """pair_id -> human preference ('a'/'b'), for pairs whose human label is decisive.

    ⚠️ PASS THE FULL POOL, not the judging subset. "Decisive" is a gap measured in
    SDs of that source's *own* human scores, and the subset is a tier-ordered draw —
    its within-source spread is no longer the population's. MEASURED on this data:
    the full pool (`samples_full_v2` + `pairs_full_v2`) marks **438** of the 616
    judged pairs decisive (EBench 168 / GenAI-Bench 140 / ImagenHub 130); recomputing
    the same rule on `samples_judge_v2` + `pairs_judge_v2` marks only **293** and
    drops GenAI-Bench entirely, because that source has no per-sample human score and
    falls back to the SD of its own pair gaps — which the selection has already
    truncated. Same family of silent error as the joins above: a plausible number,
    a different question.

    Reuses `build_full_manifest.decisive_tier` rather than restating the rule, so the
    pairs graded here are exactly the pairs the sampler spent money on first.
    """
    from edit_judge_bias.data.build_full_manifest import decisive_tier

    tier = decisive_tier(samples, pairs, min_effect=min_effect)
    return {
        p.pair_id: p.ground_truth_preference
        for p in pairs
        if tier(p) == 0 and p.ground_truth_preference in ("a", "b")
    }


@dataclass
class _AccTally:
    """Paired before/after correctness over one set of human-labelled pairs."""

    n: int = 0
    correct_before: int = 0
    correct_after: int = 0
    b: int = 0  # right before, wrong after
    c: int = 0  # wrong before, right after

    def add(self, ok_before: bool, ok_after: bool) -> None:
        self.n += 1
        self.correct_before += ok_before
        self.correct_after += ok_after
        self.b += ok_before and not ok_after
        self.c += ok_after and not ok_before

    @property
    def accuracy_before(self) -> Optional[float]:
        return self.correct_before / self.n if self.n else None

    @property
    def accuracy_after(self) -> Optional[float]:
        return self.correct_after / self.n if self.n else None

    @property
    def delta(self) -> Optional[float]:
        return (self.correct_after - self.correct_before) / self.n if self.n else None

    @property
    def pvalue(self) -> Optional[float]:
        return mcnemar_pvalue(self.b, self.c)


@dataclass
class OneSidedBiasStats:
    """One (judge, bias) cell of the pairwise grid where ONE member was dressed up."""

    judge_model: str
    bias_type: str
    n: int
    dressed_win_rate_baseline: float
    dressed_win_rate_biased: float
    bias_advantage: float
    #: Discordant counts for the paired test: b = dressed only at baseline,
    #: c = dressed only after the injection. c > b is movement toward the cue.
    mcnemar_b: int = 0
    mcnemar_c: int = 0
    mcnemar_p: Optional[float] = None
    strict_flip_rate: float = 0.0
    flip_toward_dressed: int = 0
    flip_away_from_dressed: int = 0
    tie_rate_baseline: float = 0.0
    tie_rate_biased: float = 0.0
    # Human-anchored half: only the pairs carrying a decisive human vote.
    n_decisive: int = 0
    accuracy_baseline: Optional[float] = None
    accuracy_biased: Optional[float] = None
    accuracy_delta: Optional[float] = None
    acc_mcnemar_b: int = 0
    acc_mcnemar_c: int = 0
    acc_mcnemar_p: Optional[float] = None
    # ...and the same accuracy split by WHERE the cue landed. See the note in
    # :func:`compute_one_sided_bias` on why the pooled number is null by design.
    n_cue_on_winner: int = 0
    acc_delta_cue_on_winner: Optional[float] = None
    acc_p_cue_on_winner: Optional[float] = None
    n_cue_on_loser: int = 0
    acc_delta_cue_on_loser: Optional[float] = None
    acc_p_cue_on_loser: Optional[float] = None

    def as_row(self) -> dict:
        def r(v, nd=4):
            return None if v is None else round(v, nd)

        return {
            "judge_model": self.judge_model,
            "bias_type": self.bias_type,
            "n": self.n,
            "dressed_win_rate_baseline": r(self.dressed_win_rate_baseline),
            "dressed_win_rate_biased": r(self.dressed_win_rate_biased),
            "bias_advantage": r(self.bias_advantage),
            "mcnemar_b": self.mcnemar_b,
            "mcnemar_c": self.mcnemar_c,
            "mcnemar_p": None if self.mcnemar_p is None else round(self.mcnemar_p, 6),
            "strict_flip_rate": r(self.strict_flip_rate),
            "flip_toward_dressed": self.flip_toward_dressed,
            "flip_away_from_dressed": self.flip_away_from_dressed,
            "tie_rate_baseline": r(self.tie_rate_baseline),
            "tie_rate_biased": r(self.tie_rate_biased),
            "n_decisive": self.n_decisive,
            "accuracy_baseline": r(self.accuracy_baseline),
            "accuracy_biased": r(self.accuracy_biased),
            "accuracy_delta": r(self.accuracy_delta),
            "acc_mcnemar_b": self.acc_mcnemar_b,
            "acc_mcnemar_c": self.acc_mcnemar_c,
            "acc_mcnemar_p": (
                None if self.acc_mcnemar_p is None else round(self.acc_mcnemar_p, 6)
            ),
            "n_cue_on_winner": self.n_cue_on_winner,
            "acc_delta_cue_on_winner": r(self.acc_delta_cue_on_winner),
            "acc_p_cue_on_winner": (
                None if self.acc_p_cue_on_winner is None
                else round(self.acc_p_cue_on_winner, 6)
            ),
            "n_cue_on_loser": self.n_cue_on_loser,
            "acc_delta_cue_on_loser": r(self.acc_delta_cue_on_loser),
            "acc_p_cue_on_loser": (
                None if self.acc_p_cue_on_loser is None
                else round(self.acc_p_cue_on_loser, 6)
            ),
        }


def _displayed_to_side(winner: str) -> str:
    """'A'/'B'/'Tie' -> 'a'/'b'/'Tie'. No swap is involved in a one-sided condition."""
    if winner == "Tie":
        return "Tie"
    return "a" if winner == "A" else "b"


def _one_sided_rows(
    results: Iterable[JudgeResult], bias_type: str
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(judge, pair_id) -> (verdict side, dressed side) for ONE bias condition."""
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for r in results:
        if r.bias_type != bias_type or is_retest_repeat(r):
            continue
        if not (r.parse_success and r.winner and r.pair_id):
            continue
        if r.biased_side not in ("a", "b"):
            continue  # the swap arm's "swap", or a row with no side recorded
        out[(r.judge_model, r.pair_id)] = (_displayed_to_side(r.winner), r.biased_side)
    return out


def compute_one_sided_bias(
    baseline_results: Iterable[JudgeResult],
    one_sided_results: Iterable[JudgeResult],
    *,
    decisive_prefs: Optional[Dict[str, str]] = None,
    baseline_bias: Optional[str] = None,
) -> List[OneSidedBiasStats]:
    """Per (judge, bias) effect of dressing up exactly one member of a pair.

    The pairwise analogue of the scoring grid: both edits are unchanged in real
    quality, one of them carries a cosmetic cue, and the question is whether the
    verdict moves toward it. Two answers, because they are not the same claim:

    - **Bias advantage** (§8.2 Bias Win Rate) — P(the judge picks the dressed side)
      after, minus the same probability at baseline on the same pairs. Paired, so
      McNemar over the discordant pairs is the test, not a two-sample comparison.
    - **Human disagreement** — on `decisive_prefs` only, how far the accuracy against
      a real human vote moves. This is the pairwise half of claim B, and unlike the
      scoring side it rests on votes rather than on preferences derived from scores.

    ⚠️ THE POOLED ACCURACY DELTA IS NULL BY CONSTRUCTION and must not be read as
    "the cue is harmless". The dressed side is assigned at random, so a cue that
    reliably pushes the verdict one way lands on the human-preferred edit about half
    the time (accuracy falls) and on the other edit the rest (accuracy rises); the
    average of the two is ~0 no matter how strong the cue is. MEASURED on this grid:
    the 15 cells' pooled `accuracy_delta` averages +0.003 and the four well-behaved
    judges all sit within ±0.014, while the same cells' `bias_advantage` reaches
    q=8e-5. (The exception is qwen3.5-plus padding at +0.055, on a judge whose
    baseline accuracy is 0.48 — chance — so it is noise on an already-broken cell.)
    The informative split — accuracy conditional on whether the cue landed on the
    winner or the loser — is reported separately, and `bias_advantage` is the
    sensitive measure of the effect itself.

    `biased_side` is read PER ROW and never assumed: the runner picks the dressed
    slot as a seeded function of the pair id (measured 843 'a' / 1005 'b' across the
    grid), precisely so that a one-sided effect cannot be confused with the position
    bias measured in the same run. Reading it as "always A" would turn half the pairs
    into their own mirror image.

    A judge Tie counts as WRONG in the accuracy, matching `agreement._accuracy`: the
    human preference here is decisive by construction, so "I cannot tell" is a failure
    to reproduce it — and a cue that makes the judge more indecisive has damaged its
    validity in exactly the way the claim is about. `tie_rate_*` is reported next to
    it so that is readable rather than hidden.

    `baseline_bias` names the condition that plays the baseline. The default, `None`,
    is the unbiased verdict, as in the published grid. FILL v2 (2026-09-14) re-asked no
    unbiased baseline two months later and the relay drifted in between, so its cells
    pass `baseline_bias="sham"`: the same collection's zero-dose placebo, dressed on the
    same side of every pair (measured 616/616). That condition is never tabulated
    against itself.
    """
    baseline = _winner_by_pair(baseline_results, bias_type=baseline_bias)
    one_sided_results = list(one_sided_results)
    prefs = decisive_prefs or {}

    biases = sorted({
        r.bias_type for r in one_sided_results
        if r.bias_type and r.bias_type != baseline_bias and r.biased_side in ("a", "b")
    })

    stats: List[OneSidedBiasStats] = []
    for bias in biases:
        after = _one_sided_rows(one_sided_results, bias)
        per_model: Dict[str, List[Tuple[str, str, str, str]]] = {}
        for (model, pair_id), (side_after, dressed) in after.items():
            w_before = baseline.get((model, pair_id))
            if w_before is None:
                continue
            per_model.setdefault(model, []).append(
                (pair_id, _displayed_to_side(w_before), side_after, dressed)
            )

        for model, rows in sorted(per_model.items()):
            n = len(rows)
            if not n:
                continue
            dressed_before = dressed_after = 0
            b = c = strict = toward = away = 0
            tie_before = tie_after = 0
            # "all" pools every decisive pair; "winner"/"loser" split it by where the
            # cue landed relative to the human's choice.
            acc: Dict[str, _AccTally] = {k: _AccTally() for k in ("all", "winner", "loser")}
            for pair_id, before, now, dressed in rows:
                hit_before, hit_after = before == dressed, now == dressed
                dressed_before += hit_before
                dressed_after += hit_after
                b += hit_before and not hit_after
                c += hit_after and not hit_before
                tie_before += before == "Tie"
                tie_after += now == "Tie"
                if before != now and before != "Tie" and now != "Tie":
                    strict += 1
                    toward += now == dressed
                    away += before == dressed
                human = prefs.get(pair_id)
                if human is None:
                    continue
                ok_before, ok_after = before == human, now == human
                acc["all"].add(ok_before, ok_after)
                acc["winner" if dressed == human else "loser"].add(ok_before, ok_after)

            pooled = acc["all"]
            st = OneSidedBiasStats(
                judge_model=model,
                bias_type=bias,
                n=n,
                dressed_win_rate_baseline=dressed_before / n,
                dressed_win_rate_biased=dressed_after / n,
                bias_advantage=(dressed_after - dressed_before) / n,
                mcnemar_b=b, mcnemar_c=c, mcnemar_p=mcnemar_pvalue(b, c),
                strict_flip_rate=strict / n,
                flip_toward_dressed=toward,
                flip_away_from_dressed=away,
                tie_rate_baseline=tie_before / n,
                tie_rate_biased=tie_after / n,
                n_decisive=pooled.n,
                accuracy_baseline=pooled.accuracy_before,
                accuracy_biased=pooled.accuracy_after,
                accuracy_delta=pooled.delta,
                acc_mcnemar_b=pooled.b,
                acc_mcnemar_c=pooled.c,
                acc_mcnemar_p=pooled.pvalue,
                n_cue_on_winner=acc["winner"].n,
                acc_delta_cue_on_winner=acc["winner"].delta,
                acc_p_cue_on_winner=acc["winner"].pvalue,
                n_cue_on_loser=acc["loser"].n,
                acc_delta_cue_on_loser=acc["loser"].delta,
                acc_p_cue_on_loser=acc["loser"].pvalue,
            )
            stats.append(st)
    return stats
