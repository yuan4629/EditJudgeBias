"""Fill the partial condition grids: every judge in one process, resumable, retrying.

WHAT THIS IS FOR
The published pairwise one-sided arm ran 3 cues and the anchor arm 4.
`configs/experiment/fill_v2.yaml` names the arms that add the rest, and this module
collects them. It does NOT decide what is asked: targets, prompts and rows come from
`run_scoring_judge` / `run_pairwise_judge` themselves (`_build_targets`, `build_request`,
`build_result`), so a fill row is the row those runners would have written for the same
condition, byte for byte.

    PYTHONPATH=src python -m edit_judge_bias.experiments.fill_runner plan
    PYTHONPATH=src python -m edit_judge_bias.experiments.fill_runner run --use-api --per-condition 2
    PYTHONPATH=src python -m edit_judge_bias.experiments.fill_runner run --use-api
    PYTHONPATH=src python -m edit_judge_bias.experiments.fill_runner status | stop | verify


WHAT IT ADDS OVER RUNNING THE TWO RUNNERS ONCE PER JUDGE
1. Many judges in ONE process sharing one adaptive budget of in-flight upload BYTES.
   2026-08-17: this link is uplink-bound and the limit is bytes, not calls -- five
   processes x 8 workers of ~2.7 MB payloads gave 88 transport failures and four judges
   with zero rows. One process sees every upload it has in flight, so it caps the bytes
   directly and shrinks the cap multiplicatively on transport errors (AIMD).
2. Retries that do not hold a worker hostage. The adapter's own retry sleeps inside the
   worker; here a failed call returns its slot and re-enters its lane with a jittered
   delay, so one bad channel cannot starve the healthy lanes.
3. Failures CLASSIFIED, because they need different responses: transport -> shrink the
   byte budget; rate_limit -> shrink that lane and cool it; channel ("no available
   channel") -> pause that lane, keep the job; refusal (PROHIBITED_CONTENT,
   content_policy_violation, DataInspectionFailed) -> a few spaced re-asks, then record it
   (gpt-4o's filter is probabilistic, Gemini's and Aliyun's permanent); quota / auth ->
   stop everything, so an empty wallet costs a few errors, not a night of retries.
4. Rounds. A job that exhausts its attempts is parked; when all else is done it is asked
   again with a longer timeout. A too-short timeout disguises a slow success (08-17) and
   can disguise a permanent refusal (07-29). Timeouts are transport settings, absent from
   the response-cache key, so raising them cannot change an answer.
5. Resume that survives a kill at any instant. A row is appended (flush + fsync) only
   after its raw text is on disk; a torn last line is cut off and kept at start-up; the
   response cache makes re-asking a torn row free. One OS lock per tree, so a second copy
   cannot write next to the first (the pilot's duplicate-row failure).
6. It never writes to results/v2. Fill rows join the published tree only by an explicit
   merge after collection has ended; configs/experiment/pairwise_fill_v2.yaml says why.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import http.client
import itertools
import json
import os
import random
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from edit_judge_bias.data import io
from edit_judge_bias.data.manifest_utils import default_root
from edit_judge_bias.experiments import run_pairwise_judge as rp
from edit_judge_bias.experiments import run_scoring_judge as rs
from edit_judge_bias.experiments.judge_common import manifest_for, save_raw_response
from edit_judge_bias.judges import build_adapter
from edit_judge_bias.judges.base import JudgeRequest
from edit_judge_bias.judges.parser import DEFAULT_SCORE_SCALE

#: Trees that hold published rows. A plan that points its output at one is refused.
PUBLISHED_TREES = ("results", "results/v2")

STATUS_FILE = "fill_status.json"
STOP_FILE = "STOP"
LOCK_FILE = ".fill.lock"
UNRESOLVED_FILE = "fill_unresolved.jsonl"

EXIT_COMPLETE, EXIT_UNRESOLVED, EXIT_SAFETY_STOP, EXIT_REFUSED, EXIT_USER_STOP = 0, 2, 3, 4, 5


# =========================================================================== #
# failure classification                                                      #
# =========================================================================== #
class CallError(Exception):
    """One failed attempt, with the kind that decides what happens next."""

    def __init__(self, kind: str, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


# Error-body substrings used to classify failures. The Chinese entries are the messages
# returned by common self-hosted OpenAI-compatible gateways (quota / token / no channel).
_QUOTA = ("额度不足", "余额不足", "额度已用尽", "已用尽", "用户额度", "令牌额度", "quota exceeded",
          "insufficient_quota", "insufficient quota", "exceeded your current quota",
          "not enough quota")
_AUTH = ("invalid_api_key", "incorrect api key", "无效的令牌", "令牌已过期", "无效令牌",
         "invalid token", "token expired")
_REFUSAL = ("prohibited_content", "content_policy_violation", "datainspectionfailed",
            "data_inspection_failed", "inappropriate content", "content_filter",
            "safety system", "responsibleaipolicyviolation")
_CHANNEL = ("无可用渠道", "no available channel", "负载已饱和")
_MODEL = ("model_not_found", "does not exist")


def classify_http(status: Optional[int], body: str) -> str:
    """Map an HTTP status + body to a failure kind (see the module docstring, item 3)."""
    low = body.lower()

    def has(markers: Sequence[str]) -> bool:
        return any(m in body or m in low for m in markers)

    if has(_QUOTA):
        return "quota"
    if status == 401 or has(_AUTH):
        return "auth"
    if has(_REFUSAL):
        return "refusal"
    if has(_CHANNEL) or status == 403:
        return "channel"
    if status == 429:
        return "rate_limit"
    if status == 404 or has(_MODEL):
        return "model_missing"
    return "server"


# =========================================================================== #
# the network call                                                            #
# =========================================================================== #
def _post(endpoint: str, api_key: str, body: bytes, timeout: float) -> str:
    """One POST with the adapter's exact headers; only the timeout is per call."""
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


Caller = Callable[[JudgeRequest, float], Tuple[str, bool]]


class AdapterCaller:
    """One attempt per call against a judge adapter; returns (text, served_from_cache).

    For the real OpenAI-compatible adapter this re-uses the adapter's own payload builder,
    cache key, cache path and usage log, and replaces only its retry loop -- so what is
    sent, and what a cache hit returns, is exactly what `adapter.generate` would have done.
    Any other adapter (the mock) is simply asked.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, request: JudgeRequest, timeout: float) -> Tuple[str, bool]:
        from edit_judge_bias.judges.openai_judge import (
            JudgeAPIError,
            OpenAICompatibleJudge,
            _read_error_body,
        )

        a = self.adapter
        if not isinstance(a, OpenAICompatibleJudge):
            return a.generate(request), False
        payload = a._build_payload(request)
        cache_path = a._cache_path(a._cache_key(payload))
        if cache_path is not None and cache_path.is_file():
            return cache_path.read_text(encoding="utf-8"), True
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            text_body = _post(a._endpoint, a._api_key, body, timeout)
        except urllib.error.HTTPError as e:
            err = _read_error_body(e)
            raise CallError(classify_http(e.code, err), f"HTTP {e.code}: {err[:300]}", e.code) from None
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
                http.client.HTTPException, ssl.SSLError, OSError) as e:
            raise CallError("transport", f"{type(e).__name__}: {e}"[:300]) from None
        try:
            data = json.loads(text_body)
        except json.JSONDecodeError:
            raise CallError("server", f"non-JSON body: {text_body[:200]}") from None
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            msg = json.dumps(data.get("error"), ensure_ascii=False)
            raise CallError(classify_http(None, msg), f"200 carrying an error: {msg[:300]}")
        a._log_usage(data)
        try:
            text = a._extract_text(data)
        except JudgeAPIError as e:
            raise CallError("empty", str(e)[:300]) from None
        if not isinstance(text, str) or not text.strip():
            try:
                finish = data["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError, AttributeError):
                finish = None
            raise CallError("empty", f"empty message content (finish_reason={finish})")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return text, False


class FaultInjectingCaller:
    """REHEARSAL ONLY: fail a seeded share of attempts, to exercise every recovery path."""

    def __init__(self, inner: Caller, *, seed: int = 0, transport: float = 0.04,
                 server: float = 0.03, rate_limit: float = 0.01, channel: float = 0.002,
                 permanent_refusal: float = 0.0005, latency_s: Tuple[float, float] = (0.0, 0.01)):
        self.inner = inner
        self.rates = (("transport", transport), ("server", server),
                      ("rate_limit", rate_limit), ("channel", channel))
        self.permanent_refusal = permanent_refusal
        self.latency_s = latency_s
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __call__(self, request: JudgeRequest, timeout: float) -> Tuple[str, bool]:
        with self._lock:
            r = self._rng.random()
            lat = self._rng.uniform(*self.latency_s)
        if lat:
            time.sleep(lat)
        key = hashlib.sha256((request.prompt + "|" + "|".join(map(str, request.images)))
                             .encode("utf-8")).digest()
        if int.from_bytes(key[:4], "big") / 2 ** 32 < self.permanent_refusal:
            raise CallError("refusal", "rehearsal: HTTP 400 PROHIBITED_CONTENT", 400)
        acc = 0.0
        for kind, p in self.rates:
            acc += p
            if r < acc:
                raise CallError(kind, f"rehearsal: injected {kind}")
        return self.inner(request, timeout)


# =========================================================================== #
# plan                                                                        #
# =========================================================================== #
@dataclass
class Settings:
    max_inflight: int = 48
    bytes_budget_mb: float = 24.0
    min_bytes_budget_mb: float = 3.0
    max_bytes_budget_mb: float = 96.0
    bytes_step_mb: float = 1.0
    increase_every: int = 25
    decrease_factor: float = 0.7
    decrease_gap_s: float = 5.0
    starvation_s: float = 20.0
    min_lane_limit: float = 1.0
    attempts_per_round: int = 6
    rounds: int = 4
    timeout_multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    timeout_cap_s: float = 300.0
    backoff_base_s: float = 4.0
    backoff_cap_s: float = 180.0
    refusal_attempts: int = 3
    refusal_delay_s: float = 90.0
    empty_attempts: int = 3
    rate_limit_cooldown_s: float = 30.0
    channel_cooldown_s: float = 120.0
    server_streak: int = 8
    server_cooldown_s: float = 30.0
    network_down_after: int = 30
    network_pause_s: float = 30.0
    network_pause_cap_s: float = 300.0
    quota_errors_to_stop: int = 3
    quota_window_s: float = 180.0
    quota_cooldown_s: float = 60.0
    lane_giveup_s: float = 3 * 3600.0
    local_errors_to_stop: int = 20
    max_estimated_usd: Optional[float] = None
    status_every_s: float = 15.0
    progress_every_s: float = 300.0

    @classmethod
    def from_blocks(cls, raw: dict) -> "Settings":
        s = cls()
        for block in ("transport", "retry", "tripwire", "reporting"):
            for key, value in (raw.get(block) or {}).items():
                if not hasattr(s, key):
                    raise ValueError(f"unknown fill setting {block}.{key}")
                if key == "timeout_multipliers":
                    value = tuple(float(x) for x in value)
                setattr(s, key, value)
        return s


@dataclass
class ArmSpec:
    name: str
    task: str
    config: str
    expect_new_per_judge: Optional[int] = None


@dataclass
class LaneSpec:
    config: str
    concurrency: int
    max_concurrency: int
    unit_price_usd: float


@dataclass
class Plan:
    root: Path
    tree_rel: str
    tree: Path
    done_trees: List[Path]
    arms: List[ArmSpec]
    lanes: List[LaneSpec]
    priority: List[str]
    settings: Settings
    failure_log: Path
    progress_log: Path
    tree_overridden: bool


def load_plan(plan_path, *, root: Optional[Path] = None, tree: Optional[str] = None) -> Plan:
    root = Path(root) if root is not None else default_root()
    raw = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
    tree_rel = tree or raw["tree"]
    if Path(tree_rel).is_absolute():
        raise ValueError("the fill tree must be a path relative to the project root")
    tree_abs = root / tree_rel
    for pub in PUBLISHED_TREES:
        if tree_abs.resolve() == (root / pub).resolve():
            raise ValueError(f"refusing to write into the published tree {pub!r}; fill rows "
                             "join it only through an explicit merge after collection")
    done_trees = [root / t for t in (raw.get("done_trees") or [])]
    if any(t.resolve() == tree_abs.resolve() for t in done_trees):
        raise ValueError("the fill tree cannot also be one of done_trees")
    if tree:  # a rehearsal keeps its logs next to its rows
        failure_log, progress_log = tree_abs / "logs" / "failures.jsonl", tree_abs / "logs" / "progress.log"
    else:
        failure_log, progress_log = root / raw["failure_log"], root / raw["progress_log"]
    arms = [ArmSpec(a["name"], a["task"], a["config"], a.get("expect_new_per_judge"))
            for a in raw["arms"]]
    for a in arms:
        if a.task not in ("scoring", "pairwise"):
            raise ValueError(f"arm {a.name}: task must be scoring or pairwise")
    lanes = [LaneSpec(j["config"], int(j.get("concurrency", 4)),
                      int(j.get("max_concurrency", j.get("concurrency", 4))),
                      float(j.get("unit_price_usd", 0.0))) for j in raw["judges"]]
    return Plan(root=root, tree_rel=Path(tree_rel).as_posix(), tree=tree_abs,
                done_trees=done_trees, arms=arms, lanes=lanes,
                priority=list(raw.get("priority") or []),
                settings=Settings.from_blocks(raw), failure_log=failure_log,
                progress_log=progress_log, tree_overridden=bool(tree))


@dataclass(eq=False)
class Job:
    lane: str
    arm: str
    task: str
    condition: str
    rank: int
    order: int
    result_id: str
    target: Any
    style: str
    score_scale: Optional[int]
    biased: bool
    nbytes: int = 0
    attempts: int = 0
    refusals: int = 0
    empties: int = 0
    timeouts: int = 0
    last_kind: Optional[str] = None

    def request(self, root: Path) -> Tuple[str, List[Path]]:
        if self.task == "scoring":
            return rs.build_request(self.target, style=self.style,
                                    score_scale=int(self.score_scale), root=root)
        return rp.build_request(self.target, style=self.style, root=root)


@dataclass
class LanePlan:
    name: str
    spec: LaneSpec
    judge_cfg: dict
    base_timeout: float
    jobs: List[Job]
    per_arm: Dict[str, dict]


def _arm_targets(arm: ArmSpec, judge_config: str, plan: Plan):
    """Targets exactly as the arm's own runner would build them for this judge."""
    root = plan.root
    cfg = yaml.safe_load((root / arm.config).read_text(encoding="utf-8"))
    if not plan.tree_overridden:
        for key, want in (("results_dir", plan.tree_rel), ("raw_dir", f"{plan.tree_rel}/raw_responses")):
            if Path(str(cfg.get(key, ""))).as_posix() != want:
                raise ValueError(f"{arm.config}: {key} is {cfg.get(key)!r}, the plan writes to {want!r}")
    if int(cfg.get("repeat", 1)) > 1:
        raise ValueError(f"{arm.config}: a fill arm must not repeat")
    cfg = {**cfg, "judge_config": judge_config, "judge": None}
    if arm.task == "scoring":
        score_scale = int(cfg.get("score_scale", DEFAULT_SCORE_SCALE))
        judge_cfg = rs._resolve_judge_cfg(cfg, root)
        style = rs.resolve_prompt_style(cfg, judge_cfg, None)
        judge_cfg = {"score_scale": score_scale, **judge_cfg}
        model = judge_cfg.get("model_name", "mock-judge")
        targets = rs._build_targets(cfg, root, model, style)
        misses = sum(1 for t in targets if t is rs._MISSING)
        targets = [t for t in targets if t is not rs._MISSING]
    else:
        score_scale = None
        judge_cfg = rp._resolve_judge_cfg(cfg, root)
        style = cfg.get("prompt_style", "vanilla")
        model = judge_cfg.get("model_name", "mock-judge")
        targets = rp._build_targets(cfg, root, model, style)
        misses = 0
    return model, style, score_scale, judge_cfg, targets, misses


def load_done(trees: Sequence[Path], task: str, model: str) -> Set[str]:
    done: Set[str] = set()
    for tree in trees:
        for biased in (False, True):
            path = manifest_for(tree, task, model, biased=biased)
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            done.add(json.loads(line)["result_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass  # a torn tail; `repair_tree` deals with the fill tree's
    return done


def _spread(items: List[Job], n: int) -> List[Job]:
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[min(len(items) - 1, int(i * step))] for i in range(n)]


def build_lanes(plan: Plan, *, judges: Optional[Sequence[str]] = None,
                arms: Optional[Sequence[str]] = None, per_condition: Optional[int] = None,
                ) -> Tuple[List[LanePlan], List[str]]:
    """Every pending job per judge, with the problems that must stop a run."""
    problems: List[str] = []
    lanes: List[LanePlan] = []
    size_cache: Dict[Path, int] = {}
    wanted_arms = [a for a in plan.arms if not arms or a.name in arms]
    for spec in plan.lanes:
        judge_file = yaml.safe_load((plan.root / spec.config).read_text(encoding="utf-8"))
        name = judge_file.get("model_name", "mock-judge")
        if judges and name not in judges:
            continue
        lane_jobs: List[Job] = []
        per_arm: Dict[str, dict] = {}
        lane_cfg: Optional[dict] = None
        seen: Set[str] = set()
        for arm_idx, arm in enumerate(wanted_arms):
            model, style, scale, judge_cfg, targets, misses = _arm_targets(arm, spec.config, plan)
            # Prefer the scoring arm's resolved config: it carries `score_scale`, which the
            # real adapter ignores but the mock needs to answer on the scale it is asked.
            if lane_cfg is None or ("score_scale" in judge_cfg and "score_scale" not in lane_cfg):
                lane_cfg = judge_cfg
            if misses:
                problems.append(f"{name}/{arm.name}: {misses} biased records join no sample")
            done_pub = load_done(plan.done_trees, arm.task, model)
            done_fill = load_done([plan.tree], arm.task, model)
            new = [t for t in targets if t.result_id not in done_pub]
            if arm.expect_new_per_judge is not None and len(new) != arm.expect_new_per_judge:
                problems.append(f"{name}/{arm.name}: {len(new)} targets are not yet in the "
                                f"published tree, the plan expects {arm.expect_new_per_judge}")
            pending = [t for t in new if t.result_id not in done_fill]
            by_cond: Dict[str, List[Job]] = defaultdict(list)
            for i, t in enumerate(pending):
                if t.result_id in seen:
                    problems.append(f"{name}: result_id {t.result_id} appears in two arms")
                seen.add(t.result_id)
                cond = t.bias_type or "baseline"
                rank = plan.priority.index(cond) if cond in plan.priority else len(plan.priority)
                job = Job(lane=name, arm=arm.name, task=arm.task, condition=cond, rank=rank,
                          order=arm_idx * 10_000_000 + i, result_id=t.result_id, target=t,
                          style=style, score_scale=scale, biased=bool(t.bias_type))
                by_cond[cond].append(job)
            chosen: List[Job] = []
            for cond, jobs in by_cond.items():
                chosen.extend(_spread(jobs, per_condition) if per_condition else jobs)
            for job in chosen:
                prompt, images = job.request(plan.root)
                total = 0
                for img in images:
                    if img not in size_cache:
                        size_cache[img] = img.stat().st_size if img.is_file() else -1
                    if size_cache[img] < 0:
                        problems.append(f"{name}/{arm.name}: missing image {img}")
                        total = 0
                        break
                    total += size_cache[img]
                job.nbytes = int(total * 4 / 3) + len(prompt.encode("utf-8"))
            lane_jobs.extend(chosen)
            per_arm[arm.name] = {
                "targets": len(targets), "in_published_tree": len(targets) - len(new),
                "new": len(new), "in_fill_tree": len(new) - len(pending),
                "pending": len(pending), "selected": len(chosen),
                "by_condition": dict(sorted(Counter(j.condition for j in chosen).items())),
            }
        lane_jobs.sort(key=lambda j: (j.rank, j.order))
        cfg = dict(lane_cfg or judge_file)
        lanes.append(LanePlan(name=name, spec=spec, judge_cfg=cfg,
                              base_timeout=float(cfg.get("timeout", 60.0)),
                              jobs=lane_jobs, per_arm=per_arm))
    names = [l.name for l in lanes]
    if len(set(names)) != len(names):
        problems.append(f"two lanes share a judge name: {names}")
    if judges:
        missing = sorted(set(judges) - set(names))
        if missing:
            problems.append(f"--judges names no lane in the plan: {missing}")
    return lanes, problems


# =========================================================================== #
# on-disk safety: lock, torn tails, duplicates                                #
# =========================================================================== #
class TreeLock:
    """An OS-level exclusive lock on <tree>/.fill.lock, released when the process dies."""

    def __init__(self, tree: Path):
        self.path = tree / LOCK_FILE
        self._fh = None

    def acquire(self) -> "TreeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError(f"another fill_runner holds {self.path}") from None
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._fh.close()
        self._fh = None


def _manifests(tree: Path) -> List[Path]:
    return sorted(p for sub in ("raw_judgments", "biased_judgments")
                  for p in (tree / sub).glob("*.jsonl"))


def repair_tree(tree: Path) -> Tuple[List[str], List[str]]:
    """Cut torn last lines (kept beside the file), then refuse on anything worse.

    Returns (notes, problems). A torn tail is what a kill mid-append leaves; the job it
    belonged to is simply not done yet and will be re-asked (free, from the cache). A bad
    line anywhere ELSE, or a duplicated result_id, is not a crash artefact -- stop.
    """
    notes: List[str] = []
    problems: List[str] = []
    for path in _manifests(tree):
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            keep = path.with_name(f"{path.name}.torn-{datetime.now():%Y%m%d-%H%M%S}")
            keep.write_bytes(data[cut:])
            with path.open("r+b") as fh:
                fh.truncate(cut)
            notes.append(f"cut a torn last line from {path.name} (kept as {keep.name})")
            data = data[:cut]
        ids: Counter = Counter()
        for n, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                ids[json.loads(line)["result_id"]] += 1
            except (json.JSONDecodeError, KeyError):
                problems.append(f"{path.name}:{n} is not a result row")
        dups = [rid for rid, c in ids.items() if c > 1]
        if dups:
            problems.append(f"{path.name}: {len(dups)} duplicated result_ids, e.g. {dups[0]}")
    return notes, problems


# =========================================================================== #
# scheduler                                                                   #
# =========================================================================== #
@dataclass
class Outcome:
    kind: str
    message: str = ""
    status: Optional[int] = None
    cached: bool = False
    parse_ok: bool = True
    seconds: float = 0.0
    duplicate: bool = False


@dataclass(eq=False)
class LaneState:
    name: str
    limit: float
    max_limit: int
    base_timeout: float
    unit_price: float
    ready: List[Tuple[int, int, Job]] = field(default_factory=list)
    delayed: List[Tuple[float, int, Job]] = field(default_factory=list)
    inflight: int = 0
    cooldown_until: float = 0.0
    disabled: Optional[str] = None
    streak: int = 0
    server_streak: int = 0
    last_decrease: float = 0.0
    blocked_since: Optional[float] = None
    last_progress: float = 0.0
    errors_since_progress: int = 0
    ok: int = 0
    network_calls: int = 0
    cache_hits: int = 0
    parse_failures: int = 0
    kinds: Counter = field(default_factory=Counter)
    ok_times: Deque[float] = field(default_factory=deque)
    total: int = 0


class Scheduler:
    """All dispatch decisions, under one condition variable. Workers never decide."""

    def __init__(self, lanes: List[LanePlan], settings: Settings, *,
                 clock: Callable[[], float] = time.monotonic, seed: int = 42):
        self.s = settings
        self.clock = clock
        self.cv = threading.Condition()
        self._seq = itertools.count()
        self._rng = random.Random(seed)
        now = clock()
        self.started = now
        self.lanes: Dict[str, LaneState] = {}
        self.totals: Dict[Tuple[str, str, str], int] = Counter()
        self.done_by: Dict[Tuple[str, str, str], int] = Counter()
        for lp in lanes:
            st = LaneState(name=lp.name, limit=float(lp.spec.concurrency),
                           max_limit=int(lp.spec.max_concurrency),
                           base_timeout=lp.base_timeout, unit_price=lp.spec.unit_price_usd,
                           last_progress=now, total=len(lp.jobs))
            for job in lp.jobs:
                heapq.heappush(st.ready, (job.rank, job.order, job))
                self.totals[(lp.name, job.arm, job.condition)] += 1
            self.lanes[lp.name] = st
        self.round = 1
        self.deferred: List[Job] = []
        self.final: List[Tuple[Job, Outcome]] = []
        self.inflight = 0
        self.bytes_inflight = 0
        self.bytes_budget = settings.bytes_budget_mb * 1e6
        self.last_budget_decrease = 0.0
        self.global_streak = 0
        self.transport_streak = 0
        self.paused_until = 0.0
        self.probe_mode = False
        self.pause_s = settings.network_pause_s
        self.quota_events: Deque[float] = deque()
        self.local_errors = 0
        self.stop_reason: Optional[str] = None
        self.finished = False

    # ------------------------------------------------------------ helpers
    def timeout_for(self, job: Job) -> float:
        lane = self.lanes[job.lane]
        mults = self.s.timeout_multipliers or (1.0,)
        t = lane.base_timeout * mults[min(self.round - 1, len(mults) - 1)]
        t *= 1.5 ** min(job.timeouts, 3)
        return float(min(self.s.timeout_cap_s, t))

    def stop(self, reason: str) -> None:
        with self.cv:
            if self.stop_reason is None:
                self.stop_reason = reason
            self.cv.notify_all()

    def _backoff(self, attempts: int) -> float:
        base = min(self.s.backoff_cap_s, self.s.backoff_base_s * (2 ** max(0, attempts - 1)))
        return base * self._rng.uniform(0.7, 1.3)

    def _push_ready(self, lane: LaneState, job: Job) -> None:
        heapq.heappush(lane.ready, (job.rank, job.order, job))

    def _push_delayed(self, lane: LaneState, job: Job, when: float) -> None:
        heapq.heappush(lane.delayed, (when, next(self._seq), job))

    def _promote(self, now: float) -> None:
        for lane in self.lanes.values():
            while lane.delayed and lane.delayed[0][0] <= now:
                _, _, job = heapq.heappop(lane.delayed)
                self._push_ready(lane, job)
            if (not lane.disabled and (lane.ready or lane.delayed) and lane.errors_since_progress
                    and now - lane.last_progress > self.s.lane_giveup_s):
                lane.disabled = f"no success for {self.s.lane_giveup_s / 3600:.1f} h"

    def _idle(self) -> bool:
        return self.inflight == 0 and all(
            lane.disabled or (not lane.ready and not lane.delayed)
            for lane in self.lanes.values())

    def _pick(self, now: float) -> Optional[Job]:
        if self.inflight >= self.s.max_inflight or now < self.paused_until:
            return None
        if self.probe_mode and self.inflight > 0:
            return None
        open_lanes = [l for l in self.lanes.values()
                      if not l.disabled and l.ready and now >= l.cooldown_until
                      and l.inflight < max(1, int(l.limit))]
        open_lanes.sort(key=lambda l: (l.inflight / max(1, int(l.limit)), l.name))
        # A big job at the head of a lane must not wait behind an endless stream of small
        # ones. Skipping it keeps the uplink busy, but when the budget never has room for
        # it, its whole lane stands still (2026-09-14, the real fill: one lane sat for five
        # minutes). Once a lane has been skipped for `starvation_s`, the freed bytes are
        # held for it: nothing else starts until its job fits.
        starved = [l for l in open_lanes if l.blocked_since is not None
                   and now - l.blocked_since >= self.s.starvation_s]
        if starved:
            open_lanes = [min(starved, key=lambda l: (l.blocked_since, l.name))]
        for lane in open_lanes:
            job = lane.ready[0][2]
            if self.bytes_inflight and self.bytes_inflight + job.nbytes > self.bytes_budget:
                if lane.blocked_since is None:
                    lane.blocked_since = now
                continue
            lane.blocked_since = None
            heapq.heappop(lane.ready)
            lane.inflight += 1
            self.inflight += 1
            self.bytes_inflight += job.nbytes
            return job
        return None

    def _next_wake(self, now: float) -> float:
        wakes = [now + 1.0]
        for lane in self.lanes.values():
            if lane.delayed:
                wakes.append(lane.delayed[0][0])
            if lane.cooldown_until > now:
                wakes.append(lane.cooldown_until)
        if self.paused_until > now:
            wakes.append(self.paused_until)
        return max(0.01, min(wakes) - now)

    # ------------------------------------------------------------ public API
    def acquire(self) -> Optional[Job]:
        with self.cv:
            while True:
                if self.stop_reason is not None:
                    self.cv.notify_all()
                    return None
                now = self.clock()
                self._promote(now)
                if self._idle():
                    if self.deferred and self.round < self.s.rounds:
                        self.round += 1
                        for job in self.deferred:
                            job.attempts = 0
                            self._push_ready(self.lanes[job.lane], job)
                        self.deferred = []
                        continue
                    self.finished = True
                    self.cv.notify_all()
                    return None
                job = self._pick(now)
                if job is not None:
                    return job
                self.cv.wait(timeout=self._next_wake(now))

    def finish(self, job: Job, out: Outcome) -> None:
        with self.cv:
            lane = self.lanes[job.lane]
            now = self.clock()
            lane.inflight -= 1
            self.inflight -= 1
            self.bytes_inflight -= job.nbytes
            lane.kinds[out.kind] += 1
            job.last_kind = out.kind
            getattr(self, f"_on_{out.kind}", self._on_server)(lane, job, out, now)
            self.cv.notify_all()

    # ------------------------------------------------------------ outcomes
    def _on_ok(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        lane.ok += 1
        lane.streak += 1
        lane.server_streak = 0
        lane.last_progress = now
        lane.errors_since_progress = 0
        lane.ok_times.append(now)
        while lane.ok_times and now - lane.ok_times[0] > 1800:
            lane.ok_times.popleft()
        if out.cached:
            lane.cache_hits += 1
        else:
            lane.network_calls += 1
        if not out.parse_ok:
            lane.parse_failures += 1
        self.done_by[(job.lane, job.arm, job.condition)] += 1
        self.transport_streak = 0
        if self.probe_mode:
            self.probe_mode = False
            self.pause_s = self.s.network_pause_s
        self.global_streak += 1
        if lane.streak % self.s.increase_every == 0:
            lane.limit = min(float(lane.max_limit), lane.limit + 1.0)
        if self.global_streak % self.s.increase_every == 0:
            self.bytes_budget = min(self.s.max_bytes_budget_mb * 1e6,
                                    self.bytes_budget + self.s.bytes_step_mb * 1e6)

    def _error(self, lane: LaneState) -> None:
        lane.streak = 0
        lane.errors_since_progress += 1
        self.global_streak = 0

    def _shrink_lane(self, lane: LaneState, factor: float, now: float,
                     floor: float = 1.0) -> None:
        if now - lane.last_decrease >= self.s.decrease_gap_s:
            lane.limit = max(min(floor, lane.limit), lane.limit * factor)
            lane.last_decrease = now

    def _retry(self, lane: LaneState, job: Job, now: float) -> None:
        job.attempts += 1
        if job.attempts >= self.s.attempts_per_round:
            self.deferred.append(job)
        else:
            self._push_delayed(lane, job, now + self._backoff(job.attempts))

    def _on_transport(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        if "timed out" in out.message.lower() or "timeout" in out.message.lower():
            job.timeouts += 1
        if now - self.last_budget_decrease >= self.s.decrease_gap_s:
            self.bytes_budget = max(self.s.min_bytes_budget_mb * 1e6,
                                    self.bytes_budget * self.s.decrease_factor)
            self.last_budget_decrease = now
        # A transport error is the network, not this lane: never take a lane below
        # `min_lane_limit` for it (2026-09-14: bursts cut gemini to 3 slots). 429s still can.
        self._shrink_lane(lane, self.s.decrease_factor, now, floor=self.s.min_lane_limit)
        self.transport_streak += 1
        if self.transport_streak >= self.s.network_down_after:
            self.paused_until = now + self.pause_s
            self.pause_s = min(self.s.network_pause_cap_s, self.pause_s * 2)
            self.probe_mode = True
            self.transport_streak = 0
        self._retry(lane, job, now)

    def _on_server(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        lane.server_streak += 1
        if lane.server_streak >= self.s.server_streak:
            lane.cooldown_until = max(lane.cooldown_until, now + self.s.server_cooldown_s)
            lane.server_streak = 0
        self._retry(lane, job, now)

    def _on_rate_limit(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        self._shrink_lane(lane, 0.5, now)
        lane.cooldown_until = max(lane.cooldown_until, now + self.s.rate_limit_cooldown_s)
        self._push_ready(lane, job)

    def _on_channel(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        lane.cooldown_until = max(lane.cooldown_until, now + self.s.channel_cooldown_s)
        self._push_ready(lane, job)

    def _on_refusal(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        job.refusals += 1
        if job.refusals >= self.s.refusal_attempts:
            self.final.append((job, out))
        else:
            self._push_delayed(lane, job, now + self.s.refusal_delay_s)

    def _on_empty(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        job.empties += 1
        if job.empties >= self.s.empty_attempts:
            self.final.append((job, out))
        else:
            self._push_delayed(lane, job, now + self._backoff(job.empties))

    def _on_quota(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        self.quota_events.append(now)
        while self.quota_events and now - self.quota_events[0] > self.s.quota_window_s:
            self.quota_events.popleft()
        self._push_ready(lane, job)
        if len(self.quota_events) >= self.s.quota_errors_to_stop:
            self.stop_reason = self.stop_reason or "quota"
        else:
            lane.cooldown_until = max(lane.cooldown_until, now + self.s.quota_cooldown_s)

    def _on_auth(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        self._push_ready(lane, job)
        self.stop_reason = self.stop_reason or "auth"

    def _on_model_missing(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        self._push_ready(lane, job)
        lane.disabled = f"model missing: {out.message[:120]}"

    def _on_local(self, lane: LaneState, job: Job, out: Outcome, now: float) -> None:
        self._error(lane)
        self.final.append((job, out))
        self.local_errors += 1
        if self.local_errors >= self.s.local_errors_to_stop:
            self.stop_reason = self.stop_reason or "local_errors"

    # ------------------------------------------------------------ reporting
    def remaining(self) -> Dict[str, List[Job]]:
        with self.cv:
            out: Dict[str, List[Job]] = {}
            for lane in self.lanes.values():
                out[lane.name] = [j for _, _, j in lane.ready] + [j for _, _, j in lane.delayed]
            for job in self.deferred:
                out.setdefault(job.lane, []).append(job)
            return out

    def snapshot(self) -> dict:
        with self.cv:
            now = self.clock()
            lanes = {}
            est = 0.0
            for lane in self.lanes.values():
                pending = len(lane.ready) + len(lane.delayed) + lane.inflight + sum(
                    1 for j in self.deferred if j.lane == lane.name)
                window = min(1800.0, max(1.0, now - self.started))
                rate = len(lane.ok_times) * 3600.0 / window
                usd = lane.network_calls * lane.unit_price
                est += usd
                lanes[lane.name] = {
                    "done": lane.ok, "of": lane.total, "pending": pending,
                    "rate_per_h": round(rate, 1),
                    "eta_h": round(pending / rate, 2) if rate > 0 else None,
                    "limit": round(lane.limit, 2), "inflight": lane.inflight,
                    "cooldown_s": round(max(0.0, lane.cooldown_until - now), 1),
                    "disabled": lane.disabled, "network_calls": lane.network_calls,
                    "cache_hits": lane.cache_hits, "parse_failures": lane.parse_failures,
                    "errors": {k: v for k, v in sorted(lane.kinds.items()) if k != "ok"},
                    "est_usd": round(usd, 2),
                }
            conditions: Dict[str, Dict[str, str]] = defaultdict(dict)
            for (lname, arm, cond), total in sorted(self.totals.items()):
                conditions[f"{arm}/{cond}"][lname] = f"{self.done_by[(lname, arm, cond)]}/{total}"
            state = ("stopped:" + self.stop_reason) if self.stop_reason else (
                "finished" if self.finished else (
                    "paused" if now < self.paused_until else "running"))
            return {
                "state": state, "round": self.round,
                "elapsed_h": round((now - self.started) / 3600.0, 3),
                "inflight": self.inflight,
                "bytes_inflight_mb": round(self.bytes_inflight / 1e6, 2),
                "bytes_budget_mb": round(self.bytes_budget / 1e6, 2),
                "probe_mode": self.probe_mode, "deferred": len(self.deferred),
                "final_failures": len(self.final), "est_usd": round(est, 2),
                "lanes": lanes, "conditions": dict(conditions),
            }


# =========================================================================== #
# writing                                                                     #
# =========================================================================== #
class Writer:
    """Raw text first, then one fsync'd row. The only code that touches the manifests."""

    def __init__(self, plan: Plan):
        self.root = plan.root
        self.tree = plan.tree
        self.raw_dir = Path(f"{plan.tree_rel}/raw_responses")
        self._lock = threading.Lock()
        self._handles: Dict[Path, Any] = {}
        self._written: Set[str] = set()

    def write(self, job: Job, model: str, text: str) -> Tuple[Any, bool]:
        raw_rel = save_raw_response(self.root, self.raw_dir, model, job.result_id, text)
        if job.task == "scoring":
            row = rs.build_result(job.target, text, raw_rel, model=model, style=job.style,
                                  score_scale=int(job.score_scale))
        else:
            row = rp.build_result(job.target, text, raw_rel, model=model, style=job.style)
        line = io._to_line(row) + "\n"
        path = manifest_for(self.tree, job.task, model, biased=job.biased)
        with self._lock:
            if job.result_id in self._written:
                return row, False
            fh = self._handles.get(path)
            if fh is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = open(path, "a", encoding="utf-8", newline="\n")
                self._handles[path] = fh
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
            self._written.add(job.result_id)
        return row, True

    def close(self) -> None:
        with self._lock:
            for fh in self._handles.values():
                fh.close()
            self._handles.clear()


class FailureLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, job: Job, out: Outcome, round_: int, timeout: float) -> None:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "judge": job.lane,
               "arm": job.arm, "condition": job.condition, "result_id": job.result_id,
               "kind": out.kind, "status": out.status, "message": out.message[:300],
               "attempt": job.attempts, "refusals": job.refusals, "round": round_,
               "timeout_s": round(timeout, 1), "seconds": round(out.seconds, 2)}
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _keep_awake(on: bool) -> None:
    """Stop Windows from sleeping mid-run (a sleeping laptop looks like a dead relay)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if on else 0))
    except Exception:  # noqa: BLE001 - best effort
        pass


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# =========================================================================== #
# run                                                                         #
# =========================================================================== #
@dataclass
class RunReport:
    exit_code: int
    state: str
    written: int
    unresolved: int
    remaining: int
    snapshot: dict
    notes: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def run(plan_path, *, root: Optional[Path] = None, use_api: bool = False, mock: bool = False,
        faults: bool = False, judges: Optional[Sequence[str]] = None,
        arms: Optional[Sequence[str]] = None, per_condition: Optional[int] = None,
        tree: Optional[str] = None, max_usd: Optional[float] = None,
        caller_factory: Optional[Callable[[LanePlan], Caller]] = None,
        settings_overrides: Optional[dict] = None, allow_mismatch: bool = False,
        echo: Callable[[str], None] = print) -> RunReport:
    plan = load_plan(plan_path, root=root, tree=tree)
    for key, value in (settings_overrides or {}).items():
        if not hasattr(plan.settings, key):
            raise ValueError(f"unknown fill setting {key}")
        setattr(plan.settings, key, value)
    if max_usd is not None:
        plan.settings.max_estimated_usd = max_usd
    if not (use_api or mock or caller_factory):
        raise PermissionError("fill_runner run makes paid API calls; pass --use-api (or --mock)")
    if plan.tree.resolve() == (plan.root / "results/v2_fill").resolve() and mock:
        raise ValueError("--mock must not write into the real fill tree; pass --tree")

    lock = TreeLock(plan.tree)
    try:
        lock.acquire()
    except RuntimeError as exc:
        echo(f"REFUSED: {exc}")
        return RunReport(EXIT_REFUSED, "refused:lock", 0, 0, 0, {}, problems=[str(exc)])
    writer: Optional[Writer] = None
    try:
        notes, problems = repair_tree(plan.tree)
        for n in notes:
            echo(f"note: {n}")
        lanes, plan_problems = build_lanes(plan, judges=judges, arms=arms,
                                           per_condition=per_condition)
        hard = problems + [p for p in plan_problems
                           if not (allow_mismatch and "the plan expects" in p)]
        if hard:
            for p in hard:
                echo(f"PROBLEM: {p}")
            return RunReport(EXIT_REFUSED, "refused:plan", 0, 0, 0, {}, notes, hard)
        stop_file = plan.tree / STOP_FILE
        if stop_file.exists():
            stop_file.unlink()
            echo("note: removed a STOP file left by an earlier run")

        callers: Dict[str, Caller] = {}
        for lp in lanes:
            if caller_factory is not None:
                callers[lp.name] = caller_factory(lp)
            elif mock:
                inner = AdapterCaller(build_adapter(
                    {"type": "mock", "model_name": lp.name,
                     "score_scale": lp.judge_cfg.get("score_scale", DEFAULT_SCORE_SCALE)},
                    use_api=False, root=plan.root))
                seed = int(hashlib.sha256(lp.name.encode("utf-8")).hexdigest()[:8], 16)
                callers[lp.name] = FaultInjectingCaller(inner, seed=seed) if faults else inner
            else:
                callers[lp.name] = AdapterCaller(build_adapter(lp.judge_cfg, use_api=use_api,
                                                               root=plan.root))

        sched = Scheduler(lanes, plan.settings)
        writer = Writer(plan)
        failures = FailureLog(plan.failure_log)
        status_path = plan.tree / STATUS_FILE
        total = sum(len(lp.jobs) for lp in lanes)
        echo(f"fill: {total} jobs over {len(lanes)} judges -> {plan.tree_rel} "
             f"(round 1 of {plan.settings.rounds})")

        def worker() -> None:
            while True:
                job = sched.acquire()
                if job is None:
                    return
                timeout = sched.timeout_for(job)
                t0 = time.monotonic()
                try:
                    prompt, images = job.request(plan.root)
                    text, cached = callers[job.lane](JudgeRequest(prompt, images, job.task), timeout)
                    row, wrote = writer.write(job, job.lane, text)
                    out = Outcome("ok", cached=cached, parse_ok=bool(row.parse_success),
                                  seconds=time.monotonic() - t0, duplicate=not wrote)
                except CallError as e:
                    out = Outcome(e.kind, e.message, e.status, seconds=time.monotonic() - t0)
                except Exception as e:  # noqa: BLE001 - a bug or a bad file, never a retry
                    out = Outcome("local", f"{type(e).__name__}: {e}"[:300],
                                  seconds=time.monotonic() - t0)
                if out.kind != "ok":
                    failures.write(job, out, sched.round, timeout)
                sched.finish(job, out)

        reporter_stop = threading.Event()

        def reporter() -> None:
            last_progress = 0.0
            while True:
                snap = sched.snapshot()
                snap["updated"] = datetime.now().isoformat(timespec="seconds")
                snap["pid"] = os.getpid()
                try:
                    _atomic_json(status_path, snap)
                except OSError:
                    pass
                if stop_file.exists():
                    sched.stop("user")
                cap = plan.settings.max_estimated_usd
                if cap is not None and snap["est_usd"] > cap:
                    sched.stop("tripwire")
                now = time.monotonic()
                if now - last_progress >= plan.settings.progress_every_s or reporter_stop.is_set():
                    last_progress = now
                    line = (f"{snap['updated']} {snap['state']} round={snap['round']} "
                            f"inflight={snap['inflight']} budget={snap['bytes_budget_mb']}MB "
                            f"est=${snap['est_usd']} | " + " | ".join(
                                f"{n} {v['done']}/{v['of']} {v['rate_per_h']}/h lim={v['limit']}"
                                + (f" err={sum(v['errors'].values())}" if v['errors'] else "")
                                + (f" DISABLED({v['disabled']})" if v['disabled'] else "")
                                for n, v in snap["lanes"].items()))
                    try:
                        plan.progress_log.parent.mkdir(parents=True, exist_ok=True)
                        with plan.progress_log.open("a", encoding="utf-8") as fh:
                            fh.write(line + "\n")
                    except OSError:
                        pass
                    echo(line)
                if reporter_stop.wait(plan.settings.status_every_s):
                    return

        _keep_awake(True)
        n_workers = max(1, min(plan.settings.max_inflight,
                               sum(lp.spec.max_concurrency for lp in lanes)))
        threads = [threading.Thread(target=worker, name=f"fill-{i}", daemon=True)
                   for i in range(n_workers)]
        rep = threading.Thread(target=reporter, name="fill-status", daemon=True)
        rep.start()
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            echo("interrupt: finishing the calls in flight, then stopping")
            sched.stop("interrupt")
            for t in threads:
                t.join()
        reporter_stop.set()
        rep.join()

        snap = sched.snapshot()
        remaining = sched.remaining()
        n_remaining = sum(len(v) for v in remaining.values())
        unresolved_rows = [
            {"judge": j.lane, "arm": j.arm, "condition": j.condition, "result_id": j.result_id,
             "kind": o.kind, "message": o.message[:300]} for j, o in sched.final]
        if sched.stop_reason is None:
            for lane_name, jobs in remaining.items():
                reason = sched.lanes[lane_name].disabled or "attempts exhausted"
                unresolved_rows += [{"judge": j.lane, "arm": j.arm, "condition": j.condition,
                                     "result_id": j.result_id, "kind": j.last_kind or "pending",
                                     "message": reason} for j in jobs]
        with (plan.tree / UNRESOLVED_FILE).open("w", encoding="utf-8") as fh:
            for row in unresolved_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written = sum(l["done"] for l in snap["lanes"].values())
        snap["updated"] = datetime.now().isoformat(timespec="seconds")
        snap["unresolved"] = len(unresolved_rows)
        _atomic_json(status_path, snap)
        if sched.stop_reason in ("quota", "auth", "tripwire", "local_errors"):
            code = EXIT_SAFETY_STOP
        elif sched.stop_reason in ("user", "interrupt"):
            code = EXIT_USER_STOP
        elif unresolved_rows:
            code = EXIT_UNRESOLVED
        else:
            code = EXIT_COMPLETE
        echo(f"fill: {snap['state']} written={written} unresolved={len(unresolved_rows)} "
             f"not_attempted={n_remaining if sched.stop_reason else 0} est=${snap['est_usd']}")
        return RunReport(code, snap["state"], written, len(unresolved_rows),
                         n_remaining if sched.stop_reason else 0, snap, notes, [])
    finally:
        _keep_awake(False)
        if writer is not None:
            writer.close()
        lock.release()


# =========================================================================== #
# plan / status / stop / verify                                               #
# =========================================================================== #
def cmd_plan(args) -> int:
    plan = load_plan(args.plan, tree=args.tree)
    lanes, problems = build_lanes(plan, judges=args.judges, arms=args.arms,
                                  per_condition=args.per_condition)
    grand_calls = 0
    grand_usd = 0.0
    for lp in lanes:
        usd = len(lp.jobs) * lp.spec.unit_price_usd
        grand_calls += len(lp.jobs)
        grand_usd += usd
        mb = sum(j.nbytes for j in lp.jobs) / max(1, len(lp.jobs)) / 1e6
        print(f"{lp.name:24} selected={len(lp.jobs):6d}  ~${usd:8.2f}  mean payload {mb:.2f} MB "
              f"timeout {lp.base_timeout:.0f}s")
        for arm, info in lp.per_arm.items():
            print(f"    {arm:14} targets={info['targets']} published={info['in_published_tree']} "
                  f"new={info['new']} in_fill={info['in_fill_tree']} pending={info['pending']}")
            print("        " + ", ".join(f"{c}={n}" for c, n in info["by_condition"].items()))
    print(f"TOTAL selected calls={grand_calls}  tripwire-rate cost ~${grand_usd:.2f}")
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1 if problems else 0


def cmd_status(args) -> int:
    plan = load_plan(args.plan, tree=args.tree)
    path = plan.tree / STATUS_FILE
    if not path.exists():
        print(f"no status yet at {path}")
        return 1
    snap = json.loads(path.read_text(encoding="utf-8"))
    print(f"{snap.get('updated')} state={snap['state']} round={snap['round']} "
          f"elapsed={snap['elapsed_h']}h inflight={snap['inflight']} "
          f"budget={snap['bytes_budget_mb']}MB est=${snap['est_usd']} "
          f"deferred={snap['deferred']} final_failures={snap['final_failures']}")
    for name, v in snap["lanes"].items():
        print(f"  {name:24} {v['done']:6d}/{v['of']:<6d} {v['rate_per_h']:7.1f}/h "
              f"eta={v['eta_h']}h lim={v['limit']} infl={v['inflight']} "
              f"calls={v['network_calls']} cache={v['cache_hits']} parse_fail={v['parse_failures']} "
              f"${v['est_usd']} err={v['errors']}" + (f" DISABLED={v['disabled']}" if v['disabled'] else ""))
    if args.conditions:
        for cond, per in snap["conditions"].items():
            print(f"  {cond:36} " + "  ".join(f"{k.split('-')[0]}:{v}" for k, v in per.items()))
    return 0


def cmd_stop(args) -> int:
    plan = load_plan(args.plan, tree=args.tree)
    plan.tree.mkdir(parents=True, exist_ok=True)
    (plan.tree / STOP_FILE).write_text(datetime.now().isoformat(timespec="seconds") + "\n",
                                       encoding="utf-8")
    print(f"STOP requested: {plan.tree / STOP_FILE} (in-flight calls finish, then it exits)")
    return 0


def verify_tree(plan: Plan) -> Tuple[dict, List[str]]:
    problems: List[str] = []
    per_model: Dict[str, Counter] = defaultdict(Counter)
    parse_fail: Counter = Counter()
    referenced: Set[str] = set()
    ids_by_task_model: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for path in _manifests(plan.tree):
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            problems.append(f"{path.name}: torn last line")
        task, model = path.stem.split("__", 1)
        seen: Counter = Counter()
        for n, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{path.name}:{n}: not JSON")
                continue
            seen[row["result_id"]] += 1
            ids_by_task_model[(task, model)].add(row["result_id"])
            per_model[f"{task}__{model}"][row.get("bias_type") or "baseline"] += 1
            parse_fail[f"{task}__{model}"] += not row.get("parse_success", False)
            raw = row.get("raw_response_path")
            referenced.add(raw)
            if not raw or not (plan.root / raw).is_file():
                problems.append(f"{path.name}:{n}: raw response missing ({raw})")
        dups = [k for k, c in seen.items() if c > 1]
        if dups:
            problems.append(f"{path.name}: {len(dups)} duplicated result_ids")
    for (task, model), ids in ids_by_task_model.items():
        overlap = ids & load_done(plan.done_trees, task, model)
        if overlap:
            problems.append(f"{task}__{model}: {len(overlap)} rows also exist in a done tree")
    raw_root = plan.tree / "raw_responses"
    on_disk = {p.relative_to(plan.root).as_posix() for p in raw_root.rglob("*.txt")} if raw_root.exists() else set()
    orphans = sorted(on_disk - referenced)
    summary = {
        "rows": {k: dict(sorted(v.items())) for k, v in sorted(per_model.items())},
        "parse_failures": dict(parse_fail),
        "raw_files": len(on_disk),
        "raw_without_row": len(orphans),
        "raw_without_row_examples": orphans[:5],
    }
    return summary, problems


def cmd_verify(args) -> int:
    plan = load_plan(args.plan, tree=args.tree)
    summary, problems = verify_tree(plan)
    for key, counts in summary["rows"].items():
        n = sum(counts.values())
        print(f"{key:40} rows={n:6d} parse_fail={summary['parse_failures'].get(key, 0)}  "
              + ", ".join(f"{c}={k}" for c, k in counts.items()))
    print(f"raw files={summary['raw_files']} without a row={summary['raw_without_row']} "
          "(a kill between raw write and row append leaves these; resume re-asks them free)")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("verify " + ("FAILED" if problems else "OK"))
    return 1 if problems else 0


def _overrides(pairs: Optional[Sequence[str]]) -> dict:
    """`--set key=value ...` -> {key: yaml-parsed value}; unknown keys fail in `run`."""
    out = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def cmd_run(args) -> int:
    report = run(args.plan, use_api=args.use_api, mock=args.mock, faults=args.faults,
                 judges=args.judges, arms=args.arms, per_condition=args.per_condition,
                 tree=args.tree, max_usd=args.max_usd, allow_mismatch=args.allow_mismatch,
                 settings_overrides=_overrides(args.set))
    return report.exit_code


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", default="configs/experiment/fill_v2.yaml")
    common.add_argument("--tree", default=None,
                        help="write somewhere other than the plan's tree (rehearsals)")
    select = argparse.ArgumentParser(add_help=False)
    select.add_argument("--judges", nargs="+", default=None)
    select.add_argument("--arms", nargs="+", default=None)
    select.add_argument("--per-condition", type=int, default=None,
                        help="calibration: at most N evenly spread jobs per judge x arm x condition")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", parents=[common, select]).set_defaults(fn=cmd_plan)
    r = sub.add_parser("run", parents=[common, select])
    r.add_argument("--use-api", action="store_true")
    r.add_argument("--mock", action="store_true", help="mock judges (needs --tree)")
    r.add_argument("--faults", action="store_true", help="with --mock: inject failures")
    r.add_argument("--max-usd", type=float, default=None)
    r.add_argument("--allow-mismatch", action="store_true",
                   help="run even if a lane's new-target count differs from the plan")
    r.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                   help="override a transport/retry/tripwire setting for this run only")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", parents=[common])
    s.add_argument("--conditions", action="store_true")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("stop", parents=[common]).set_defaults(fn=cmd_stop)
    sub.add_parser("verify", parents=[common]).set_defaults(fn=cmd_verify)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
