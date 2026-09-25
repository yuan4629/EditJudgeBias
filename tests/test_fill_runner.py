"""Tests for the fill orchestrator (`experiments/fill_runner.py`).

The orchestrator is only worth having if (1) it asks exactly what the published runners
ask, (2) every recovery path ends with each target written once, and (3) it stops when
continuing would waste money. Each test below pins one of those.
"""

from __future__ import annotations

import io as _io
import json
import socket
import threading
import time
import urllib.error
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from conftest import make_rgb_image

from edit_judge_bias.data import io
from edit_judge_bias.data.schema import BiasedRecord, PairRecord, SampleRecord
from edit_judge_bias.experiments import fill_runner as fr
from edit_judge_bias.experiments.run_pairwise_judge import run as run_pairwise
from edit_judge_bias.experiments.run_scoring_judge import run as run_scoring
from edit_judge_bias.judges import build_adapter
from edit_judge_bias.judges.base import JudgeRequest

FAST = dict(backoff_base_s=0.001, backoff_cap_s=0.002, refusal_delay_s=0.001,
            rate_limit_cooldown_s=0.001, channel_cooldown_s=0.001, server_cooldown_s=0.001,
            network_pause_s=0.001, network_pause_cap_s=0.002, quota_cooldown_s=0.001,
            status_every_s=0.02, progress_every_s=0.02, decrease_gap_s=0.0)
CUES = ["brightness", "sham"]
N_SAMPLES = 4
N_PAIRS = 3


def _quiet(_msg: str) -> None:
    pass


def _project(root: Path, *, expect_pair=None, expect_score=None) -> Path:
    """A miniature repo: 4 samples, 3 pairs over them, two cues, two mock judges."""
    samples, biased = [], []
    for i in range(N_SAMPLES):
        make_rgb_image(root / f"orig/{i}.png")
        make_rgb_image(root / f"edit/{i}.png", color=(10 * i, 90, 40))
        samples.append(SampleRecord(
            sample_id=f"s{i}", source_dataset="I2EBench", edit_type="remove",
            content_category="object", original_image_path=f"orig/{i}.png",
            instruction=f"Remove object {i}.", edit_model=f"m{i % 2}",
            edited_image_path=f"edit/{i}.png",
        ))
        for cue in CUES:
            make_rgb_image(root / f"biased/{cue}/{i}.png", color=(200, 10 * i, 5))
            biased.append(BiasedRecord(
                biased_id=f"s{i}__{cue}", base_sample_id=f"s{i}", bias_type=cue,
                bias_strength=1.0, biased_image_path=f"biased/{cue}/{i}.png",
            ))
    io.write_jsonl(root / "samples.jsonl", samples)
    io.write_jsonl(root / "biased.jsonl", biased)
    pairs = [PairRecord(
        pair_id=f"p{i}", sample_id_a=f"s{i}", sample_id_b=f"s{i + 1}",
        original_image_path=f"orig/{i}.png", instruction=f"Remove object {i}.",
        edited_image_a_path=f"edit/{i}.png", edited_image_b_path=f"edit/{i + 1}.png",
        edit_model_a=f"m{i % 2}", edit_model_b=f"m{(i + 1) % 2}",
        source_dataset="I2EBench", edit_type="remove",
    ) for i in range(N_PAIRS)]
    io.write_jsonl(root / "pairs.jsonl", pairs)

    (root / "judges").mkdir(exist_ok=True)
    (root / "judges/j1.yaml").write_text(yaml.safe_dump(
        {"type": "mock", "model_name": "j1", "timeout": 30}), encoding="utf-8")
    (root / "judges/j2.yaml").write_text(yaml.safe_dump(
        {"type": "mock", "model_name": "j2", "scoring_prompt_style": "viescore"}), encoding="utf-8")

    prompt_pair = [{"bias_type": "bandwagon", "bandwagon_target": "seeded"},
                   {"bias_type": "model_name", "model_name_source": "true_name"}]
    prompt_score = [{"bias_type": "bandwagon", "bandwagon": True},
                    {"bias_type": "model_name", "model_name_source": "true_name"}]

    def arm(name, tree, **extra):
        cfg = {"prompt_style": "vanilla", "judge_config": "judges/j1.yaml", "seed": 42,
               "raw_dir": f"{tree}/raw_responses", "results_dir": tree, **extra}
        (root / name).write_text(yaml.safe_dump(cfg), encoding="utf-8")

    pair_extra = dict(pairs="pairs.jsonl", include_position_swap=False,
                      one_sided_biases={"biased": "biased.jsonl", "bias_types": CUES},
                      prompt_biases=prompt_pair)
    score_extra = dict(samples="samples.jsonl", biased="biased.jsonl", bias_types=CUES,
                       score_original=False, score_biased=True, score_scale=10,
                       prompt_biases=prompt_score)
    arm("pair_fill.yaml", "fill", **pair_extra)
    arm("score_fill.yaml", "fill", **score_extra)
    arm("pair_ref.yaml", "ref", **pair_extra)
    arm("score_ref.yaml", "ref", **score_extra)
    # the "published" tree already holds the pairwise baselines and the sham scoring cells
    arm("pair_pub.yaml", "pub", pairs="pairs.jsonl", include_position_swap=False)
    arm("score_pub.yaml", "pub", samples="samples.jsonl", biased="biased.jsonl",
        bias_types=["sham"], score_original=False, score_scale=10)
    for j in ("judges/j1.yaml", "judges/j2.yaml"):
        run_pairwise(root / "pair_pub.yaml", root=root, judge_config=j)
        run_scoring(root / "score_pub.yaml", root=root, judge_config=j)

    plan = {
        "tree": "fill", "done_trees": ["pub"],
        "failure_log": "logs/fill_failures.jsonl", "progress_log": "logs/fill_progress.log",
        "arms": [
            {"name": "pairwise_fill", "task": "pairwise", "config": "pair_fill.yaml",
             "expect_new_per_judge": N_PAIRS * 4 if expect_pair is None else expect_pair},
            {"name": "anchor_fill", "task": "scoring", "config": "score_fill.yaml",
             "expect_new_per_judge": N_SAMPLES * 3 if expect_score is None else expect_score},
        ],
        "priority": ["sham", "bandwagon", "brightness", "model_name"],
        "judges": [
            {"config": "judges/j1.yaml", "concurrency": 2, "max_concurrency": 3,
             "unit_price_usd": 0.01},
            {"config": "judges/j2.yaml", "concurrency": 2, "max_concurrency": 3,
             "unit_price_usd": 0.002},
        ],
    }
    (root / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    return root / "plan.yaml"


EXPECTED_PER_JUDGE = N_PAIRS * 4 + N_SAMPLES * 3


def _rows(root: Path, tree: str):
    out = {}
    for sub in ("raw_judgments", "biased_judgments"):
        for p in (root / tree / sub).glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    assert r["result_id"] not in out, f"duplicate {r['result_id']}"
                    out[r["result_id"]] = r
    return out


def _mock_factory(wrap=None):
    def factory(lp):
        inner = fr.AdapterCaller(build_adapter(
            {"type": "mock", "model_name": lp.name, "score_scale": 10}, use_api=False))
        return wrap(lp, inner) if wrap else inner
    return factory


def _run(plan, root, **kw):
    kw.setdefault("settings_overrides", FAST)
    kw.setdefault("echo", _quiet)
    return fr.run(plan, root=root, **kw)


# --------------------------------------------------------------------------- #
# 1. it asks what the published runners ask                                    #
# --------------------------------------------------------------------------- #
def test_fill_rows_are_the_rows_the_published_runners_write(tmp_path: Path):
    plan = _project(tmp_path)
    report = _run(plan, tmp_path, mock=True)
    assert report.exit_code == fr.EXIT_COMPLETE, report
    for j in ("judges/j1.yaml", "judges/j2.yaml"):
        run_pairwise(tmp_path / "pair_ref.yaml", root=tmp_path, judge_config=j)
        run_scoring(tmp_path / "score_ref.yaml", root=tmp_path, judge_config=j)
    fill, ref = _rows(tmp_path, "fill"), _rows(tmp_path, "ref")
    assert len(fill) == 2 * EXPECTED_PER_JUDGE
    for rid, row in fill.items():
        want = dict(ref[rid])
        assert want["raw_response_path"].startswith("ref/raw_responses/")
        got = dict(row)
        assert got["raw_response_path"] == "fill/" + want["raw_response_path"][len("ref/"):]
        assert (tmp_path / got["raw_response_path"]).read_text(encoding="utf-8") == \
            (tmp_path / want["raw_response_path"]).read_text(encoding="utf-8")
        got.pop("raw_response_path"), want.pop("raw_response_path")
        assert got == want, rid
    # the judge-declared style survives: j2 scores with the VIEScore rubric
    assert any(r["prompt_type"] == "viescore_scoring" for r in fill.values() if r["judge_model"] == "j2")
    # nothing that was already published was asked again
    pub = _rows(tmp_path, "pub")
    assert not set(fill) & set(pub)


def test_count_mismatch_against_the_plan_refuses_to_start(tmp_path: Path):
    plan = _project(tmp_path, expect_score=N_SAMPLES * 4)  # forgets the published sham cells
    report = _run(plan, tmp_path, mock=True)
    assert report.exit_code == fr.EXIT_REFUSED
    assert any("the plan expects" in p for p in report.problems)
    assert not _rows(tmp_path, "fill")


def test_published_tree_is_refused(tmp_path: Path):
    plan = _project(tmp_path)
    for tree in ("results/v2", "results"):
        with pytest.raises(ValueError, match="published tree"):
            fr.load_plan(plan, root=tmp_path, tree=tree)


# --------------------------------------------------------------------------- #
# 2. every recovery path ends with each target written exactly once            #
# --------------------------------------------------------------------------- #
def test_transient_failures_are_retried_until_every_target_lands_once(tmp_path: Path):
    plan = _project(tmp_path)
    attempts = defaultdict(int)
    lock = threading.Lock()
    script = ["transport", "server", "rate_limit", "channel", "empty"]

    def wrap(lp, inner):
        def call(request, timeout):
            key = (lp.name, request.prompt, tuple(map(str, request.images)))
            with lock:
                n = attempts[key]
                attempts[key] += 1
            if n < 2:
                kind = script[hash(key) % len(script)] if n == 0 else "transport"
                raise fr.CallError(kind, f"injected {kind}")
            return inner(request, timeout)
        return call

    report = _run(plan, tmp_path, caller_factory=_mock_factory(wrap))
    assert report.exit_code == fr.EXIT_COMPLETE, report.snapshot
    assert len(_rows(tmp_path, "fill")) == 2 * EXPECTED_PER_JUDGE
    fails = (tmp_path / "logs/fill_failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(fails) == 2 * 2 * EXPECTED_PER_JUDGE
    summary, problems = fr.verify_tree(fr.load_plan(plan, root=tmp_path))
    assert problems == [] and summary["raw_without_row"] == 0


def test_permanent_refusal_is_recorded_and_never_written(tmp_path: Path):
    plan = _project(tmp_path)
    calls = defaultdict(int)
    victim = {}
    lock = threading.Lock()

    def wrap(lp, inner):
        def call(request, timeout):
            key = (lp.name, request.prompt, tuple(map(str, request.images)))
            with lock:
                if lp.name == "j1" and "key" not in victim:
                    victim["key"] = key          # the first question j1 is asked
                hit = victim.get("key") == key
                if hit:
                    calls[key] += 1
            if hit:
                raise fr.CallError("refusal", "HTTP 400: PROHIBITED_CONTENT", 400)
            return inner(request, timeout)
        return call

    report = _run(plan, tmp_path, caller_factory=_mock_factory(wrap))
    assert report.exit_code == fr.EXIT_UNRESOLVED
    assert list(calls.values()) == [fr.Settings().refusal_attempts]
    unresolved = [json.loads(l) for l in (tmp_path / "fill" / fr.UNRESOLVED_FILE)
                  .read_text(encoding="utf-8").splitlines()]
    assert len(unresolved) == 1 and unresolved[0]["kind"] == "refusal"
    assert unresolved[0]["result_id"] not in _rows(tmp_path, "fill")
    assert len(_rows(tmp_path, "fill")) == 2 * EXPECTED_PER_JUDGE - 1


def test_empty_wallet_stops_everything_and_a_rerun_finishes_the_rest(tmp_path: Path):
    plan = _project(tmp_path)
    served = {"n": 0}
    lock = threading.Lock()

    def wrap(lp, inner):
        def call(request, timeout):
            with lock:
                served["n"] += 1
                n = served["n"]
            if n > 6:
                raise fr.CallError("quota", "HTTP 403: 用户额度不足", 403)
            return inner(request, timeout)
        return call

    report = _run(plan, tmp_path, caller_factory=_mock_factory(wrap))
    assert report.exit_code == fr.EXIT_SAFETY_STOP and report.state == "stopped:quota"
    first = len(_rows(tmp_path, "fill"))
    assert 0 < first < 2 * EXPECTED_PER_JUDGE
    # the quota errors must not have burned the jobs' attempts: nothing is unresolved
    assert (tmp_path / "fill" / fr.UNRESOLVED_FILE).read_text(encoding="utf-8") == ""

    report = _run(plan, tmp_path, caller_factory=_mock_factory())
    assert report.exit_code == fr.EXIT_COMPLETE
    assert len(_rows(tmp_path, "fill")) == 2 * EXPECTED_PER_JUDGE
    assert report.written == 2 * EXPECTED_PER_JUDGE - first


def test_torn_last_line_is_cut_kept_and_its_job_asked_again(tmp_path: Path):
    plan = _project(tmp_path)
    assert _run(plan, tmp_path, mock=True).exit_code == fr.EXIT_COMPLETE
    man = tmp_path / "fill/biased_judgments/pairwise__j1.jsonl"
    lines = man.read_text(encoding="utf-8").splitlines(keepends=True)
    lost = json.loads(lines[-1])["result_id"]
    man.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 2], encoding="utf-8")

    notes = []
    report = fr.run(plan, root=tmp_path, mock=True, settings_overrides=FAST, echo=notes.append)
    assert report.exit_code == fr.EXIT_COMPLETE and report.written == 1
    assert any("torn last line" in n for n in notes)
    assert list((tmp_path / "fill/biased_judgments").glob("pairwise__j1.jsonl.torn-*"))
    assert lost in _rows(tmp_path, "fill")
    assert fr.verify_tree(fr.load_plan(plan, root=tmp_path))[1] == []


def test_a_bad_line_that_is_not_a_torn_tail_refuses_to_start(tmp_path: Path):
    plan = _project(tmp_path)
    assert _run(plan, tmp_path, mock=True).exit_code == fr.EXIT_COMPLETE
    man = tmp_path / "fill/biased_judgments/pairwise__j1.jsonl"
    lines = man.read_text(encoding="utf-8").splitlines(keepends=True)
    man.write_text(lines[0] + lines[0] + "".join(lines[1:]), encoding="utf-8")
    report = _run(plan, tmp_path, mock=True)
    assert report.exit_code == fr.EXIT_REFUSED
    assert any("duplicated result_ids" in p for p in report.problems)


def test_second_instance_on_the_same_tree_is_refused(tmp_path: Path):
    plan = _project(tmp_path)
    held = fr.TreeLock(tmp_path / "fill").acquire()
    try:
        report = _run(plan, tmp_path, mock=True)
        assert report.exit_code == fr.EXIT_REFUSED and report.state == "refused:lock"
    finally:
        held.release()
    assert _run(plan, tmp_path, mock=True).exit_code == fr.EXIT_COMPLETE


def test_stop_file_lets_in_flight_calls_land_then_exits(tmp_path: Path):
    plan = _project(tmp_path)

    def wrap(lp, inner):
        def call(request, timeout):
            time.sleep(0.15)
            return inner(request, timeout)
        return call

    timer = threading.Timer(0.4, lambda: (tmp_path / "fill" / fr.STOP_FILE).write_text("x"))
    timer.start()
    report = _run(plan, tmp_path, caller_factory=_mock_factory(wrap))
    timer.join()
    assert report.exit_code == fr.EXIT_USER_STOP
    partial = len(_rows(tmp_path, "fill"))
    assert 0 < partial < 2 * EXPECTED_PER_JUDGE
    assert fr.verify_tree(fr.load_plan(plan, root=tmp_path))[1] == []
    report = _run(plan, tmp_path, caller_factory=_mock_factory())
    assert report.exit_code == fr.EXIT_COMPLETE
    assert len(_rows(tmp_path, "fill")) == 2 * EXPECTED_PER_JUDGE


def test_per_condition_calibration_takes_n_per_judge_arm_condition(tmp_path: Path):
    plan = _project(tmp_path)
    report = _run(plan, tmp_path, mock=True, per_condition=1)
    assert report.exit_code == fr.EXIT_COMPLETE
    rows = _rows(tmp_path, "fill")
    per = defaultdict(int)
    for r in rows.values():
        per[(r["judge_model"], r["task_type"], r["bias_type"])] += 1
    # pairwise: brightness, sham, bandwagon, model_name; scoring: brightness, bandwagon, model_name
    assert len(per) == 2 * (4 + 3) and set(per.values()) == {1}


# --------------------------------------------------------------------------- #
# 3. the transport and the scheduler                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status,body,kind", [
    (400, '{"error":{"message":"PROHIBITED_CONTENT"}}', "refusal"),
    (400, "Your request was rejected: content_policy_violation", "refusal"),
    (400, '{"code":"DataInspectionFailed"}', "refusal"),
    (403, "用户额度不足, 剩余额度: $0.01", "quota"),
    (429, "You exceeded your current quota", "quota"),
    (401, "invalid_api_key", "auth"),
    (503, "当前分组 default 下对于模型 x 无可用渠道", "channel"),
    (429, "Too Many Requests", "rate_limit"),
    (404, "model_not_found", "model_missing"),
    (500, "get_token_error", "server"),
    (400, "bad upstream", "server"),
])
def test_http_failures_are_classified(status, body, kind):
    assert fr.classify_http(status, body) == kind


def _openai_adapter(tmp_path: Path):
    from edit_judge_bias.judges.openai_judge import OpenAICompatibleJudge

    return OpenAICompatibleJudge("judge-x", "https://relay.invalid/v1", "sk-test",
                                 cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl")


def test_adapter_caller_serves_the_cache_and_classifies_errors(tmp_path: Path, monkeypatch):
    img = make_rgb_image(tmp_path / "a.png")
    request = JudgeRequest("prompt", [img, img], "scoring")
    caller = fr.AdapterCaller(_openai_adapter(tmp_path))
    posts = []

    def ok(endpoint, key, body, timeout):
        posts.append(timeout)
        return json.dumps({"choices": [{"message": {"content": '{"overall_score": 7}'}}],
                           "usage": {"prompt_tokens": 1}})

    monkeypatch.setattr(fr, "_post", ok)
    assert caller(request, 42.0) == ('{"overall_score": 7}', False)
    assert caller(request, 99.0) == ('{"overall_score": 7}', True)
    assert posts == [42.0]  # the timeout is per call, and a cache hit makes no call

    other = JudgeRequest("another prompt", [img], "scoring")

    def refused(endpoint, key, body, timeout):
        raise urllib.error.HTTPError(endpoint, 400, "Bad Request", {},
                                     _io.BytesIO(b'{"error":{"message":"PROHIBITED_CONTENT"}}'))

    monkeypatch.setattr(fr, "_post", refused)
    with pytest.raises(fr.CallError) as e:
        caller(other, 10.0)
    assert e.value.kind == "refusal" and e.value.status == 400

    def slow(endpoint, key, body, timeout):
        raise socket.timeout("The read operation timed out")

    monkeypatch.setattr(fr, "_post", slow)
    with pytest.raises(fr.CallError) as e:
        caller(other, 10.0)
    assert e.value.kind == "transport"

    monkeypatch.setattr(fr, "_post", lambda *a: json.dumps({"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}))
    with pytest.raises(fr.CallError) as e:
        caller(other, 10.0)
    assert e.value.kind == "empty" and "content_filter" in e.value.message


def _lane(name, n, *, conc=4, nbytes=1_000_000):
    spec = fr.LaneSpec(config="x", concurrency=conc, max_concurrency=conc, unit_price_usd=0.01)
    jobs = [fr.Job(lane=name, arm="a", task="pairwise", condition="c", rank=0, order=i,
                   result_id=f"{name}-{i}", target=None, style="vanilla", score_scale=None,
                   biased=True, nbytes=nbytes) for i in range(n)]
    return fr.LanePlan(name=name, spec=spec, judge_cfg={}, base_timeout=60.0, jobs=jobs, per_arm={})


def test_byte_budget_caps_what_is_in_flight():
    now = [0.0]
    s = fr.Settings(bytes_budget_mb=2.5, max_inflight=10)
    sched = fr.Scheduler([_lane("x", 5)], s, clock=lambda: now[0])
    with sched.cv:
        picked = [sched._pick(now[0]) for _ in range(4)]
    assert [p is not None for p in picked] == [True, True, False, False]


def test_a_big_job_is_not_starved_by_a_stream_of_small_ones():
    # 2026-09-14, the real fill: the budget stayed full of small calls from other lanes, so
    # a lane whose next job was big was skipped on every pick and stood still for minutes.
    now = [0.0]
    s = fr.Settings(bytes_budget_mb=3.0, max_inflight=50, starvation_s=20.0)
    sched = fr.Scheduler([_lane("a", 40, conc=40, nbytes=500_000),
                          _lane("b", 1, conc=4, nbytes=3_000_000)], s, clock=lambda: now[0])
    with sched.cv:
        picked = [sched._pick(now[0]) for _ in range(7)]
    assert [j.lane if j else None for j in picked] == ["a"] * 6 + [None]
    assert sched.lanes["b"].blocked_since == 0.0
    running = picked[:6]

    now[0] = 10.0                                   # not starved yet: small calls still flow
    sched.finish(running.pop(0), fr.Outcome("ok"))
    with sched.cv:
        nxt = sched._pick(now[0])
    assert nxt is not None and nxt.lane == "a"
    running.append(nxt)

    now[0] = 25.0                                   # starved: the freed bytes are held for b
    sched.finish(running.pop(0), fr.Outcome("ok"))
    with sched.cv:
        assert sched._pick(now[0]) is None
    for job in running:
        sched.finish(job, fr.Outcome("ok"))
    with sched.cv:
        big = sched._pick(now[0])
    assert big is not None and big.lane == "b"
    assert sched.lanes["b"].blocked_since is None


def test_a_transport_burst_leaves_the_budget_and_lanes_at_their_floors():
    # 2026-09-14, the real fill: SSL-EOF bursts arrived whatever the load, and each one cut
    # the budget from 96 to 7 MB and a lane to 3 slots, then took ten minutes to climb back.
    now = [0.0]
    s = fr.Settings(bytes_budget_mb=24, min_bytes_budget_mb=20, decrease_factor=0.85,
                    decrease_gap_s=0.0, min_lane_limit=6, network_down_after=1000)
    sched = fr.Scheduler([_lane("x", 40, conc=12, nbytes=100_000)], s, clock=lambda: now[0])
    with sched.cv:
        jobs = [sched._pick(now[0]) for _ in range(12)]
    for job in jobs:
        sched.finish(job, fr.Outcome("transport", "EOF occurred in violation of protocol"))
    assert sched.bytes_budget == pytest.approx(20e6)
    assert sched.lanes["x"].limit == pytest.approx(6.0)
    with sched.cv:
        job = sched._pick(now[0])
    sched.finish(job, fr.Outcome("rate_limit", "HTTP 429"))
    assert sched.lanes["x"].limit == pytest.approx(3.0)      # a real 429 still halves it


def test_transport_storm_shrinks_the_budget_then_pauses_and_probes_with_one_call():
    now = [100.0]
    s = fr.Settings(bytes_budget_mb=10, min_bytes_budget_mb=3, network_down_after=3,
                    network_pause_s=30, decrease_gap_s=0.0)
    sched = fr.Scheduler([_lane("x", 6, conc=6)], s, clock=lambda: now[0])
    with sched.cv:
        jobs = [sched._pick(now[0]) for _ in range(3)]
    for job in jobs:
        sched.finish(job, fr.Outcome("transport", "The read operation timed out"))
    assert sched.bytes_budget == pytest.approx(10e6 * 0.7 ** 3)
    assert sched.probe_mode and sched.paused_until == pytest.approx(130.0)
    with sched.cv:
        assert sched._pick(now[0]) is None           # paused
    now[0] = 131.0
    with sched.cv:
        probe = sched._pick(now[0])
        assert probe is not None and sched._pick(now[0]) is None   # one call at a time
    sched.finish(probe, fr.Outcome("ok"))
    assert not sched.probe_mode
    # a job that timed out gets a longer timeout next time, and more again next round
    assert sched.timeout_for(jobs[0]) == pytest.approx(90.0)
    sched.round = 2
    assert sched.timeout_for(jobs[0]) == pytest.approx(180.0)

def test_a_floor_at_the_ceiling_means_transport_errors_never_shrink_a_lane():
    # fill_v2.yaml sets min_lane_limit >= every max_concurrency: SSL-EOF bursts hit all lanes
    # at once, and x0.85 per error against +1 per 25 successes held the slowest lane at ~9/20.
    now = [0.0]
    s = fr.Settings(bytes_budget_mb=24, min_bytes_budget_mb=20, decrease_factor=0.85,
                    decrease_gap_s=0.0, min_lane_limit=24, network_down_after=1000,
                    max_inflight=48)
    sched = fr.Scheduler([_lane("slow", 40, conc=20, nbytes=100_000)], s, clock=lambda: now[0])
    with sched.cv:
        jobs = [sched._pick(now[0]) for _ in range(20)]
    assert all(j is not None for j in jobs)
    for job in jobs:
        sched.finish(job, fr.Outcome("transport", "EOF occurred in violation of protocol"))
    assert sched.lanes["slow"].limit == pytest.approx(20.0)
    assert sched.bytes_budget == pytest.approx(20e6)
    with sched.cv:
        job = sched._pick(now[0])
    sched.finish(job, fr.Outcome("rate_limit", "HTTP 429"))
    assert sched.lanes["slow"].limit == pytest.approx(10.0)
