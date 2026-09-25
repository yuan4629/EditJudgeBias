"""Content-pool overlap detector tests (the data-integration overlap gate).

The two dimensions are tested independently on purpose: the regression this gate
exists to prevent (GenAI-Bench vs MagicBrush dev, 96.6% overlap) is invisible to
the pixel dimension, and the one before it (ImagenHub ⊂ MagicBrush dev) would be
invisible to the text dimension if instructions had been paraphrased.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edit_judge_bias.data import io
from edit_judge_bias.data.overlap import (
    VERDICT_INDEPENDENT,
    VERDICT_NEEDS_EXCLUSION,
    VERDICT_REJECT,
    Sha1Cache,
    build_index,
    check,
    compare,
    exit_code_for,
    main,
    normalize_instruction,
)
from edit_judge_bias.data.schema import SampleRecord


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _sample(root: Path, sid: str, source: str, instruction: str, orig: str,
            pixels: bytes | None = None, model: str = "m1") -> SampleRecord:
    """One record; writes the original image bytes so sha1 has something to read."""
    if pixels is not None:
        p = root / orig
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pixels)
    edited = f"edited/{sid}.png"
    (root / edited).parent.mkdir(parents=True, exist_ok=True)
    (root / edited).write_bytes(b"edited")
    return SampleRecord(
        sample_id=sid, source_dataset=source, edit_type="add",
        content_category="object", original_image_path=orig,
        instruction=instruction, edit_model=model, edited_image_path=edited,
    )


def _write(root: Path, name: str, records) -> Path:
    path = root / "data" / "manifests" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write_jsonl(path, records)
    return path


@pytest.fixture()
def pools(tmp_path: Path):
    """An 'existing' pool plus three candidate sources with known relationships."""
    old = [
        _sample(tmp_path, "o1", "OldPool", "Add a hat to the man.", "orig/a.png", b"AAA"),
        _sample(tmp_path, "o2", "OldPool", "Remove the car.", "orig/b.png", b"BBB"),
        _sample(tmp_path, "o3", "OldPool", "Make the sky blue.", "orig/c.png", b"CCC"),
        _sample(tmp_path, "o4", "OldPool", "Replace the dog with a cat.", "orig/d.png", b"DDD"),
    ]
    # Same content pool, regenerated pixels: text collides, sha1 does not.
    regen = [
        _sample(tmp_path, "r1", "Regen", "add a hat to the man", "regen/a.png", b"XXX"),
        _sample(tmp_path, "r2", "Regen", "Remove the car!", "regen/b.png", b"YYY"),
        _sample(tmp_path, "r3", "Regen", "Make the sky blue", "regen/c.png", b"ZZZ"),
        _sample(tmp_path, "r4", "Regen", "Blur the background.", "regen/d.png", b"WWW"),
    ]
    # Redistributed inputs, paraphrased instructions: sha1 collides, text does not.
    redis = [
        _sample(tmp_path, "d1", "Redist", "Put a hat on him.", "redist/a.png", b"AAA"),
        _sample(tmp_path, "d2", "Redist", "Delete the vehicle.", "redist/b.png", b"BBB"),
        _sample(tmp_path, "d3", "Redist", "Recolour the sky.", "redist/c.png", b"CCC"),
        _sample(tmp_path, "d4", "Redist", "Add snow.", "redist/d.png", b"DDD"),
    ]
    fresh = [
        _sample(tmp_path, "f1", "Fresh", "Add a bicycle.", "fresh/a.png", b"111"),
        _sample(tmp_path, "f2", "Fresh", "Remove the lamp post.", "fresh/b.png", b"222"),
        _sample(tmp_path, "f3", "Fresh", "Turn the grass yellow.", "fresh/c.png", b"333"),
        _sample(tmp_path, "f4", "Fresh", "Swap the mug for a glass.", "fresh/d.png", b"444"),
    ]
    return {
        "root": tmp_path,
        "old": _write(tmp_path, "samples_old.jsonl", old),
        "regen": _write(tmp_path, "samples_regen.jsonl", regen),
        "redist": _write(tmp_path, "samples_redist.jsonl", redis),
        "fresh": _write(tmp_path, "samples_fresh.jsonl", fresh),
    }


# --------------------------------------------------------------------------- #
# normalize_instruction                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", [
    ("Add a hat.", "add a hat"),
    ("Remove  the   car!", "Remove the car"),
    ("Make the sky BLUE?", "make the sky blue"),
])
def test_normalization_collides_on_punctuation_case_and_spacing(a, b):
    assert normalize_instruction(a) == normalize_instruction(b)


def test_normalization_keeps_genuinely_different_instructions_apart():
    assert normalize_instruction("Add a hat") != normalize_instruction("Add a cat")


# --------------------------------------------------------------------------- #
# The two dimensions, each alone                                               #
# --------------------------------------------------------------------------- #
def test_text_dimension_catches_a_regenerated_pool_that_pixels_miss(pools):
    """The GenAI-Bench case: same instructions, every byte different."""
    new = build_index(pools["regen"], root=pools["root"])
    old = build_index(pools["old"], root=pools["root"])
    r = compare(new, old)
    assert r["instruction_overlap"] == pytest.approx(0.75)
    assert r["sha1_overlap"] == 0.0
    assert r["verdict"] == VERDICT_REJECT  # 0.75 >= REJECT_AT


def test_pixel_dimension_catches_redistributed_inputs_that_text_misses(pools):
    """The ImagenHub case: identical original bytes, paraphrased instructions."""
    new = build_index(pools["redist"], root=pools["root"])
    old = build_index(pools["old"], root=pools["root"])
    r = compare(new, old)
    assert r["instruction_overlap"] == 0.0
    assert r["sha1_overlap"] == pytest.approx(1.0)
    assert r["verdict"] == VERDICT_REJECT


def test_a_genuinely_new_pool_passes_both_dimensions(pools):
    new = build_index(pools["fresh"], root=pools["root"])
    old = build_index(pools["old"], root=pools["root"])
    r = compare(new, old)
    assert (r["instruction_overlap"], r["sha1_overlap"]) == (0.0, 0.0)
    assert r["verdict"] == VERDICT_INDEPENDENT


def test_reverse_direction_distinguishes_subset_from_superset(tmp_path: Path):
    """A 2-of-2 subset of a 10-record pool: forward 1.0, reverse 0.2."""
    big = [_sample(tmp_path, f"b{i}", "Big", f"Edit number {i}.", f"o/{i}.png",
                   bytes([i])) for i in range(10)]
    small = [_sample(tmp_path, f"s{i}", "Small", f"Edit number {i}.", f"s/{i}.png",
                     bytes([i])) for i in range(2)]
    new = build_index(_write(tmp_path, "samples_small.jsonl", small), root=tmp_path)
    other = build_index(_write(tmp_path, "samples_big.jsonl", big), root=tmp_path)
    r = compare(new, other)
    assert r["instruction_overlap"] == pytest.approx(1.0)
    assert r["instruction_overlap_reverse"] == pytest.approx(0.2)
    assert r["sha1_overlap"] == pytest.approx(1.0)
    assert r["sha1_overlap_reverse"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Verdict thresholds                                                           #
# --------------------------------------------------------------------------- #
def test_overlap_between_thresholds_and_reject_asks_for_an_exclusion_block(tmp_path: Path):
    """20 records, 3 shared instructions = 15% -> over 10% but under 60%."""
    old = [_sample(tmp_path, f"o{i}", "Old", f"Edit {i}.", f"o/{i}.png", bytes([i]))
           for i in range(20)]
    new = [_sample(tmp_path, f"n{i}", "New", f"Edit {i}." if i < 3 else f"Novel {i}.",
                   f"n/{i}.png", bytes([200 + i])) for i in range(20)]
    idx_new = build_index(_write(tmp_path, "samples_new.jsonl", new), root=tmp_path)
    idx_old = build_index(_write(tmp_path, "samples_old.jsonl", old), root=tmp_path)
    r = compare(idx_new, idx_old)
    assert r["instruction_overlap"] == pytest.approx(0.15)
    assert r["verdict"] == VERDICT_NEEDS_EXCLUSION


def test_worst_comparison_decides_the_overall_verdict(pools):
    report = check(pools["fresh"], [pools["old"]], root=pools["root"])
    assert report["_meta"]["verdict"] == VERDICT_INDEPENDENT
    report = check(pools["fresh"], [pools["old"], pools["regen"], pools["redist"]],
                   root=pools["root"])
    # Fresh is independent of Old, but overlaps nothing else either -> still clean.
    assert report["_meta"]["verdict"] == VERDICT_INDEPENDENT
    report = check(pools["regen"], [pools["fresh"], pools["old"]], root=pools["root"])
    assert report["_meta"]["verdict"] == VERDICT_REJECT


@pytest.mark.parametrize("verdict,code", [
    (VERDICT_INDEPENDENT, 0), (VERDICT_NEEDS_EXCLUSION, 1), (VERDICT_REJECT, 2),
])
def test_exit_code_is_the_gate(verdict, code):
    assert exit_code_for({"_meta": {"verdict": verdict}}) == code


# --------------------------------------------------------------------------- #
# Report shape (the published contract in <src>.yaml provenance.overlap_report) #
# --------------------------------------------------------------------------- #
def test_report_keys_are_vs_source_name_plus_meta(pools):
    report = check(pools["regen"], [pools["old"]], root=pools["root"])
    assert set(report) == {"vs_OldPool", "_meta"}
    assert report["_meta"]["new_source"] == "Regen"
    assert report["_meta"]["new_records"] == 4


def test_two_manifests_of_the_same_source_do_not_collide(tmp_path: Path):
    recs = [_sample(tmp_path, "x1", "Same", "Do a thing.", "x/1.png", b"1")]
    a = _write(tmp_path, "samples_a.jsonl", recs)
    b = _write(tmp_path, "samples_b.jsonl", recs)
    new = _write(tmp_path, "samples_n.jsonl",
                 [_sample(tmp_path, "n1", "New", "Other thing.", "n/1.png", b"9")])
    report = check(new, [a, b], root=tmp_path)
    assert set(report) == {"vs_Same", "vs_Same@samples_b", "_meta"}


def test_mixed_source_manifest_is_named_by_its_stem(pools, tmp_path: Path):
    mixed = _write(tmp_path, "samples_pool_v9.jsonl", [
        _sample(tmp_path, "m1", "SrcA", "Thing one.", "m/1.png", b"1"),
        _sample(tmp_path, "m2", "SrcB", "Thing two.", "m/2.png", b"2"),
    ])
    report = check(pools["fresh"], [mixed], root=pools["root"])
    assert "vs_samples_pool_v9" in report


# --------------------------------------------------------------------------- #
# Running before the pixels exist (protocol §2 chicken-and-egg escape)         #
# --------------------------------------------------------------------------- #
def test_no_pixels_mode_still_reports_the_text_dimension(pools):
    report = check(pools["regen"], [pools["old"]], root=pools["root"], with_pixels=False)
    r = report["vs_OldPool"]
    assert r["instruction_overlap"] == pytest.approx(0.75)
    assert r["sha1_overlap"] is None
    assert report["_meta"]["pixels_compared"] is False


def test_missing_original_files_are_counted_not_fatal(tmp_path: Path):
    recs = [_sample(tmp_path, "p1", "Partial", "Present.", "p/1.png", b"1"),
            _sample(tmp_path, "p2", "Partial", "Absent.", "p/2.png", None)]
    idx = build_index(_write(tmp_path, "samples_p.jsonl", recs), root=tmp_path)
    assert idx.missing_files == 1
    assert idx.hashed == 1
    assert len(idx.sha1s) == 1


def test_sha1_share_is_over_hashable_originals_not_all_records(tmp_path: Path):
    """One of two originals is missing; a single match must read as 100%, not 50%."""
    old = [_sample(tmp_path, "o1", "Old", "Alpha.", "o/1.png", b"SHARED")]
    new = [_sample(tmp_path, "n1", "New", "Beta.", "n/1.png", b"SHARED"),
           _sample(tmp_path, "n2", "New", "Gamma.", "n/2.png", None)]
    idx_new = build_index(_write(tmp_path, "samples_n.jsonl", new), root=tmp_path)
    idx_old = build_index(_write(tmp_path, "samples_o.jsonl", old), root=tmp_path)
    assert compare(idx_new, idx_old)["sha1_overlap"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# sha1 cache                                                                   #
# --------------------------------------------------------------------------- #
def test_sha1_cache_round_trips_and_invalidates_on_change(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"first")
    cache_path = tmp_path / "provenance" / ".sha1_cache.json"
    cache = Sha1Cache(cache_path)
    first = cache.get(img)
    cache.save()
    assert json.loads(cache_path.read_text(encoding="utf-8"))

    reloaded = Sha1Cache(cache_path)
    assert reloaded.get(img) == first
    img.write_bytes(b"second and longer")
    assert reloaded.get(img) != first


def test_corrupt_cache_file_is_ignored(tmp_path: Path):
    cache_path = tmp_path / ".sha1_cache.json"
    cache_path.write_text("{not json", encoding="utf-8")
    img = tmp_path / "img.png"
    img.write_bytes(b"data")
    assert Sha1Cache(cache_path).get(img)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_writes_the_report_and_returns_the_gate_code(pools, tmp_path: Path, capsys):
    out = tmp_path / "data" / "provenance" / "overlap_regen.json"
    code = main(["--new", str(pools["regen"]), "--against", str(pools["old"]),
                 "--out", str(out), "--root", str(pools["root"]),
                 "--cache", str(tmp_path / ".cache.json")])
    assert code == 2
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["vs_OldPool"]["instruction_overlap"] == pytest.approx(0.75)
    assert "reject" in capsys.readouterr().out


def test_cli_dry_run_touches_nothing(pools, tmp_path: Path):
    out = tmp_path / "never.json"
    assert main(["--new", str(pools["regen"]), "--against", str(pools["old"]),
                 "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


# --------------------------------------------------------------------------- #
# Near-duplicate dimension                                                     #
# --------------------------------------------------------------------------- #
def _photo(path: Path, seed: int, size: int = 512) -> None:
    """A textured pseudo-photo — ORB needs corners, so noise beats a gradient."""
    np = pytest.importorskip("numpy")
    from PIL import Image
    rng = np.random.default_rng(seed)
    # Blocky noise upsampled to `size`: plenty of repeatable corners at both scales.
    small = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((size, size), Image.BICUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


def _cropped_reencode(src: Path, dst: Path) -> None:
    """What EBench-18K did to the restoration datasets: crop, resize, re-encode."""
    from PIL import Image
    img = Image.open(src)
    w, h = img.size
    img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))) \
       .resize((400, 400), Image.BICUBIC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))) \
       .resize((400, 400), Image.BICUBIC).save(dst, quality=80)


@pytest.fixture()
def cropped_pools(tmp_path: Path):
    """Old pool of 4 photos; new source re-uses 2 of them cropped, plus 2 fresh."""
    pytest.importorskip("cv2")
    for i in range(4):
        _photo(tmp_path / f"old/{i}.jpg", seed=100 + i)
    for i in range(2):
        _cropped_reencode(tmp_path / f"old/{i}.jpg", tmp_path / f"new/{i}.jpg")
    for i in range(2, 4):
        _photo(tmp_path / f"new/{i}.jpg", seed=900 + i)
    old = [_sample(tmp_path, f"o{i}", "Old", f"Old instruction {i}.", f"old/{i}.jpg")
           for i in range(4)]
    new = [_sample(tmp_path, f"n{i}", "New", f"Deblur the image.", f"new/{i}.jpg")
           for i in range(4)]
    return {"root": tmp_path,
            "old": _write(tmp_path, "samples_old.jsonl", old),
            "new": _write(tmp_path, "samples_new.jsonl", new)}


def test_sha1_misses_a_cropped_reencode_that_near_dup_catches(cropped_pools):
    """The EBench-18K case. This is the whole reason the third dimension exists."""
    root = cropped_pools["root"]
    report = check(cropped_pools["new"], [cropped_pools["old"]], root=root)
    assert report["vs_Old"]["sha1_overlap"] == 0.0  # bytes differ: exact hashing blind
    assert report["vs_Old"]["verdict"] == VERDICT_INDEPENDENT  # ...and wrongly clean

    report = check(cropped_pools["new"], [cropped_pools["old"]], root=root,
                   near_dup_sample=10)
    near = report["vs_Old"]["near_dup"]
    assert near["sampled"] == 4
    assert near["rate"] == pytest.approx(0.5)  # 2 of 4 originals are recycled
    assert report["vs_Old"]["verdict"] == VERDICT_NEEDS_EXCLUSION


def test_near_dup_matches_name_the_offending_pair_with_its_inlier_count(cropped_pools):
    report = check(cropped_pools["new"], [cropped_pools["old"]],
                   root=cropped_pools["root"], near_dup_sample=10)
    matches = report["vs_Old"]["near_dup"]["matches"]
    assert {m["new"] for m in matches} == {"new/0.jpg", "new/1.jpg"}
    assert all(m["inliers"] >= 50 for m in matches)
    # Each recycled original is matched to the photo it actually came from.
    assert {(m["new"], m["other"]) for m in matches} == {
        ("new/0.jpg", "old/0.jpg"), ("new/1.jpg", "old/1.jpg")}


def test_unrelated_photos_do_not_trip_the_inlier_threshold(tmp_path: Path):
    pytest.importorskip("cv2")
    for i in range(4):
        _photo(tmp_path / f"a/{i}.jpg", seed=1000 + i)
        _photo(tmp_path / f"b/{i}.jpg", seed=2000 + i)
    a = _write(tmp_path, "samples_a.jsonl",
               [_sample(tmp_path, f"a{i}", "A", f"Alpha {i}.", f"a/{i}.jpg") for i in range(4)])
    b = _write(tmp_path, "samples_b.jsonl",
               [_sample(tmp_path, f"b{i}", "B", f"Beta {i}.", f"b/{i}.jpg") for i in range(4)])
    report = check(a, [b], root=tmp_path, near_dup_sample=10)
    assert report["vs_B"]["near_dup"]["rate"] == 0.0
    assert report["_meta"]["verdict"] == VERDICT_INDEPENDENT


def test_near_dup_is_off_unless_asked_for(cropped_pools):
    report = check(cropped_pools["new"], [cropped_pools["old"]], root=cropped_pools["root"])
    assert report["vs_Old"]["near_dup"] is None
    assert report["_meta"]["near_dup_sample"] == 0


def test_near_dup_sample_is_seeded_and_reported(cropped_pools):
    """A sample of 2 out of 4 must be reproducible and must say it was a sample."""
    root = cropped_pools["root"]
    first = check(cropped_pools["new"], [cropped_pools["old"]], root=root, near_dup_sample=2)
    again = check(cropped_pools["new"], [cropped_pools["old"]], root=root, near_dup_sample=2)
    assert first["vs_Old"]["near_dup"]["sampled"] == 2
    assert first["vs_Old"]["near_dup"] == again["vs_Old"]["near_dup"]


def test_cli_near_dup_flag_defaults_its_sample_size(cropped_pools, tmp_path: Path):
    code = main(["--new", str(cropped_pools["new"]), "--against", str(cropped_pools["old"]),
                 "--root", str(cropped_pools["root"]), "--near-dup",
                 "--cache", str(tmp_path / "c.json")])
    assert code == 1  # needs_exclusion
