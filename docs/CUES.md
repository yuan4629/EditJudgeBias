# Cue catalogue

A **cue** is a change to the judge's input that should not change a correct verdict: it is
designed to leave what was edited, and how well, intact. For image cues this is checked per
cue on a sample of rendered images, not certified for every image (see stage 3 in
[REPRODUCTION.md](REPRODUCTION.md#3-preservation-validation-paid)).

Each image cue is a deterministic injector in `src/edit_judge_bias/bias/`, parameterised by
`configs/bias/<cue>.yaml`, and writes a new image. Originals and edited images are never
modified. Prompt cues change only the text around the images.

Sizes scale with the image's short edge, so a cue is equally prominent at 512 px and 1024
px. The C-class cues use an estimate of the edit region, taken from the original-vs-edited
difference or from the dataset's mask when one ships.

## Protocol

| Cue | What changes | Where |
|---|---|---|
| `position` | the two edits of a pair are shown in swapped A/B order; no pixel changes | pairwise runner (`include_position_swap`) |

## Pixels, global (B-class)

| Cue | What it does | Default parameters |
|---|---|---|
| `brightness` | scales brightness | factor 1.2 |
| `saturation` | scales colour saturation | factor 1.2 |
| `watermark` | semi-transparent authority text in a corner | "Reference Image", opacity 0.45, bottom-right |
| `text_overlay` | a semi-transparent band with a keyword from the instruction | top band, opacity 0.5 |
| `padding` | black border around the image | 10% of each dimension per side |
| `aesthetic_filter` | a "beauty filter": contrast, warm cast, unsharp mask | contrast 1.1, warm shift 8 |

## Content near the edit (C-class)

| Cue | What it does | Default parameters |
|---|---|---|
| `zoom_inset` | a magnified crop of the edit region, placed in a corner | zoom 2.0, inset at most 40% of the frame |
| `detail_caption` | a caption strip below the image restating the instruction ("Edited: ...") | at most 14 words |
| `distraction` | a small synthetic sticker placed **outside** the edit region | area at most 4% of the image |
| `region_annotation` | a red box around the edit region | stroke 3 px at 512 px |

## Prompt (A-class)

| Cue | What changes in the prompt |
|---|---|
| `bandwagon` | fabricated social proof: "Note: previous reviewers rated this edit very highly." In pairwise: "Note: 90% of human evaluators preferred Image A/B.", with the endorsed side seeded per pair |
| `model_name` | "This edited image was produced by the {editor} model.", naming the true editor |

`prompt_bias.py` also defines controls for `model_name`: a permuted editor name, an
anonymous version, and a literal name.

## Controls (not cues)

| Condition | Purpose |
|---|---|
| `sham` | a JPEG round trip at quality 95: the same picture as a new file, run through the same pipeline. Its effect is the zero-dose floor every cue is read against. It is also the validators' calibration: a validator that flags `sham` is miscalibrated. `sham` is a zero-dose control, not a dose-matched one. |
| `edit_damage_{25,50,100}` | reverts that fraction of the estimated edit region to the original pixels, so by construction the instruction is not carried out there. It is the validators' positive control (sensitivity) and is never shown to judges. |
| `edit_damage_blur` | blurs the edit region: the edit stays, its detail is destroyed. It is a second positive control. |

## Where each cue is used

| Arm (config) | Cues |
|---|---|
| Scoring, breadth block (`scoring_breadth_v2.yaml`) | all 11 image conditions (10 cues + `sham`) + `bandwagon` + `model_name` |
| Scoring, human-anchor block (`scoring_anchor_v2.yaml`) | `padding`, `text_overlay`, `brightness`, `region_annotation` |
| Scoring, anchor fill (`scoring_anchor_fill_v2.yaml`) | the remaining image cues + `sham` + both prompt cues |
| Pairwise (`pairwise_v2.yaml`) | `position` + one-sided `padding`, `text_overlay`, `brightness` |
| Pairwise fill (`pairwise_fill_v2.yaml`) | the remaining one-sided image cues + `sham` + both prompt cues |

In the pairwise arms a cue is **one-sided**: only one of the two images carries it, and
the dressed side is a seeded function of the pair id.

## Adding a cue

1. Subclass `BiasInjector` (`bias/base.py`). Every random draw must use the injector's
   seeded generator.
2. Register the injector in `bias/registry.py`.
3. Add `configs/bias/<cue>.yaml`.
4. Add a test in `tests/test_bias_injectors.py`.
5. Add the cue to an injection config and to the arm configs that should use it.
6. Validate preservation on it (stage 3) before reading any judge effect.
