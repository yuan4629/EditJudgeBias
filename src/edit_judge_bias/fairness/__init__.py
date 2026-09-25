"""Fairness track.

This package contains exploratory analyses that are not part of the
experiments reported in the submitted paper.

The demographic biases (gender, skin_tone) are a *separate* track from the main
A/B/C invariance benchmark: attribute edits change the person, so the three-way
quality-preservation invariant no longer applies and we switch to a counterfactual
fairness protocol (attribute score gap, paired Wilcoxon). This package holds:

- ``image_edit_client`` — a client for an OpenAI-compatible ``/v1/images/edits``
  endpoint (e.g. a Qwen-Image-Edit deployment), used under scheme alpha to
  edit the *original* image ("change the person to a woman, keep everything else
  identical"), producing a matched counterfactual original.
- ``attribute_templates`` — the gender / skin_tone edit prompts and control axes.

Ethics: identity-irrelevant instructions only; aggregate reporting, no individual
call-outs.
"""
