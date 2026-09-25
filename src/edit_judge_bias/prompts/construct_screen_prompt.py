"""Construct-validity screen for a D-class skin-lightness scene. ONE image, four questions.

★ WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL
The numeric pool screens measure geometry (person fraction, face fraction, skin fraction,
blob count) and they are now good at it -- the framing band selects head-and-shoulders single
subjects where the old pool selected gulls and guitar bodies. But a contact sheet of a seeded
random sample of 72 in-band OmniEdit scenes found **30 of them (42%) unusable** for reasons no
geometric measurement can reach:

    multi-subject frames, couples, and side-by-side collages   (the biggest group)
    non-photographic subjects -- CGI, anime, 3D renders, statues, an ogre
    theatrical face paint (a Joker, a skull, a vampire) that hides the skin being measured
    MINORS whose instructions never mention a child                <- an ethics hard stop

★ TWO FREE SCREENS WERE BUILT FOR THESE AND BOTH WERE REJECTED ON THEIR OWN CONTROLS.
"Exactly one detected face" agreed with hand labels on 6 of 8 controls, and BOTH misses were
two-subject scenes reported as one -- Haar's failure is recall, so no merging fixes it and the
error direction is the harmful one. "The skin region has chroma" works (B&W control 7.5 versus
colour 22-33) but removes 2 of 723, because the skin mask is itself a chroma rule. There is no
free path to these four questions; that is the finding, not an excuse.

★ EVERY QUESTION IS PHRASED SO THAT "TRUE" IS THE USABLE ANSWER.
`no_minor`, not `any_minor`. The parser derives `pass` as the conjunction and treats a missing
field as NOT satisfied, so a dropped or unparseable field can only ever cost a scene, never
admit one. Getting this backwards on the ethics question is the one error with no recovery.
"""

from __future__ import annotations

_TEMPLATE = """You are screening a photograph for a research dataset. Look at the image and \
answer four questions about it. Be strict: when a question is borderline, answer false.

1. is_photograph: was this image captured by a camera pointed at real human beings? Answer \
true even if the scene is staged, posed, a studio or fashion shoot, a red-carpet or press \
photo, or a still from a film or television production -- those are photographs of real \
people. Answer false only if no real person was photographed: illustration, painting, \
drawing, anime, cartoon, 3D render, CGI, video-game capture, statue, waxwork, doll, \
mannequin, or a fully AI-generated image.
2. single_subject: is there exactly ONE person whose FACE is visible? Answer false if a \
second person's face is also visible, if there is a crowd or an audience behind the subject, \
or if the image is a collage or side-by-side composite of more than one photo. A stray hand \
or arm at the edge of the frame does not by itself make the answer false.
3. no_minor: does everyone shown appear to be an adult (18 or older)? Answer false if any \
child, infant or teenager appears anywhere in the frame, including in the background.
4. skin_visible: is the subject's bare skin (face, neck or arms) clearly visible in natural \
colour? Answer false for heavy theatrical face paint, clown or skull makeup, body paint, a \
covering mask, or a black-and-white / monochrome photograph.

"pass" is true ONLY if all four are true.

Return only valid JSON:
{
  "is_photograph": true/false,
  "single_subject": true/false,
  "no_minor": true/false,
  "skin_visible": true/false,
  "pass": true/false,
  "reason": "<short reason>"
}"""


def build_construct_screen_prompt() -> str:
    """The screen prompt. Takes no arguments on purpose.

    ⚠️ The wording is CALIBRATED: the reported false-reject floor and the agreement with a
    human read were both measured against this exact text. Changing a question invalidates
    both and they must be re-measured before the screen is used to gate anything.

    ★ REVISION 2, 2026-07-31, after the first calibration returned a 25% false-reject floor
    and the seven rejected controls were adjudicated at FULL SIZE (not thumbnail size). Two
    specification gaps in revision 1 accounted for most of it, and both were mine:

    * `is_photograph` rejected FILM AND TELEVISION STILLS as "not a real photograph". Three of
      the seven. A still is a camera pointed at a real actor, which is exactly the subject
      this arm needs; the criterion that was meant is "was a real person photographed", not
      "is the scene candid". Now stated positively, with staged/studio/press/film named as
      passing.
    * `single_subject` counted a stray arm at the frame edge as a second person. It caught one
      genuine case (another person reaching into a vending machine) and over-rejected two. The
      construct is "whose skin tone is being manipulated", which a second FACE compromises and
      a background limb does not, so the question now asks about faces.

    `no_minor` was NOT touched. It flagged 4 of 20 study rows, and at full size three are
    plainly teenaged while the fourth is CGI. Its one control error (a young-looking adult in
    an evening gown) is in the safe direction, which the prompt asks for deliberately.
    """
    return _TEMPLATE


#: Fields the screen returns, in the order the report prints them. `pass` is their conjunction.
SCREEN_FIELDS = ("is_photograph", "single_subject", "no_minor", "skin_visible")
