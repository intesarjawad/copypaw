# Field-test protocol

## Pre-registration and collection deadline

The pre-registration is the commit that gave the protocol, everything outside
the results block, its current text: walking back from the newest commit
touching this file, the oldest one whose protocol still matches. It must precede
my invitation and collection. The protocol outside the results block remains
unchanged after that commit. Scripts write only inside the results block, so an
edit to the protocol after collection opens moves the pre-registration past the
arrivals already logged, and the checks below fail.

When `scripts/eval_motion.py --field-test` first writes results, its first entry
in the block is the pre-registration commit's hash and ISO timestamp.

Collection closes three days before launch. If fewer than 20 clips have arrived, I decide
between moving the launch and publishing the results block as
"interim, k of 20, collection open". The protocol stays unchanged. The script
writes the same interim wording in the README's field-test line.

## Inputs, arrival order and quotas

Collect 20 uncurated clips. By-eye eligibility requires a side view of one animal
walking, trotting or running, lasting 3 to 8 s. Take the first 20 eligible clips
in arrival order. Log each clip's arrival date in `docs/field_test.csv` before
running it.

The collection must include at least:

| Kind | Clips |
| --- | --- |
| Dogs | 10 |
| Cats | 5 |
| Dark-furred animals | 3 |
| Portrait HEVC clips | 3 |
| Clips with a person in frame | 2 |

Meet quotas by asking for more of the missing kind, never by dropping a clip
already logged. A clip that fails in the pipeline stays as a failure row.

## Run and drawing assignment

Run each clip once, with defaults only, then allow dot fixes. Run on my
Mac. Use the family's own child's drawing when supplied and record its artwork
release ID. Otherwise, assign one of the six sample drawings I made by a seeded
shuffle. Read `joints_moved` from that run's `rig_edits.json` and record the
`quality.json` warnings.

## Raters, session order and usability rubric

I am rater 1. Rater 2 has not worked on the project and has seen no output
of it before the session. Raters watch on my Mac, in person.

Complete the two-choice trials for all clips before the rubric, so no rater has
seen a clip's output before its trial. Raters give their rubric answers
independently after watching the side-by-side.

A clip is usable only when all of these hold:

- The pipeline finishes without error.
- `gait_r >= 0.6`.
- Both raters independently answer yes to "is this recognisably this animal's movement".
- There is no limb tear.

## Two-choice trials

For each clip, `scripts/eval_motion.py --make-2afc` renders the same drawing driven
by three other field-test clips, the foils. Assign foils by a seeded shuffle,
using the same species where enough clips exist and topping up from any species
when fewer exist. Record `foil_same_species` for every trial. A cross-species foil
is easier to reject, so report same-species-only accuracy and p-values beside
the totals.

For each foil, produce randomised, unlabelled pairs in both conditions:

| Condition | Pair construction | Status |
| --- | --- | --- |
| Offset | Each render starts at an independent seeded random frame within its first stride and loops. Footfall sync with the clip cannot decide the choice; only gait character can. | Primary outcome, the stronger claim |
| Synced | Both renders start at the clip's first frame. | Secondary outcome |

Each rater picks which render moves like the pet. Interleave all trials for all
clips in one seeded random order. There are 60 trials per rater per condition.
Store picks in `docs/field_test_2afc.csv`, with one row per trial and columns
`clip`, `foil`, `foil_same_species`, `condition`, `rater`, `pick`, `correct`.
The complete collection has 240 trial rows, 6 per clip per rater.

## Pre-registered tests and power

For each rater and condition, use a one-sided binomial test against chance
accuracy 0.5, at alpha 0.05. The 60-trial test needs 37 correct answers; its exact
alpha is 0.046. Do not pool raters' counts: they see the same clips.

| True per-trial accuracy | Power with 60 trials | Power with 20 trials |
| --- | --- | --- |
| 65% | 0.75 | 0.25 |
| 70% | 0.94 | 0.42 |
| 75% | 0.99 | 0.62 |

The three trials for a clip share that clip. As a pre-registered sensitivity
analysis, report a clip-level sign test for each rater and condition over the
20 clips: is the majority of each clip's three picks correct? A clip's majority
is correct with probability `p^3 + 3p^2(1-p)`, where `p` is per-trial accuracy.
The sign test needs 15 of 20 clips correct, with alpha 0.021.

| True per-trial accuracy | Clip-level sign-test power |
| --- | --- |
| 65% | 0.49 |
| 70% | 0.75 |
| 75% | 0.92 |

## Public-claim decision rule

Make the public claim "raters pick the right render above chance" only if, in
the offset condition, both raters' 60-trial binomial tests and both raters'
clip-level sign tests reach `p < 0.05`.

Otherwise, every public mention reports all four offset p-values and makes no
claim. Describe synced results only as secondary. `eval_motion.py` prints the
verdict line, "claim met" or "claim not met", which README and launch copy quote.

## Records and analysis

`docs/field_test.csv` contains one row per clip: arrival date; clip licence or
consent ID; `scope` (`evaluation` or `publish`); `person_in_frame` (`yes` or `no`);
drawing source (child's own with release ID, or my sample); `joints_moved`;
`quality.json` warnings; `gait_r`; both raters' yes/no verdicts; failure category.
`person_in_frame` set to `yes` forces `scope` to `evaluation`.

`scripts/eval_motion.py --field-test docs/field_test.csv` writes the table and
summary only inside the results block. It reports:

- Usable count and Cohen's kappa for the two raters' yes/no verdicts.
- Each rater's correct two-choice picks out of 60 in each condition, with each
  one-sided binomial p-value against 0.5.
- Clip-level sign tests for each rater and condition.
- Same-species-only accuracy and p-values beside the totals.
- The decision-rule verdict line.
- Separate summaries for child-drawn and engineer-drawn rows.
- Failures retained and categorised, never removed.

The clip CSV must have 20 rows, or k rows marked interim under the deadline rule.
The script fails if any arrival date precedes the pre-registration commit's date.
`tests/unit/test_field_test_protocol.py` finds the pre-registration commit by
comparing text outside the results block with `git show <hash>:docs/field_test.md`
and fails on the same arrival dates.

## Consent, private handling and deletion

Use [the artwork release](consent/artwork_release.md) and
[the clip release](consent/clip_release.md), prepared before I invite submissions.
Evaluation only is pre-ticked; publication is a separate unticked opt-in. Ask
field-test families only for the default. I request publication permission
separately for material intended for public examples.
Record credit preference, and never collect a child's full name, face or age.
Assets stay outside the repo licence. Withdrawal removes the asset from future
releases and the site within 7 days; copies already shared cannot be recalled.

Clips and drawings arrive only through the private Dropbox route in
[the call for clips](call-for-clips.md), which carries the disclosure also
included word for word in both releases. Never post clips or drawings in an
issue or send them by email. I move each arrival to
`~/Copypaw-field-test/` on my Mac, outside every git repo, and empty
Dropbox on arrival. Run there; raters watch there in person and nothing is
uploaded. The Dropbox sender's name and email never enter the repo or the CSV.

When `eval_motion.py --field-test` first writes results, it opens an
issue titled "Delete field-test raw material", due 30 days later, with no file
names in the issue. A withdrawal opens an issue due in 7 days. Delete both the
files and the Dropbox request on this schedule.

The consent-scope check in `tests/unit/test_assets_clean.py`, required before
any field-test asset exists, must prevent assets with an evaluation-only consent
ID or `person_in_frame` set to `yes` from being committed or built into the site.

## Results

<!-- results:start -->
<!-- results:end -->
