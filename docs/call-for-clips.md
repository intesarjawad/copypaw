# Call for field-test clips

Draft: the private Dropbox file request is not configured. I will configure it
before inviting submissions, after the protocol's pre-registration commit.

Private submission route: not yet configured.

Please send a side-view clip of one animal walking, trotting or running, lasting
3 to 8 s. We are collecting the first 20 eligible clips in arrival order, with
at least 10 dogs, 5 cats, 3 dark-furred animals, 3 portrait HEVC clips and 2 clips
with a person in frame. We ask for more of a missing kind rather than dropping
an arrival already logged. Pipeline failures stay in the results.

If your child has drawn the pet, you may include that drawing with
[the artwork release](consent/artwork_release.md). Otherwise, we use a sample
drawing I made. Include [the clip release](consent/clip_release.md).
We ask field-test families only for the pre-ticked evaluation-only scope:
run through the pipeline and seen by the two raters, never published.
A person in frame forces evaluation-only scope. Do not provide a child's full
name, face or age. The releases let you state a credit preference; assets stay
outside the repo licence.

Use only the private route above: never post clips or drawings in an issue or send them by email.

Collection closes three days before launch. If fewer than 20 clips arrive, I choose
between moving the launch and publishing "interim, k of 20, collection
open", keeping the protocol unchanged. Read [the protocol](field_test.md) for
the rubric, trials and analysis.

## Dropbox disclosure

Submissions pass through Dropbox, a company outside this project; Dropbox may ask
the sender for a name and email address so I can tell who sent what
([help.dropbox.com/share/received-file-request](https://help.dropbox.com/share/received-file-request),
checked 2026-11-04); any name works there, and opening the link in a private
browser window keeps a signed-in Dropbox account's details from being prefilled;
I move each arrival to a local folder and empty Dropbox on arrival,
and both the files and the request are deleted on the schedule below; Dropbox's
own privacy policy applies while the files are with it
([dropbox.com/privacy](https://dropbox.com/privacy), effective 2025-05-30,
checked 2026-11-04; Dropbox announces a change effective 2027-01-01, so the link,
not a summary, is given).

The name and email never enter the repo or `field_test.csv`.

## Private handling, deletion and withdrawal

Submit clips and drawings only through
the private Dropbox route, once configured.
Never post clips or drawings in an issue or send them by email.
I move arrivals to `~/Copypaw-field-test/` on my Mac,
outside every git repo, and run them there. The two raters watch on that Mac,
in person, and nothing is uploaded.

When `eval_motion.py --field-test` first writes results, it opens an
issue titled "Delete field-test raw material", due 30 days later, without file
names. A withdrawal opens an issue due in 7 days. Both the files and the Dropbox
request are deleted on this schedule. Withdrawal removes the asset from future
releases and the site within 7 days. Copies already shared cannot be recalled.
