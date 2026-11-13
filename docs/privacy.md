# Privacy

Copypaw has two separate parts: the product, which you run yourself, and a
consent-based field test, which will collect clips and drawings from volunteers.
They follow different rules, and this page keeps them apart.

## The product

**There is no server.** Copypaw has no upload service and no account. Your pet
clips and drawings are read, processed and written on your own computer.

- **Metadata stays out of the results.** A drawing is decoded into an array and
  a clip is decoded and re-encoded, so EXIF tags and container metadata such as a
  location tag do not enter the outputs. A unit test checks every committed image
  for EXIF and every committed video for location tags.
- **The local app listens on loopback only.** `copypaw app` binds `127.0.0.1` on a
  free port. It makes a random token for each launch, carries it in the opened
  address's fragment (which a browser never sends to a server), and requires it on
  every `/api/` request and every result file. It accepts only the `Host` values
  `127.0.0.1:<port>` and `localhost:<port>`, and refuses request bodies over 60 MB.
- **Inputs live in one temporary folder.** The app stores your inputs and results in
  a `copypaw-session-*` folder under the operating system's temporary directory and
  removes it on exit. If the process is killed, the folder remains until the next
  launch, which removes stale ones.
- **The log is short.** The app logs the method, the route and the status code of each
  request, and nothing from the request body.
- **The site sets no cookies and runs no analytics.** The site and `/try/` load no
  third-party script and use self-hosted fonts. Browser compute, where it is offered,
  stays in your tab.

### What does leave your computer

- **Model weights.** The first `copypaw capture` or app run downloads the pose and
  detector weights from Hugging Face and from a GitHub release, then checks each file
  against a pinned SHA-256. The requests name the weight files; they carry none of your
  inputs. The hosts see your network address, as any download does.
  `copypaw models fetch` does the download on its own, ahead of time.
- **Page requests.** The public site is hosted on GitHub Pages. This project runs no
  server and sees no visitor logs. GitHub operates Pages and keeps its own server logs
  under [GitHub's privacy statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).

### Status of these claims

The server contract above is enforced by `tests/unit/test_app_server.py`, and the
metadata contract by `tests/unit/test_assets_clean.py`. The metadata test covers files
committed to this repository, not clips you supply. The full contract has not been
confirmed by an independent review, and [ADR-009](adr/ADR-009.md) records the open items.

## Consent policy

Every clip or drawing that someone else made enters the project only with a
signed-off release, and the release states a scope:

- **Evaluation only** is the default and is pre-ticked. The material is run through
  the pipeline and seen by the two raters, and never published.
- **Publish** is a separate opt-in covering the repository, the site, social posts and
  talks. It is requested separately for material meant for public examples. It is never
  asked of a field-test family.
- A clip with a **person in frame** is always evaluation only and is never published.
- Do not provide a child's full name, face or age. The release records a credit
  preference: anonymous, a first name or a pseudonym.
- Assets stay outside the repository's licence.
- **Withdrawal** removes the asset from future releases and from the site within 7
  days. Copies already shared cannot be recalled. To withdraw, open an issue that
  gives the consent ID and attaches no files.

A test blocks any asset whose consent ID belongs to an evaluation-only or
person-in-frame row of `docs/field_test.csv` from being committed or built into the
site. The templates are:

- [Artwork release](consent/artwork_release.md)
- [Clip release](consent/clip_release.md)

## The field test

Unlike the product, the field test collects clips and drawings. The protocol is in
[field_test.md](field_test.md) and the invitation is in
[call-for-clips.md](call-for-clips.md). The invitation is still a
draft: the private route below is not configured, and no clip has been collected.

- **Route.** Clips and drawings arrive only through a private Dropbox file request.
  They are never posted in an issue or sent by email. Dropbox is a company outside this
  project; it may ask the sender for a name and email address, and its own privacy
  policy applies while the files are with it. The releases give the details and links.
- **Handling.** Each arrival is moved to a folder on the maintainer's Mac outside every git
  repository, and Dropbox is emptied on arrival. The run and the raters' viewing happen
  on that Mac, and nothing is uploaded.
- **Sender details.** The sender's name and email never enter the repository or
  `docs/field_test.csv`.
- **Deletion.** When `scripts/eval_motion.py --field-test` first writes results, it opens
  an issue titled "Delete field-test raw material", due 30 days later, with no file names.
  A withdrawal opens an issue due in 7 days. The files and the Dropbox request are deleted
  on that schedule.
