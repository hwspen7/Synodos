# Phase 1 end-of-release evidence: AMI human words and GitHub Actions

Phase 1's 10-language, 110-clip, single-speaker ASR benchmark is accepted.
This document covers two separate remaining acceptance claims. One is the
real TS3003b mixed-microphone **final caption** comparison with verified
human words. The other is execution of real hosted GitHub CI. Neither is
certified by a local unit test or the fact that the tool exists.

## Official AMI human reference and mixed-audio limitation

The University of Edinburgh AMI corpus publishes human-created,
time-aligned orthographic annotations, in `ami_public_manual_1.6.2.zip`
(CC BY 4.0), at:

https://groups.inf.ed.ac.uk/ami/download/

The frozen speaker-activity RTTM previously used by this project is NOT
an orthographic transcript and cannot establish CER or WER.

`tools.ami_human_text_reference` takes the published XML `<w>` elements
from `TS3003b.<speaker>.words.xml` covering 1500-1740 seconds, sorts
the annotated words in start-time order, and writes a plain-text reference
and a SHA256 source/metadata audit under gitignored `benchmark/fixed195`.
It checks the duration, multiple speaker files, finite word times,
source archive boundaries and official source fingerprint. It never
invents a text reference from ASR predictions.

**Important:** AMI manual transcript annotations include words from
multiple simultaneously speaking people. Some may be clearly audible
in individual headset channels yet not recoverable from the test
headset *mix*. Chronologically merging both speakers into one text
sequence makes ordinary global CER/WER a conservative diagnostic with
overlap-order artifacts. A low CER or WER is not a substitute for
listening and reviewing clear single-speaker regions. Do not mark
`-HumanReferenceVerified` solely because the download has a human
annotation label. Inspect the actual mix and resolve ambiguous turns.

## Collect both evidence streams in one invocation

On the Windows PC with the original 240-second TS3003b audio,
Nemotron model weights and Python 3.12 venv:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin fix/ts3003b-integrity-forensics-20261104
git merge --ff-only FETCH_HEAD

$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_phase1_closure.ps1 -PythonExe $PY
```

The runner attempts to download the official AMI manual zip directly
into `benchmark/fixed195/ami_shadow10/` without modifying the C:
drive. If the university server is unavailable, obtain the 22 MB
`ami_public_manual_1.6.2.zip` through the official download page
and save it to that exact private folder before re-running.

The runner automatically selects the newest complete local
`content_review_singles_*.json`, excluding summary and triage files.
It revalidates the 110-clip ten-language gate without GPU inference.

The TS3003b live replay uses `-ContentFirst` and now exports both
the privately held final transcript and a segment-timing audit.
The source words, predicted words, reference SHA256, replay JSON,
audio counts, change/disappearance counts and separate speaker DER
statistics are available for manual comparison on the E drive.
The strict only_words DER failure remains a **failure**, even if
the separately evaluated forced-aligned diagnostic looks better.

After a reviewer independently checks the manual transcript against
the actual mixed audio, selectively excluding or documenting words
not intelligible in that mix and recording the review, re-run with
`-HumanReferenceVerified` to enable the strict human text gate.
It does not bypass the original DER acceptance protocol.

Do not upload raw meeting transcripts or audio outside ignored local
directories. Reports in the GitHub PR must contain aggregate scores
and SHA256 only.

## GitHub Actions runner dispatch blocker

`tools.diagnose_github_actions.ps1` fetches job check-run annotations
using authenticated GitHub CLI (`gh auth login`) or the `GH_TOKEN`
or `GITHUB_TOKEN` process variable. It never writes credentials.
The repository `hwspen7/meeting` is private, so anonymous requests
cannot access its check-run annotations.

Usage:

```powershell
cd E:\Meeting-Fixed10-Test
gh auth status
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\diagnose_github_actions.ps1
```

All five observed jobs in project/operations/Windows workflows were
failing before allocating a runner or executing any workflow steps,
and normal job logs were unavailable. GitHub check-run annotations
are needed to distinguish account billing/budget restrictions from
another scheduler issue. If an annotation cites failed payments or
the spending limit, check the personal owner account:

https://github.com/settings/billing

Open **Billing & Licensing > Budgets and alerts** and inspect Actions
spend, all overlapping hard budgets and payment status. GitHub can
block hosted runners before a single step starts when a budget is
exhausted. The tool cannot change billing, account entitlements or
runner provisioning. If billing is correct but the annotation persists,
contact GitHub Support. Do not bypass branch protection or forge
passing checks.

After account-level dispatch is restored, use the GitHub Actions
**Re-run failed jobs** option for the branch/PR checks. Project,
Operations and Windows workflows also include future manual dispatch
configuration; availability of workflow_dispatch depends on GitHub's
default-branch rules.

## Stop/go release criteria

Phase 1 may be merged and tagged only when all are independently true:

1. Selected 110-case human-reference CER/deletion gate succeeds.
2. Final live-caption, audio count and confirmed-text-churn gates are
   measured and pass on reviewed mixed audio, with clear speech
   not missing, duplicated or invented.
3. The original strict speaker diarization gate and explicitly agreed
   product-level speaker criteria pass, or exceptions are documented
   and approved as a separate release decision (never relabel results).
4. GitHub Actions actually assign runners and run/complete required
   project, database, Windows and operations checks successfully.
5. Source commit, case fingerprints, human reference SHA and private
   evidence reports are frozen. No raw private transcript is committed.

`tools.run_phase1_closure.ps1` fails closed on any unresolved gate.
It intentionally never merges or tags a PR merely because a script
exists. Keep PR #5 as draft until these conditions are demonstrated.
