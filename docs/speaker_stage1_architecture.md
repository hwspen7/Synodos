# Stage 1: isolated speaker-model comparison

This phase implements **candidate interfaces and reference-backed experiments**.
It does not silently switch the production diarizer or replace saved speaker
voiceprints. The live baseline remains FunASR CAM++ + the existing clustering
and historical label stability rules.

## Why the models are separate

- **CAM++** produces a speaker embedding. Existing clustering and ASR segment
  boundaries are required to derive speaker turns.
- **ECAPA-TDNN** produces a *different* embedding space for same/different
  speaker verification. It is not a complete diarization pipeline. Existing
  CAM++ voiceprints cannot be compared to ECAPA vectors.
- **Nemotron 3 Diarization** produces anonymous "who spoke when" activity turns,
  including possible overlap. Its speaker channels are not enrolled identities.
- For reproducibility, these models run only when a user explicitly invokes
  the comparison script. They do not load during normal transcription.

Files:
- speech/speaker_backends.py: typed interfaces, audio/vector validation, three
  adapters, and model constructors.
- tools/speaker_stage1_compare.py: human-reference speaker DER and same/different
  speaker verification comparison (EER, AUC, optional fixed-threshold FAR/FRR).
- tools/speaker_stage1_merge.py: combine CAM++ and ECAPA reports produced in
  different Python environments, only when they share a manifest checksum.
- tests/quality/test_speaker_stage1_compare.py: deterministic contracts with
  fake adapters and local WAVs, without model downloads.

## Install optional models without touching the live backend

Use **separate** Python 3.12 environments on Windows. First make sure your
RTX 4070 Ti SUPER (16 GB) CUDA PyTorch installation matches the environment and verify it
with the official PyTorch installation guidance. Do not change the working
live ASR virtual environment. Stop active meetings before running comparisons.

    uv venv .venv-speaker-ecapa --python 3.12
    uv pip install --python .venv-speaker-ecapa\Scripts\python.exe -r requirements-speaker-ecapa.txt
    uv venv .venv-speaker-nemotron --python 3.12
    uv pip install --python .venv-speaker-nemotron\Scripts\python.exe -r requirements-speaker-nemotron.txt

Each environment must also contain a compatible PyTorch build; SpeechBrain
additionally needs a compatible torchaudio build. The requirements are
candidates, not a tested lockfile for this Windows GPU. Test these
dependencies separately rather than force-installing into the current app.

### First-run Transformers processor dependency: librosa

Nemotron's `NemotronAsrStreamingFeatureExtractor` requires `librosa`.
The candidate requirements now include it and the doctor will flag it
before the model tries to download weights. To install the missing
dependency into **only** the E: drive Nemotron environment, use:

    $env:UV_CACHE_DIR = "E:\Meeting-AI-Cache\uv"
    $env:HF_HOME = "E:\Meeting-AI-Cache\huggingface"
    $env:HF_HUB_CACHE = "E:\Meeting-AI-Cache\huggingface\hub"
    $env:TEMP = "E:\Meeting-AI-Cache\temp"
    $env:TMP = "E:\Meeting-AI-Cache\temp"
    New-Item -ItemType Directory -Force $env:UV_CACHE_DIR, $env:HF_HUB_CACHE, $env:TEMP | Out-Null
    uv pip install --python .\.venv-speaker-nemotron\Scripts\python.exe -r requirements-speaker-nemotron.txt

The Hugging Face unauthenticated warning affects rate limits; the Windows
symlink warning affects disk space and cache efficiency. Neither one
explains a missing librosa import. Do not delete the model cache to fix
this issue or install packages into the main backend environment.

Before downloading candidate weights, run the no-download environment check:

    .venv-speaker-ecapa\Scripts\python.exe -m tools.speaker_stage1_doctor --model ecapa --device cuda
    .venv-speaker-nemotron\Scripts\python.exe -m tools.speaker_stage1_doctor --model nemotron3 --device cuda

The doctor checks dependency presence, basic version compatibility, CUDA
build selection and GPU availability. It does not load weights or establish
inference performance.

### Windows: fix "cuda_available: false" in both optional environments

Start with NVIDIA driver visibility and the *actual* PyTorch build. A Python
package named `torch` does not by itself guarantee CUDA support:

    nvidia-smi
    .venv-speaker-ecapa\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    .venv-speaker-nemotron\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

If `torch.version.cuda` prints `None`, the Python environment has a
CPU-only PyTorch build. On the verified RTX 4070 Ti SUPER with NVIDIA driver 610.88,
CUDA 12.8 is supported. Install the known official **CUDA 12.8** PyTorch 2.11.0 wheel
in **only** these optional environments:

    uv pip install --python .venv-speaker-ecapa\Scripts\python.exe "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-speaker-nemotron\Scripts\python.exe "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128

Check `nvidia-smi` first. If the driver does not support CUDA 12.8,
install a driver supported by your GPU or choose the matching officially
published CUDA 12.6 wheel (`+cu126` and `/whl/cu126`).
Do not change the currently functioning live Whisper environment, and
do not attempt model benchmarks until the doctor reports CUDA available.

Official installation table:
https://pytorch.org/get-started/previous-versions/

TorchAudio 2.11 introduced a stable ABI that supports newer PyTorch
releases. Mismatched 2.11.x / 2.15.x numbers alone do not prove a broken
installation. These pinned wheels are a **known CUDA configuration**,
not a claim that older version pairs are inherently incompatible.

Models download their pretrained weights **only on explicit invocation**.
First download needs network access and model license acceptance where
required. Nemotron 3 is currently available from Hugging Face under
the model publisher's license. Review the terms before use.

Model documentation:
- https://huggingface.co/nvidia/Nemotron-3-Diarization
- https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

### Frozen test data and reference requirements

Use the existing gitignored benchmark/fixed195/ directory for real recordings,
speaker enrollment/test pairs and scored outputs. No speech, voiceprints,
reference turn labels or personal names should be committed.

Create a local JSON manifest, for example:

    benchmark/fixed195/speaker_stage1_manifest.json

Its schema is:

    {
      "schema": "meeting-speaker-stage1-manifest-v1",
      "diarization_cases": [
        {
          "id": "fixed195-en-m01",
          "audio": "audio/fixed195-en-m01.wav",
          "sha256": "64-character SHA-256 of the exact WAV",
          "reference_turns": [],
          "camplus_replay": {
            "path": "live_caption_replay.json",
            "sha256": "64-character SHA-256 of the completed baseline JSON"
          }
        }
      ],
      "verification_pairs": [
        {
          "id": "same-speaker-case-01",
          "left": {"audio": "audio/speaker_a_enroll.wav",
                   "sha256": "64-character SHA-256"},
          "right": {"audio": "audio/speaker_a_test.wav",
                    "sha256": "64-character SHA-256"},
          "same_speaker": true
        },
        {
          "id": "different-speaker-case-01",
          "left": {"audio": "audio/speaker_a_enroll.wav",
                   "sha256": "64-character SHA-256"},
          "right": {"audio": "audio/speaker_b_test.wav",
                    "sha256": "64-character SHA-256"},
          "same_speaker": false
        }
      ]
    }

**This is a schema template, not a usable benchmark**. The empty
reference_turns array and placeholder checksums intentionally fail
validation. Replace them with human-verified data. A diarization reference is:

    {"start_ms": 1200, "end_ms": 3140, "speaker": "speaker_a"}

Turns are in source-WAV milliseconds, may overlap, and must be checked by
listening (or independently verified against isolated licensed source tracks).
Use anonymous speaker labels if possible; these labels are never an output
from the model under test. The comparison does not generate gold data.

PowerShell checksums:

    (Get-FileHash benchmark/fixed195/audio/fixed195-en-m01.wav -Algorithm SHA256).Hash.ToLower()
    (Get-FileHash benchmark/fixed195/live_caption_replay.json -Algorithm SHA256).Hash.ToLower()

The CAM++ baseline must be a completed single-loop real-time WebSocket replay
of the **same WAV**. It provides an anonymous final timeline under
final_cluster_diagnostics.anonymous_timeline. Its segment boundaries follow
the live ASR/cluster pipeline. The current replay tool also records
source_audio_sha256, and the comparison checks that digest against the
manifest audio checksum. Existing replay JSON files **without** that field
are intentionally rejected. Re-run the same audio once after pulling the
latest code; do not fabricate or paste the hash manually.

    python -m tools.live_caption_ws_replay --audio benchmark/fixed195/audio/fixed195-en-m01.wav --loops 1 --speed 1.0 --output benchmark/fixed195/stage1_live_baseline_en_m01.json

Then update camplus_replay.path and camplus_replay.sha256 in your *local*
manifest using the newly generated report and its Get-FileHash value.
An aggregate report from a different WAV or a truncated timeline is rejected.

For voiceprints, use several consented samples per speaker, recorded on
different days and microphones, plus different-speaker pairs. Include both
male/female and accented/noisy variations if ethically collected. The test
should be split by *person or recording session*, not by reusing near-identical
windows of the same utterance on both sides.

## Run the comparisons

**1. Diarization: CAM++ live baseline versus Nemotron 3 offline**

The baseline report must have the new source_audio_sha256 field; see above.

    .venv-speaker-nemotron\Scripts\python.exe -m tools.speaker_stage1_compare --mode diarization --root benchmark/fixed195 --manifest benchmark/fixed195/speaker_stage1_manifest.json --diar-device cuda --output benchmark/fixed195/stage1_diarization.json

The Nemotron adapter uses the official Transformers
AutoModelForAudioFrameClassification offline API. Its 16 kHz mono WAV input
is currently capped at 180 seconds for bounded, per-clip comparison. For
longer recordings use separate frozen clips; do not pretend these constitute
one continuous streaming inference. If the dependency is missing, the
candidate receives an explicit failure, and the CLI exits nonzero.

**2. CAM++ speaker verification in the existing backend environment**

    python -m tools.speaker_stage1_compare --mode verification --verification-backend camplus --root benchmark/fixed195 --manifest benchmark/fixed195/speaker_stage1_manifest.json --output benchmark/fixed195/stage1_camplus_verification.json

**3. ECAPA speaker verification in the isolated environment**

    .venv-speaker-ecapa\Scripts\python.exe -m tools.speaker_stage1_compare --mode verification --verification-backend ecapa --ecapa-device cuda --root benchmark/fixed195 --manifest benchmark/fixed195/speaker_stage1_manifest.json --output benchmark/fixed195/stage1_ecapa_verification.json

To measure operational FAR/FRR, set predeclared thresholds with
--camplus-threshold or --ecapa-threshold. Do not tune thresholds on the
test set. EER and AUC are descriptive discrimination scores derived from
the pairs; they cannot establish an operational identity threshold.

**4. Merge same-manifest verification reports**

    python -m tools.speaker_stage1_merge --camplus-report benchmark/fixed195/stage1_camplus_verification.json --ecapa-report benchmark/fixed195/stage1_ecapa_verification.json --output benchmark/fixed195/stage1_paired_verification.json

The merge fails if either output comes from a different manifest, contains
the wrong backend, or has incomplete/no mixed-label test coverage. Use
unique filenames for new runs; outputs are exclusive-create.

## Automated regression

    python -m unittest tests.quality.test_speaker_stage1_compare -v
    python -m unittest discover -s tests -t . -v

No GPU is needed for these software contracts. CI must not claim real
model performance from deterministic fixtures.

## Acceptance gate before any production switch

1. A complete frozen, independently checked dataset with at least two,
   preferably four or more, speakers per conversational case; include A-B-A
   recurrence, long gaps, interruptions and genuine overlapping speech.
2. Per-case DER, overlapping-speaker recall, speaker-count error and
   boundary quality from both production and Nemotron; comparison is
   **not** perfectly apples-to-apples, because the baseline is ASR-boundary
   constrained and the candidate is acoustic.
3. Verification EER/AUC and FAR/FRR at thresholds fixed using a disjoint
   development set, with checks for false acceptance of unknown people.
4. Cold/warm inference time, process RSS, peak GPU VRAM, and 60-minute
   stability, with *no* concurrent heavy candidate inference in live ASR.
5. No voiceprint vectors from different models stored in the same identity
   space. A future promotion requires model-tagged embedding records, safe
   re-enrollment/migration, and explicit consent.
6. No inferred change to historical speaker IDs or user-confirmed text
   without a separate review step.

Passing CI demonstrates only adapter and benchmark plumbing. It does not
mean the NVIDIA or SpeechBrain weights were installed or that either model
improves real-world speaker accuracy. The live defaults remain untouched.

## First real verification comparison on Fixed195 isolated speaker channels

The public Fixed195 dialogue collector stores two speaker-isolated audio
channels alongside each mixed WAV in
benchmark/fixed195/public_multi_unscored.jsonl. These are useful for a
**same-recording diagnostic** even though the mixtures are explicitly
`is_scoreable_diarization=false` and have no human timestamp references.
Do not make or claim a DER result from those unscored mixtures.

First ensure the private source file exists locally (it is gitignored):

    Test-Path benchmark/fixed195/public_multi_unscored.jsonl

Then generate checksum-verified, nonoverlapping 5-second samples for each
channel. The tool checks both stored stem SHA-256 hashes and audio geometry,
selects two suitably active windows per channel, and produces two same-
speaker and two different-speaker pairs per recording. No transcript text,
model prediction, or speaker identity is imported or committed.

    python -m tools.speaker_stage1_pairs --root benchmark/fixed195 --output benchmark/fixed195/speaker_stage1_manifest.json

To inspect just the English test case first, use a different output path,
and the optional `--case-id fixed195-en-m01`. Generated clips are private
and written to benchmark/fixed195/speaker_stage1_samples/. All filenames
are exclusive-create, so the tool refuses to overwrite an earlier trial.

The selection uses signal energy, **not speech classification**. Listen
to the selected snippets to ensure each contains one individual and no
dominant cross-channel bleed, music, or other interference. Skip the test
if this cannot be confirmed. The same recording provides same-person
samples in identical acoustic conditions, which can make apparent EER
unrealistically good. It cannot establish cross-session identification.

Run the two embedding models **sequentially**, with their separate Python
environments and the same unchanged manifest:

    python -m tools.speaker_stage1_compare --mode verification --verification-backend camplus --root benchmark/fixed195 --manifest benchmark/fixed195/speaker_stage1_manifest.json --output benchmark/fixed195/stage1_camplus_diag.json

    .venv-speaker-ecapa\Scripts\python.exe -m tools.speaker_stage1_compare --mode verification --verification-backend ecapa --ecapa-device cuda --root benchmark/fixed195 --manifest benchmark/fixed195/speaker_stage1_manifest.json --output benchmark/fixed195/stage1_ecapa_diag.json

    python -m tools.speaker_stage1_merge --camplus-report benchmark/fixed195/stage1_camplus_diag.json --ecapa-report benchmark/fixed195/stage1_ecapa_diag.json --output benchmark/fixed195/stage1_paired_diag.json

Results include EER and AUC if both positive and negative trials succeeded.
Without predeclared thresholds, FAR and FRR remain null. The merged report
retains `verification_provenance.cross_session_identity_evaluable=false`,
and model promotion remains disabled.

For real speaker-memory acceptance, replace the diagnostic corpus with
consented enrollment and test material from **different recording
sessions**, with varied microphones, languages, levels of noise and
unknown-speaker rejection. Human-confirmed identity labels are mandatory.
A separate development split is needed before selecting an operating
threshold, and test speakers must not leak into threshold calibration.
