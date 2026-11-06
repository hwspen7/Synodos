# Optional offline speaker model, isolated from the live ASR environment

The optional second-opinion speaker model is pyannote Community-1. It requires
user approval of the Hugging Face model terms and a Hugging Face access token.
It is not bundled or downloaded by Meeting of Pioneers.

This module is **not necessary for normal recording**. It is used only when
the operator explicitly requests a review of an existing WAV recording after
stopping the microphone.

## Dependency isolation (Windows)

Create a separate Python environment on a drive with sufficient storage.
This is deliberately distinct from meeting\.venv. Exact supported pyannote
and PyTorch versions must be validated against your machine and Python release;
do not force-install or downgrade torch inside meeting\.venv.

Example, from the Meeting project root:

~~~powershell
py -3.11 -m venv E:\AI\venvs\meeting-speaker
E:\AI\venvs\meeting-speaker\Scripts\python.exe -m pip install --upgrade pip
E:\AI\venvs\meeting-speaker\Scripts\python.exe -m pip install pyannote.audio
~~~

Read and accept the model conditions:
https://huggingface.co/pyannote/speaker-diarization-community-1

Set a Hugging Face read token as an environment variable in the process that
starts Meeting. Do not paste tokens into logs, issue trackers or chat messages.

~~~powershell
$env:HF_TOKEN = (Read-Host "HF read token")
$env:MEETING_SPEAKER_REFINER_PYTHON = "E:\AI\venvs\meeting-speaker\Scripts\python.exe"
~~~

Launch Meeting from this same PowerShell session. The intelligence worker
runs the second acoustic model as a child of the selected Python interpreter.
The live Whisper environment and its CUDA dependencies remain unchanged.
CPU is the default to avoid competing with Whisper and Saluki; set
MEETING_OFFLINE_DIARIZATION_DEVICE=cuda only after validating GPU memory.

The separate interpreter must be able to import the meeting project directory
from its working directory. The process exits after each review, releasing
its memory. Cached models are reused from Hugging Face's local cache.

## API

Submit an intelligence job when recording is stopped:

~~~json
{
  "kind": "speaker_refine",
  "meeting_id": "YOUR_MEETING_ID",
  "payload": {
    "filename": "THE_32_DIGIT_HEX_WAV_FILENAME.wav"
  }
}
~~~

POST this to /api/intelligence/jobs. Poll GET /api/intelligence/jobs/JOB_ID.

The output is a speaker timeline relative to the start of **that recording
file**, not the accumulated meeting transcript timeline. A meeting resumed
after previous recordings may have different absolute sample offsets. It
would be unsafe to auto-apply the anonymous labels to previously assigned
participants, so the system returns review-only acoustic evidence.

No speaker names or original transcript text are changed. If the model
cannot load, the job fails clearly without altering the primary recording.

## Accuracy acceptance

Compare model output with manually annotated recording-specific speaker
turns using tools/validation/dialogue_speaker_benchmark.py, including short answers,
speaker recurrence, and overlap. Compare this against existing CAM++ results
on the same WAVs before making either model the default.
