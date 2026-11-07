<p align="center">
  <img src="public/favicon.svg" alt="Synodos logo" width="88" />
</p>

<h1 align="center">Synodos</h1>

<p align="center"><strong>Meeting intelligence from a shared audio source.</strong></p>

<p align="center">
  <img alt="Local-first" src="https://img.shields.io/badge/Local--first-AI-EB7100?style=flat-square" />
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="Vue 3" src="https://img.shields.io/badge/Vue-3-42B883?style=flat-square&logo=vuedotjs&logoColor=white" />
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-17-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-333333?style=flat-square" /></a>
</p>

---

Synodos is a local-first AI platform for multi-speaker meetings. It brings real-time speech recognition together with speaker diarization to produce speaker-aware transcripts and contextual meeting intelligence from a single shared audio stream.

Rather than relying on a separate audio channel for every participant, Synodos is designed around the acoustics of a conversation happening in the same room.

## How it works

~~~text
   Shared microphone / room audio
                 |
          16 kHz mono PCM
                 |
              WebSocket
                 |
        +--------+--------+
        |                 |
   Speech recognition   Speaker modeling
   Whisper / Qwen3-ASR  CAM++ / Nemotron 3
        |                 |
        +--------+--------+
                 |
       Speaker-aware transcript
                 |
        Meeting intelligence
                 |
         Review and export
~~~

The recognition and speaker-analysis paths process the same incoming conversation. Independent Nemotron 3 diarization is configurable, while CAM++ provides the default speaker-embedding evidence.

## Under the hood

| Area | Implementation |
| :--- | :--- |
| **Real-time speech** | Streaming WebSocket audio, voice activity detection, timestamped captions, and vocabulary-aware recognition. |
| **Multilingual ASR** | **faster-whisper large-v3** with configurable **Qwen3-ASR 1.7B** hybrid routing. The speech-language interface supports 10 configured languages. |
| **Speaker diarization** | **CAM++** voice embeddings and speaker clustering, with an optional independent **NVIDIA Nemotron 3** stream. **ECAPA-TDNN** is available for speaker-verification comparison. |
| **Meeting intelligence** | Local **Ollama** models with **PostgreSQL** retrieval for evidence-linked answers, meeting summaries, and follow-up information. |
| **Application** | **FastAPI** with Python 3.12, **Vue 3 / TypeScript**, and a local workspace for transcript review, speaker naming, translation, and Markdown/JSON export. |

### Designed for the full conversation

Synodos keeps recognized speech connected to its timing and speaker evidence. Meeting records can be reviewed and corrected after recording, while the Assistant draws on saved conversation context instead of treating every utterance as an isolated message.

Saved meetings, speaker profiles, and local terminology libraries remain under the user's control. AI processing can run against a local model service.

## Getting started

**Requirements:** Python **3.12 recommended** (3.12+ supported), [uv](https://docs.astral.sh/uv/), Node.js **24.12+**, and [pnpm](https://pnpm.io/). An NVIDIA CUDA environment can be used for GPU inference.

Clone the repository:

~~~powershell
git clone https://github.com/hwspen7/Synodos.git
cd Synodos
~~~

**1. Backend** — from the repository root:

~~~powershell
uv sync --python 3.12
uv pip install -r requirements/speech-whisper.txt
uv run uvicorn main:app --host 127.0.0.1 --port 8000
~~~

**2. Frontend** — in a second terminal:

~~~powershell
pnpm install
pnpm dev
~~~

Open the local address shown by Vite. The frontend connects to the FastAPI server through the Vite development proxy. The API health endpoint is available at `http://127.0.0.1:8000/api/health`.

### Model and service configuration

Core speech settings are controlled through environment variables:

| Setting | Purpose |
| :--- | :--- |
| `MEETING_ASR_ENGINE` | Select `whisper` or the multilingual `hybrid` route. |
| `MEETING_ASR_LANGUAGE` | Set a source language or use `auto` detection. |
| `MEETING_INDEPENDENT_DIARIZATION` | Enable the optional `nemotron3` speaker stream. |
| `MEETING_AI_ENABLED` | Enable the local meeting-intelligence service. |
| `MEETING_AI_ENDPOINT` | Connect to an Ollama-compatible local model endpoint. |

The Windows launcher is provided as [`Start-Meeting.cmd`](Start-Meeting.cmd) for an already configured local environment. Optional model backends and PostgreSQL services use their respective local dependencies.

## Repository

| Directory | Responsibility |
| :--- | :--- |
| [`speech/`](speech/) | ASR engines, acoustic preprocessing, diarization, and caption integrity. |
| [`sessions/`](sessions/) | Real-time session processing, speaker channels, and transcription state. |
| [`intelligence/`](intelligence/) | Meeting knowledge, retrieval, evidence verification, and translation. |
| [`api/`](api/) | FastAPI endpoints and WebSocket transport. |
| [`src/`](src/) | Vue interface and meeting workspace. |
| [`tests/`](tests/) | Backend, frontend, speech-quality, and integration checks. |

Local recordings, downloaded models, private evaluation corpora, and generated reports are excluded from Git tracking.

## License

Synodos is released under the [MIT License](LICENSE). Third-party models and datasets remain subject to their respective terms.
