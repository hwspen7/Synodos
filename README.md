<p align="center">
  <img src="public/favicon.svg" alt="Synodos" width="96" />
</p>

<h1 align="center">Synodos</h1>

<p align="center"><strong>Meeting intelligence, built around the conversation.</strong></p>

<p align="center">
  A local-first AI meeting platform for understanding multi-speaker conversations from a shared audio source.
</p>

<p align="center">
  <a href="#why-synodos">Overview</a> ·
  <a href="#capabilities">Capabilities</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#model-stack">Models</a> ·
  <a href="#getting-started">Get started</a> ·
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img alt="Local first" src="https://img.shields.io/badge/Local--first-AI-EB7100?style=flat-square" />
  <img alt="Shared audio" src="https://img.shields.io/badge/Audio-Shared%20source-BC6C25?style=flat-square" />
  <img alt="10 speech languages" src="https://img.shields.io/badge/Languages-10-D97B23?style=flat-square" />
  <img alt="Whisper large-v3" src="https://img.shields.io/badge/Whisper-large--v3-5E5AA8?style=flat-square" />
  <img alt="faster-whisper" src="https://img.shields.io/badge/faster--whisper-ASR-5557AD?style=flat-square" />
  <img alt="Qwen3-ASR 1.7B" src="https://img.shields.io/badge/Qwen3--ASR-1.7B-6B52AC?style=flat-square" />
  <img alt="FunASR" src="https://img.shields.io/badge/FunASR-Speech-9959B6?style=flat-square" />
  <img alt="CAM++" src="https://img.shields.io/badge/CAM%2B%2B-Voiceprints-B85B79?style=flat-square" />
  <img alt="Nemotron 3" src="https://img.shields.io/badge/Nemotron-3-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img alt="ECAPA-TDNN" src="https://img.shields.io/badge/ECAPA--TDNN-Speaker%20verification-8D64A4?style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-GPU-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-Local%20LLMs-202020?style=flat-square" />
  <img alt="Saluki 27B" src="https://img.shields.io/badge/Saluki-27B-6B4D3C?style=flat-square" />
  <img alt="Argos Translate" src="https://img.shields.io/badge/Argos-Translate-317B95?style=flat-square" />
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="WebSocket" src="https://img.shields.io/badge/WebSocket-Realtime-3F697B?style=flat-square" />
  <img alt="Vue 3" src="https://img.shields.io/badge/Vue-3-42B883?style=flat-square&logo=vuedotjs&logoColor=white" />
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
  <img alt="Vite" src="https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white" />
  <img alt="PostgreSQL 17" src="https://img.shields.io/badge/PostgreSQL-17-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
  <img alt="uv" src="https://img.shields.io/badge/uv-Python%20packages-6655AA?style=flat-square" />
  <img alt="pnpm" src="https://img.shields.io/badge/pnpm-Package%20manager-F69220?style=flat-square&logo=pnpm&logoColor=white" />
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-444444?style=flat-square" /></a>
</p>

---

## Why Synodos

Most meeting transcription pipelines work best when each participant has a separate microphone or audio channel. In a room, however, the conversation often arrives as **one mixed audio stream**. The system must recognize the words while determining which voice they belong to.

**Synodos is designed for this shared-audio setting.** It brings streaming multilingual speech recognition together with acoustic speaker evidence, then connects the resulting transcript to meeting-level understanding. The emphasis is on preserving the structure of the conversation, not simply producing a block of text.

From a live discussion to a saved meeting record, Synodos keeps the speech, speaker attribution, corrections, and AI-generated findings connected.

## Capabilities

| | Capability | What it enables |
| :---: | --- | --- |
| 🎙️ | **Live meeting transcription** | Stream microphone audio to the recognition service and follow timestamped speech as the meeting unfolds. |
| 👥 | **Speaker-aware conversations** | Organize utterances into speaker turns using acoustic embeddings, clustering, and optional independent diarization. |
| 🌐 | **Multilingual recognition** | Detect spoken language or select a supported language, with a configurable Whisper–Qwen recognition pipeline. |
| 🪪 | **Speaker profiles** | Enroll voice samples, associate recognized speakers with local profiles, and revise speaker labels during review. |
| 🧩 | **Transcript review** | Edit saved text and speaker attribution while maintaining revision-aware meeting records and supporting evidence. |
| 🧠 | **Meeting intelligence** | Generate contextual summaries and extract decisions, follow-ups, or questions from saved conversations. |
| 🔎 | **Evidence-linked Q&A** | Search meeting content through PostgreSQL-backed retrieval and return answers tied to source excerpts. |
| 💬 | **Translation** | Translate live or saved captions using installed offline language pairs or a configured local model. |
| 📚 | **Terminology libraries** | Supply domain-specific names and vocabulary as speech-recognition hints. |
| 💾 | **Local meeting workspace** | Save, rename, resume, inspect, and export meetings in Markdown or JSON. Optional WAV recording preserves source audio. |

### One conversation, multiple voices

Synodos processes a shared microphone stream rather than treating each person as a separate input channel. Its speech and speaker paths work from the same acoustic timeline. CAM++ embeddings help maintain speaker consistency, while an optional Nemotron 3 stream provides independent speaker-turn evidence. Confirmed text and speaker information are reconciled into a readable, time-aligned transcript.

### A transcript that retains context

Meeting records hold more than plain text. They retain timestamps, speaker associations, revision information, and context for subsequent review. This makes it possible to correct a transcript without losing the relationship between what was said and the conversation it came from.

### Intelligence grounded in the meeting

The local AI worker analyzes saved conversations for summaries, decisions, action items, and speaker-specific contributions. Retrieval can combine lexical matching with optional embeddings, while evidence-aware answers are linked to saved source excerpts. Meetings become an accessible knowledge base rather than isolated recordings.

## Architecture

The audio pipeline separates **what was said** from **who said it**, then brings both streams together before creating the durable meeting record.

~~~mermaid
flowchart TD
    A["Shared microphone<br/>16 kHz mono PCM"] --> B["WebSocket ingestion<br/>audio buffering and VAD"]
    B --> C["Multilingual ASR<br/>Whisper large-v3 / Qwen3-ASR"]
    B --> D["Speaker evidence<br/>CAM++ / optional Nemotron 3"]
    C --> E["Timed speech segments"]
    D --> F["Speaker turns and embeddings"]
    E --> G["Caption confirmation<br/>speaker-aware alignment"]
    F --> G
    G --> H["Saved meeting and review"]
    H --> I["PostgreSQL knowledge store"]
    I --> J["Local Ollama worker"]
    J --> K["Summaries · Decisions · Evidence-linked Q&A"]
    H --> L["Transcript · Translation · Export"]
~~~

**Design principles**

- **Audio-first speaker evidence.** Speaker activity is derived from acoustic information, not inferred solely from recognized sentences.
- **Content-preserving transcription.** Voice activity detection, bounded recognition windows, commit boundaries, and duplicate guards support stable caption delivery.
- **Speaker continuity.** The session processor maintains speaker state across incoming windows, combining automatic evidence with reviewed labels.
- **Independent workloads.** Recognition scheduling and a separate meeting-intelligence worker keep conversational analysis distinct from time-sensitive audio processing.
- **Traceable interpretation.** Saved transcript revisions and source-linked evidence connect AI findings back to the underlying meeting.

## Model stack

Synodos uses specialized models for distinct stages of processing. Model selection is configurable, and optional components can be enabled for comparison or independent analysis.

| Technology | Role in Synodos |
| --- | --- |
| **Whisper large-v3** via **faster-whisper** | Primary multilingual speech-to-text engine with timestamped recognition. |
| **Qwen3-ASR 1.7B** | Optional second ASR backend used by the language-routed hybrid recognition engine. |
| **FunASR** | Speech processing components including voice activity detection, punctuation, and speaker modeling. |
| **CAM++** | Default speaker embeddings for clustering and local voiceprint matching. |
| **NVIDIA Nemotron 3 Diarization** | Configurable independent acoustic speaker-turn analysis. |
| **ECAPA-TDNN / SpeechBrain** | Additional speaker-verification model for comparison and background evaluation. |
| **Ollama** | Local language-model serving for summaries, structured analysis, translation, and meeting questions. |
| **Saluki 27B** | Language-model profile configured in the Windows launcher for meeting intelligence. |
| **Argos Translate** | Optional on-device language-pair translation for captions. |

The application supports automatic spoken-language selection and the following **10 configured languages**:

English · Chinese · French · Spanish · Arabic · Indonesian · Japanese · Russian · German · Hindi

The optional hybrid ASR route uses a separately configured Qwen worker. Translation runs through available installed language pairs or the selected local text model.

## From a meeting to usable knowledge

<details>
<summary><strong>1. Capture and recognize</strong></summary>

A browser microphone streams 16 kHz PCM over WebSocket. The session processor manages audio windows, speech activity, recognition scheduling, and the transition from provisional to confirmed captions. Selected terminology libraries can provide ASR vocabulary hints.

</details>

<details>
<summary><strong>2. Resolve speakers and review the transcript</strong></summary>

Speaker embeddings and acoustic turn evidence are matched against the current conversation. Local speaker profiles can be created or updated, and meeting-specific names may be assigned during review. Saved captions support text corrections and speaker-attribution revisions.

</details>

<details>
<summary><strong>3. Analyze the conversation</strong></summary>

With local intelligence enabled, the worker processes saved transcript context into meeting summaries and structured findings. Analysis modes include meetings, research discussions, classrooms, interviews, standups, and sales conversations. Speaker-scoped insights focus on the selected person's attributed turns.

</details>

<details>
<summary><strong>4. Search, translate, and export</strong></summary>

Meeting chunks can be indexed for lexical retrieval, with optional vector embeddings. Evidence-linked answers reference stored excerpts. Captions can be translated, reviewed, and exported as Markdown or JSON.

</details>

## Getting started

### Requirements

- **Python 3.12** recommended (3.12 or newer required), with [uv](https://docs.astral.sh/uv/)
- **Node.js 24.12+** with [pnpm](https://pnpm.io/)
- An audio input device for live recording
- Optional NVIDIA GPU and CUDA environment for accelerated model inference

### 1. Clone

~~~powershell
git clone https://github.com/hwspen7/Synodos.git
cd Synodos
~~~

### 2. Start the speech API

Run from the repository root:

~~~powershell
uv sync --python 3.12
uv pip install -r requirements/speech-whisper.txt
uv run uvicorn main:app --host 127.0.0.1 --port 8000
~~~

The API exposes its health status at [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

### 3. Start the workspace

In another terminal:

~~~powershell
pnpm install
pnpm dev
~~~

Open the local URL displayed by Vite, allow microphone access, and create a meeting.

### Windows launcher

[`Start-Meeting.cmd`](Start-Meeting.cmd) starts the configured local Windows meeting stack. The launcher supports the repository's installed Python environment, PostgreSQL service, Ollama instance, frontend, and model-specific runtime settings. Review [`tools/windows/Start-Meeting.ps1`](tools/windows/Start-Meeting.ps1) when adapting its local paths and services to a different machine.

## Configuration

Speech, speaker analysis, and intelligence services are controlled by environment variables.

| Variable | Description |
| --- | --- |
| `MEETING_ASR_ENGINE` | `whisper` for the primary speech engine, or `hybrid` for the Whisper/Qwen route. |
| `MEETING_ASR_MODEL` | Recognition model identifier, such as `large-v3`. |
| `MEETING_ASR_LANGUAGE` | Input language or `auto`. |
| `MEETING_ASR_DEVICE` | Speech-model inference device, such as `cuda`. |
| `MEETING_HYBRID_QWEN_PYTHON` | Executable for the separately configured Qwen ASR worker. |
| `MEETING_INDEPENDENT_DIARIZATION` | `off` or `nemotron3` for independent diarization. |
| `MEETING_RECORD_AUDIO` | Enable source-audio recording for meetings. |
| `MEETING_ASR_ADAPTIVE_AUDIO` | Enable optional adaptive processing for live ASR input. |
| `MEETING_STORAGE_BACKEND` | `legacy` for local meeting files, or `postgres` for PostgreSQL. |
| `DATABASE_URL` | PostgreSQL connection URL when that backend is selected. |
| `MEETING_AI_ENABLED` | Enable the local meeting-intelligence service. |
| `MEETING_AI_ENDPOINT` | Address of the configured Ollama or compatible local model server. |
| `MEETING_AI_TEXT_MODEL` | Model selected for language-based meeting analysis. |
| `MEETING_AI_EMBEDDINGS` | Enable vector embeddings for the retrieval pipeline. |

### Local meeting intelligence with PostgreSQL

Meeting summaries, structured action tracking, and evidence-based questions use the PostgreSQL-backed intelligence store and its separate worker. The following example uses an existing PostgreSQL database and a locally available Ollama model.

~~~powershell
uv sync --python 3.12 --extra postgres
uv pip install -r requirements/intelligence.txt

$env:MEETING_STORAGE_BACKEND = "postgres"
$env:DATABASE_URL = "postgresql://USER:PASSWORD@127.0.0.1:5432/meeting"
$env:MEETING_AI_ENABLED = "1"
$env:MEETING_AI_PROVIDER = "ollama"
$env:MEETING_AI_ENDPOINT = "http://127.0.0.1:11434"
$env:MEETING_AI_TEXT_MODEL = "your-local-model"
~~~

Start the FastAPI service with these settings, then run the worker in a separate terminal using the same configuration:

~~~powershell
uv run python -m intelligence.worker
~~~

PostgreSQL schema migrations are applied through the database connection layer. Optional speech and speaker models have their own dependency manifests in [`requirements/`](requirements/).

## Repository structure

~~~text
Synodos/
├── api/                 HTTP routes and WebSocket transport
├── core/                Runtime settings, infrastructure, and observability
├── speech/
│   ├── asr/             Whisper, Qwen, and hybrid recognition
│   ├── audio/           Audio windows, VAD, and acoustic processing
│   ├── diarization/     Speaker embeddings and acoustic turn analysis
│   └── captions/        Transcript confirmation, alignment, and translation
├── sessions/            Meeting state, speaker continuity, and scheduling
├── intelligence/        Local model gateway, retrieval, and AI worker
├── storage/             Meeting persistence and PostgreSQL migrations
├── src/                 Vue and TypeScript workspace
├── tools/               Diagnostics, validation, and Windows launch scripts
├── tests/               Functional, contract, and regression tests
├── public/              Synodos visual assets
├── Start-Meeting.cmd    Windows entry point
└── README.md
~~~

## Testing and quality

The repository includes tests for the audio processing lifecycle, caption integrity, speaker state, meeting review, multilingual recognition contracts, local data protection, and database-backed intelligence.

~~~powershell
uv run python -m unittest discover -s tests -t .
pnpm type-check
pnpm build-only
~~~

GitHub Actions additionally exercises frontend integration, backend regressions, PostgreSQL storage, speech-model contracts, and release-safety checks. Private acoustic evaluation corpora and local model weights are maintained outside the repository.

## Local-first data handling

Synodos is designed to keep meeting records and model execution under the operator's control. Runtime storage can use local files or a configured PostgreSQL instance. Ollama can serve language models through a loopback endpoint, while the application separates live ASR workloads from queued meeting analysis.

The repository excludes local recordings, voiceprint stores, downloaded models, private benchmark datasets, generated evaluation results, and environment secrets through ignore rules and repository checks. Third-party models are downloaded or installed independently of the source repository.

## License

Synodos is open source under the [MIT License](LICENSE). External speech models, language models, pretrained weights, and datasets retain their respective licenses.

<p align="center"><sub>Built for conversations that share a room, not a microphone per person.</sub></p>
