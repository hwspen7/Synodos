<p align="center">
  <img src="public/favicon.svg" alt="Synodos" width="96" />
</p>

<h1 align="center">Synodos</h1>

<p align="center"><strong>Intelligence for multi-speaker conversations captured through a shared microphone.</strong></p>

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

## The engineering problem

A real meeting is a continuous acoustic exchange, not a sequence of isolated audio channels. When everyone shares one microphone, the recording mixes voices, short turns, pauses and interruptions into **a single timeline**. Transcribing the words does not, by itself, establish who spoke or what the conversation means.

Synodos is designed around this constraint. Speech recognition determines **what was said**; a parallel acoustic path develops evidence for **who said it**. The system reconciles both against a common clock before committing an editable meeting record. A separate knowledge layer then builds structured, source-linked understanding from that record.

| Engineering concern | Technical mechanism | Purpose |
| --- | --- | --- |
| **Continuous audio** | 16 kHz PCM, bounded buffers, absolute sample offsets | Maintain a shared time reference for every inference path |
| **Multilingual content** | Whisper large-v3 with configurable Qwen3-ASR routing | Recover language-aware utterances from speech windows |
| **Multiple speakers** | CAM++ embeddings, session clustering, optional Nemotron 3 | Distinguish recurring voices without independent microphones |
| **Transcript integrity** | Content-first projection, overlap handling, revisioned review | Preserve the original words while resolving attribution |
| **Meeting understanding** | Local LLM worker, PostgreSQL retrieval, evidence checks | Transform saved speech into traceable knowledge |

The important architectural decision is that **the text hypothesis, speaker hypothesis and human-reviewed meeting record are different kinds of evidence**. They interact through explicit timing and revision contracts.

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

## Technical approach

### 01 · Acoustic acquisition and scheduling

The browser streams **16 kHz mono PCM** over WebSocket. A thread-safe rolling buffer retains absolute sample offsets as confirmed audio is reclaimed. A bounded inference scheduler prioritizes recognition requests and enforces queue capacity, while an independent outbound queue sends versioned transcript snapshots to the workspace.

The speech path operates on bounded windows. Optional adaptive conditioning can improve the ASR input while the source PCM remains available to voiceprint and independent acoustic analysis. Acoustic coverage and bounded retry mechanisms keep recovery tied to actual speech evidence.

**Implementation:** [`sessions/audio_buffer.py`](sessions/audio_buffer.py), [`inference/scheduler.py`](inference/scheduler.py), [`speech/audio/`](speech/audio/), [`sessions/outbox.py`](sessions/outbox.py).

### 02 · Language-aware recognition and caption commits

**Whisper large-v3** is the primary recognizer, using `faster-whisper` for local execution. A configurable **Qwen3-ASR 1.7B** worker contributes a second decoding route through language selection and bounded fallback. The hybrid policy validates speech support, timing and transcript coverage when evaluating candidates.

Recognition candidates are not immediately permanent. The caption layer separates provisional text from confirmed segments, applies temporal deduplication and commit-boundary rules, and revisits VAD-supported speech gaps without inventing unsupported text.

**Implementation:** [`speech/asr/`](speech/asr/), [`speech/captions/`](speech/captions/).

### 03 · Independent multi-speaker inference

The speaker path analyzes the same acoustic source without requiring the speech recognizer to identify people from words. **CAM++** embeddings provide speaker similarity and enrolled-profile evidence. With independent diarization enabled, **NVIDIA Nemotron 3** supplies streaming speaker activity, including the temporal relationship between overlapping voices.

A raw diarization channel is not automatically a permanent person identity. Session-level corroboration considers recurring acoustic observations, embedding consistency, activity margins and established speaker references. Provisional assignments and manual review provide distinct states for the evolving meeting record.

**Implementation:** [`speech/diarization/`](speech/diarization/), [`sessions/channels/`](sessions/channels/), [`sessions/identity/`](sessions/identity/).

### 04 · Content-first speaker reconciliation

ASR and diarization have different segmentation boundaries. Synodos maps both to a common timeline and projects speaker evidence **onto confirmed text**, rather than allowing a speaker model to rewrite the transcript. Text conservation checks protect the original utterances. Overlapping acoustic activity is recorded without duplicating the same recognized words across speakers.

Saved captions support revision-aware text and identity edits. The system retains original ASR content, source timestamps and review provenance, so downstream analysis can be associated with the proper meeting revision.

**Implementation:** [`speech/captions/content_first_projection.py`](speech/captions/content_first_projection.py), [`speech/captions/turn_integrity.py`](speech/captions/turn_integrity.py), [`sessions/manual_review.py`](sessions/manual_review.py).

### 05 · Asynchronous meeting intelligence

Live speech processing and language-model analysis have different latency characteristics. Synodos uses a standalone, job-driven AI worker to analyze persisted meetings without sharing the live ASR scheduling queue. Workloads can be deferred during active recording.

The PostgreSQL-backed intelligence store holds jobs, structured findings, knowledge chunks and source revisions. Retrieval can combine multilingual lexical matching with optional embeddings. Evidence-focused answers are checked against actual stored excerpts before being presented.

**Implementation:** [`intelligence/worker.py`](intelligence/worker.py), [`intelligence/store.py`](intelligence/store.py), [`intelligence/retrieval.py`](intelligence/retrieval.py), [`intelligence/evidence_qa.py`](intelligence/evidence_qa.py).

## Architecture

Synodos has three distinct but connected inference paths: **linguistic recognition**, **acoustic speaker reasoning** and **meeting knowledge processing**. Explicit timing, identity and revision boundaries connect the paths without conflating their outputs.

### End-to-end system topology

~~~mermaid
flowchart LR
  subgraph CAPTURE["01 / Capture"]
    direction TB
    MIC["Shared microphone"]
    WEB["Vue workspace"]
    PCM["16 kHz mono PCM"]
    MIC --> PCM
  end

  subgraph TRANSPORT["02 / Transport + session state"]
    direction TB
    WS["FastAPI WebSocket"]
    SESSION["Session lifecycle"]
    BUF["Bounded rolling buffer"]
    CLOCK["Absolute acoustic timeline"]
    OUT["Versioned outbox"]
    WS --> SESSION --> BUF --> CLOCK
    SESSION --> OUT
  end

  subgraph ASR["03 / Linguistic inference"]
    direction TB
    WINDOW["VAD + bounded windows"]
    FILTER["Optional adaptive signal conditioning"]
    QUEUE["Priority-aware scheduler"]
    WHISPER["faster-whisper large-v3"]
    QWEN["Optional Qwen3-ASR worker"]
    POLICY["Hybrid routing + gap recovery"]
    COMMIT["Temporal deduplication + confirmation"]
    WINDOW --> FILTER --> QUEUE
    QUEUE --> WHISPER --> POLICY --> COMMIT
    QUEUE -. "hybrid" .-> QWEN
    QWEN --> POLICY
  end

  subgraph VOICE["04 / Acoustic speaker inference"]
    direction TB
    CAM["CAM++ embeddings"]
    PROFILE["Enrolled voiceprints"]
    CLUSTER["Session speaker clustering"]
    NEM["Optional Nemotron 3 stream"]
    ACT["Speaker activity + overlap"]
    GATE["Channel corroboration + continuity"]
    CAM --> CLUSTER --> GATE
    PROFILE --> CLUSTER
    NEM --> ACT --> GATE
  end

  subgraph FUSION["05 / Timeline fusion"]
    direction TB
    ALIGN["Timestamp alignment"]
    GUARD["Text conservation + speaker policy"]
    FINAL["Speaker-aware transcript"]
    ALIGN --> GUARD --> FINAL
  end

  subgraph STORAGE["06 / Persistent meeting state"]
    direction TB
    SAVE["Durable save queue"]
    REVIEW["Human correction + revision"]
    DATABASE["Local files / PostgreSQL"]
    SAVE --> DATABASE
    REVIEW <--> DATABASE
  end

  subgraph KNOWLEDGE["07 / Meeting intelligence"]
    direction TB
    JOB["Persistent job queue"]
    WORKER["Independent AI worker"]
    OLLAMA["Local Ollama gateway"]
    INDEX["Revision-aware retrieval"]
    EVID["Evidence selection + verification"]
    ART["Summaries, decisions + grounded answers"]
    JOB --> WORKER --> OLLAMA --> ART
    INDEX --> EVID --> ART
  end

  PCM --> WS
  BUF --> WINDOW
  CLOCK --> CAM
  CLOCK -. "independent PCM" .-> NEM
  COMMIT --> ALIGN
  GATE --> ALIGN
  FINAL --> SAVE
  FINAL --> OUT
  OUT --> WEB
  DATABASE --> REVIEW
  DATABASE --> JOB
  DATABASE --> INDEX
  ART --> DATABASE
  ART --> WEB
~~~

The recognition path owns the confirmed words. The acoustic path contributes speaker evidence. The reconciliation stage resolves their different time boundaries into a consistent record, which can later be reviewed or indexed.

### Live processing and confirmation sequence

~~~mermaid
sequenceDiagram
  autonumber
  participant Browser as Browser / Vue
  participant Socket as WebSocket
  participant Session as SessionProcessor
  participant Scheduler as ASR scheduler
  participant ASR as Whisper / Qwen
  participant Voice as Speaker workers
  participant Fusion as Timeline reconciliation
  participant Store as Meeting store

  Browser->>Socket: Open meeting stream
  Socket->>Session: Initialize session and bounded audio buffer
  loop Incoming audio chunks
    Browser->>Socket: 16 kHz PCM frames
    Socket->>Session: Append at absolute sample offset
    par Content recognition
      Session->>Scheduler: Submit speech window
      Scheduler->>ASR: Decode bounded audio
      ASR-->>Session: Timestamped candidates
      Session->>Session: Speech coverage, duplicate and commit checks
    and Acoustic speaker evidence
      Session->>Voice: Submit eligible audio windows
      Voice-->>Fusion: Speaker embeddings and activity
    end
    Session->>Fusion: Confirmed text + source time
    Fusion->>Fusion: Corroborate speakers, preserve wording
    Fusion-->>Session: Speaker-aware caption projection
    Session-->>Socket: Versioned transcript snapshot
    Socket-->>Browser: Updated meeting view
  end
  Browser->>Socket: Stop / save
  Socket->>Session: Finalize pending session state
  Session->>Store: Persist meeting and evidence
  Store-->>Browser: Reviewable saved record
  Note over Browser,Store: Edits carry revisions and source provenance
~~~

### Operational design contracts

| Boundary | Mechanism | Engineering purpose |
| --- | --- | --- |
| **Audio → inference** | Bounded buffer, absolute offsets, priority queue | Control load while maintaining one acoustic timebase |
| **Preview → confirmed speech** | Speech coverage, temporal deduplication, commit guards | Establish a stable transcript progression |
| **Channel → speaker identity** | CAM++ similarity, corroborated turns, stable session state | Separate an acoustic source from an asserted person |
| **Speaker activity → caption** | Content-first, overlap-aware projection | Attach attribution without duplicating or rewriting words |
| **Meeting → durable state** | Save queue, revisioned review, persisted evidence | Keep downstream analysis aligned with the saved record |
| **Meeting → knowledge** | Independent AI jobs, retrieval scope, source checks | Connect model interpretation to verifiable conversation content |

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

## From a meeting to verifiable knowledge

The intelligence layer operates on **saved, revisioned meeting content**, not the live ASR hypothesis. It preserves segment IDs, speaker context and source timing through indexing and structured analysis. This allows the Assistant to connect its findings to the meeting record rather than treating an isolated summary as authoritative.

### Knowledge and evidence provenance

~~~mermaid
flowchart TB
  subgraph SOURCE["01 / Saved context"]
    direction LR
    MEET["Transcript with segment IDs"]
    REVIEW["Human-reviewed revisions"]
    MEDIA["Optional imported media"]
  end

  subgraph PREP["02 / Index preparation"]
    direction LR
    CHUNK["Source-aligned chunks"]
    VERSION["Revision tracking"]
    LEX["Lexical index + scoring"]
    EMB["Optional semantic embeddings"]
    PG["PostgreSQL knowledge records"]
    CHUNK --> VERSION --> PG
    CHUNK --> LEX
    CHUNK --> EMB
  end

  subgraph QUERY["03 / Retrieval and local inference"]
    direction LR
    SCOPE["Question + meeting scope"]
    RANK["Bounded candidate ranking"]
    LLM["Local Ollama worker"]
    SELECT["Evidence excerpt selection"]
    SCOPE --> RANK --> SELECT
    LLM --> SELECT
  end

  subgraph OUTPUT["04 / Evidence-linked output"]
    direction LR
    VERIFY["Source-text verification"]
    FRESH["Revision freshness checks"]
    EXCERPT["Attributable excerpts"]
    ART["Structured meeting findings"]
    VERIFY --> FRESH --> EXCERPT
  end

  MEET --> CHUNK
  MEDIA --> CHUNK
  REVIEW -. "supersedes source revision" .-> VERSION
  PG --> RANK
  LEX --> RANK
  EMB --> RANK
  SELECT --> VERIFY
  MEET --> LLM
  LLM --> ART
  ART --> PG
  EXCERPT --> SCOPE
~~~

The local model can propose relevant excerpts, but the application validates those selections against stored text. Structured summaries, decisions and follow-ups retain their relationship to the source revision.

<details>
<summary><strong>Capture and review a conversation</strong></summary>

Create a meeting, choose a source language or automatic detection, and capture the room through the browser microphone. Review confirmed speech, speaker labels and timing. After saving, apply transcript corrections and speaker attribution edits with revision tracking.

</details>

<details>
<summary><strong>Analyze the meeting in context</strong></summary>

Use the local AI worker for summaries, decisions, follow-ups and speaker-specific perspectives. Available analysis modes include meeting, research, classroom, interview, standup and sales contexts.

</details>

<details>
<summary><strong>Retrieve and reuse its knowledge</strong></summary>

Index saved transcripts, ask evidence-focused questions, translate captions and export Markdown or JSON. Terminology libraries and speaker profiles remain available across meetings.

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
