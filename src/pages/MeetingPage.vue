<script setup lang="ts">
import IntelligencePanel from '@/components/IntelligencePanel.vue'
import Phase07Status from '@/components/Phase07Status.vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { downloadUrl, hotwordsApi, meetingsApi, speakersApi } from '@/api/client'
import { RealtimeClient } from '@/api/ws'
import { startMicRecording } from '@/audio/recorder'
import AppIcon from '@/components/AppIcon.vue'
import { speakerColor } from '@/speaker/clustering'
import type {
  HotwordLibrary,
  Paragraph,
  Segment,
  ServerReady,
  ServerTranscript,
  Speaker,
} from '@/types'

const props = defineProps<{ id: string }>()
const route = useRoute()
const router = useRouter()

const WS_BASE = (import.meta.env.VITE_WS_URL as string | undefined) ?? ''
const language = ref('en')
const translationTarget = ref('en')
const recordAudio = ref(false)
const partialText = ref('')
const partialLanguage = ref('')
const audioAnalysis = ref('')
const recordingList = ref<string[]>([])
const supportedLanguages = [
  ['auto', 'Auto detect'], ['zh', 'Chinese'], ['en', 'English'],
  ['ja', 'Japanese'], ['ko', 'Korean'], ['es', 'Spanish'],
  ['fr', 'French'], ['de', 'German'], ['pt', 'Portuguese'], ['ru', 'Russian'],
]

type Status = 'idle' | 'connecting' | 'recording' | 'stopping' | 'disconnected'

const recording = ref(false)
const status = ref<Status>('idle')
const error = ref('')
const toast = ref('')
const vadThreshold = ref(10)
const recThreshold = ref(2)
const serverInfo = ref<ServerReady | null>(null)
const meetingTitle = ref('')
const bufferedMs = ref(0)
const segments = ref<Segment[]>([])
const mergeGapS = ref(2)
const transcriptMeta = ref({ paragraphs: 0, segments: 0 })
const transcriptEl = ref<HTMLElement | null>(null)
const saving = ref(false)
const speakers = ref<Speaker[]>([])
const hotwords = ref<HotwordLibrary[]>([])
const showParticipants = ref(false)
const selectedParticipantIds = ref<string[]>([])
const showHotwords = ref(false)
const selectedHotwordIds = ref<string[]>([])
const spkThreshold = ref(0.3)
const applying = ref(false)
const micLevel = ref(0)
const micSilent = ref(false)

let client: RealtimeClient | null = null
let recorder: Awaited<ReturnType<typeof startMicRecording>> | null = null
let clientSeq = 0
let stickToBottom = true
let stopRequested = false
let realtimeError = false

const statusText = computed(() => {
  switch (status.value) {
    case 'connecting':
      return 'Connecting...'
    case 'recording':
      return 'Recording'
    case 'stopping':
      return 'Finishing...'
    case 'disconnected':
      return 'Disconnected'
    default:
      return 'Not started'
  }
})

const statusClass = computed(() => {
  if (recording.value || status.value === 'stopping') return 'ok'
  return status.value === 'disconnected' ? 'bad' : 'muted'
})

/** Group neighboring same-speaker segments locally into paragraphs. */
const paragraphs = computed<Paragraph[]>(() => buildParagraphs(segments.value))
const untranslatedCount = computed(() => segments.value.filter((seg) =>
  translationTarget.value && seg.language && seg.language !== 'und' &&
  seg.language !== translationTarget.value && !seg.translation?.trim(),
).length)
const participantNames = computed(() => {
  const byId = new Map(speakers.value.map((s) => [s.id, s.name]))
  return selectedParticipantIds.value
    .map((id) => byId.get(id))
    .filter((n): n is string => Boolean(n))
})
const hotwordNames = computed(() => {
  const byId = new Map(hotwords.value.map((h) => [h.id, h.name]))
  return selectedHotwordIds.value
    .map((id) => byId.get(id))
    .filter((n): n is string => Boolean(n))
})

function wsUrl(): string {
  const params = new URLSearchParams({
    vad_threshold: String(vadThreshold.value),
    rec_threshold: String(recThreshold.value),
    meeting_id: props.id,
    language: language.value,
    translate_to: translationTarget.value,
    record_audio: String(recordAudio.value),
    participants: selectedParticipantIds.value.join(','),
    hotwords: selectedHotwordIds.value.join(','),
    spk_threshold: String(spkThreshold.value),
  })
  if (WS_BASE) {
    const separator = WS_BASE.includes('?') ? '&' : '?'
    return `${WS_BASE}${separator}${params.toString()}`
  }
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${location.host}/ws/audio?${params.toString()}`
}

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 2600)
}

function mergeGapMs(): number {
  return Math.round((mergeGapS.value || 2) * 1000)
}

/** Join text without breaking CJK scripts or concatenating Latin words. */
function joinTranscriptText(left: string, right: string): string {
  const a = left.trimEnd()
  const b = right.trimStart()
  if (!a) return b
  if (!b) return a
  if (/[[({\uff08\u300a\u300c\u300e\u201c\u2018]$/.test(a) || /^[,!.?;:%)\]}\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u300b\u300d\u300f\u201d\u2019]/.test(b)) return a + b
  const cjk = /[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/
  if (cjk.test(a.slice(-1)) || cjk.test(b.charAt(0))) return a + b
  return a + ' ' + b
}

/** Merge adjacent segments with matching speakers and short gaps. */
function buildParagraphs(segs: Segment[]): Paragraph[] {
  const ordered = [...segs].sort((a, b) => a.start_ms - b.start_ms)
  const gap = mergeGapMs()
  const paras: Paragraph[] = []
  for (const seg of ordered) {
    const last = paras[paras.length - 1]
    const visibleText = translationTarget.value && seg.translation?.trim()
      ? seg.translation : seg.text
    if (
      last &&
      seg.speaker_id === last.speaker_id &&
      seg.start_ms - last.end_ms < gap
    ) {
      last.end_ms = seg.end_ms
      last.segmentIds.push(seg.id)
      last.text = joinTranscriptText(last.text, visibleText)
    } else {
      paras.push({
        id: seg.id,
        speaker_id: seg.speaker_id,
        speaker_name: seg.speaker_name,
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        text: visibleText,
        segmentIds: [seg.id],
        color: seg.color,
      })
    }
  }
  return paras
}

/** Replace the entire segment list when the backend sends a snapshot. */
function applyTranscript(msg: ServerTranscript): void {
  const segs: Segment[] = msg.segments.map((s) => ({
    ...s,
    color: speakerColor(s.speaker_id),
  }))
  segments.value = segs
  transcriptMeta.value = {
    segments: msg.segments.length,
    paragraphs: buildParagraphs(segs).length,
  }
  void scrollIfNeeded()
}

async function saveMeeting(quiet = false): Promise<boolean> {
  saving.value = true
  try {
    await meetingsApi.save(props.id)
    if (!quiet) showToast('Saved to disk')
    return true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    return false
  } finally {
    saving.value = false
  }
}

/** Load saved participant selections and speaker records before connecting. */
async function loadMeetingConfig(): Promise<void> {
  try {
    const { meeting } = await meetingsApi.get(props.id)
    selectedParticipantIds.value = meeting.participant_ids ?? []
    selectedHotwordIds.value = meeting.hotword_ids ?? []
    spkThreshold.value = meeting.spk_threshold ?? 0.3
  } catch {
    // Use default threshold 0.3 and empty participant/hotword selections.
    selectedParticipantIds.value = []
    selectedHotwordIds.value = []
    spkThreshold.value = 0.3
  }
  try {
    speakers.value = (await speakersApi.list()).speakers
    hotwords.value = (await hotwordsApi.list()).hotwords
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function toggleParticipant(id: string): void {
  const next = new Set(selectedParticipantIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedParticipantIds.value = [...next]
}

function toggleHotword(id: string): void {
  const next = new Set(selectedHotwordIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedHotwordIds.value = [...next]
}

/** Save participants and reconnect to update speaker labels. */
async function applyParticipants(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showParticipants.value = false
    if (!(await saveMeeting(true))) return
    connect()
    showToast('Participant settings applied')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    applying.value = false
  }
}

/** Save hotword selections and reconnect with new recognition terms. */
async function applyHotwords(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showHotwords.value = false
    if (!(await saveMeeting(true))) return
    connect()
    showToast('Hotword settings applied')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    applying.value = false
  }
}

async function exportMeeting(format: 'md' | 'json'): Promise<void> {
  const ok = await saveMeeting(true)
  if (!ok) return
  downloadUrl(meetingsApi.exportUrl(props.id, format))
  showToast(format === 'md' ? 'Markdown exported' : 'JSON exported')
}

async function copyTranscript(): Promise<void> {
  const text = paragraphs.value
    .map((item) => `[${item.speaker_name || `Speaker ${item.speaker_id}`}] ${item.text}`)
    .join('\n')
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
  }
  showToast('Transcript copied')
}

function clearTranscript(): void {
  segments.value = []
  transcriptMeta.value = { paragraphs: 0, segments: 0 }
}

async function startMic(): Promise<void> {
  if (!client?.connected || recording.value) return
  error.value = ''
  micLevel.value = 0
  micSilent.value = false
  let silentCount = 0
  try {
    recorder = await startMicRecording(
      (pcm) => client?.sendPcm(pcm),
      16000,
      (level) => {
        micLevel.value = level
        // Warn after about 1.5 seconds of continuous microphone silence.
        if (level < 0.0003) {
          silentCount++
          if (silentCount > 550) micSilent.value = true
        } else {
          silentCount = 0
          micSilent.value = false
        }
      },
    )
    recording.value = true
    status.value = 'recording'
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    status.value = serverInfo.value ? 'idle' : status.value
  }
}

function start(): void {
  if (recording.value || status.value === 'connecting' || status.value === 'stopping') return
  if (client?.connected) {
    void startMic()
    return
  }
  // Reconnect to resume the meeting after the previous session ends.
  error.value = ''
  connect(true)
}

function stop(): void {
  // Wait for final processing and automatic saving before closing.
  recording.value = false
  stopRequested = true
  recorder?.stop()
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  status.value = 'stopping'
  client?.sendStop()
}

function stopInternal(): void {
  recording.value = false
  recorder?.stop()
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  client?.close()
  client = null
  clientSeq += 1
}

function connect(autostart = false): void {
  // Dispose the previous socket before attempting a new connection.
  const previous = client
  client = null
  previous?.close()
  error.value = ''
  clearTranscript()
  bufferedMs.value = 0
  partialText.value = ''
  serverInfo.value = null
  stickToBottom = true
  stopRequested = false
  realtimeError = false

  const mySeq = ++clientSeq
  client = new RealtimeClient({
    url: wsUrl(),
    onReady: (msg) => {
      if (clientSeq !== mySeq) return
      serverInfo.value = msg
      meetingTitle.value = msg.title ?? 'Untitled Meeting'
      mergeGapS.value = msg.merge_gap_s || 2
      if (msg.participant_ids) selectedParticipantIds.value = msg.participant_ids
      if (msg.hotword_ids) selectedHotwordIds.value = msg.hotword_ids
      if (msg.spk_threshold !== undefined) spkThreshold.value = msg.spk_threshold
      status.value = 'idle'
      // Start automatically on create, resume, or restart.
      if (autostart || route.query.autostart === '1') void startMic()
    },
    onPartial: (text, detectedLanguage) => {
      if (clientSeq !== mySeq) return
      partialText.value = text
      partialLanguage.value = detectedLanguage
    },
    onAudioAnalysis: (result) => {
      if (clientSeq !== mySeq) return
      audioAnalysis.value = [result.emotion, ...(result.events || []),
        (result.overlap?.length || 0) > 0 ? 'Overlapping speech detected' : '']
        .filter(Boolean).join(', ')
    },
    onFeatureWarning: (message) => {
      if (clientSeq !== mySeq) return
      error.value = message
    },
    onTranscript: (msg) => {
      if (clientSeq !== mySeq) return
      partialText.value = ''
      applyTranscript(msg)
    },
    onState: (ms) => {
      if (clientSeq !== mySeq) return
      bufferedMs.value = ms
    },
    onError: (message) => {
      if (clientSeq !== mySeq) return
      realtimeError = true
      error.value = message
    },
    onClosed: (reason, saved) => {
      if (clientSeq !== mySeq) return
      client = null
      recording.value = false
      recorder?.stop()
      recorder = null
      if (reason === 'closed') {
        status.value = 'idle'
        if (saved === false) {
          error.value = 'Final meeting save failed. Please check the server log.'
        } else if (stopRequested && !realtimeError) {
          showToast('Stopped and automatically saved')
        }
      } else {
        status.value = 'disconnected'
        if (!realtimeError) error.value = 'Connection lost before final confirmation.'
      }
      stopRequested = false
    },
  })
  client.connect()
  status.value = 'connecting'
}

function onTranscriptScroll(): void {
  const el = transcriptEl.value
  if (!el) return
  stickToBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 80
}

async function scrollIfNeeded(): Promise<void> {
  if (!stickToBottom) return
  await nextTick()
  const el = transcriptEl.value
  if (el) el.scrollTop = el.scrollHeight
}

function formatTime(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(s).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

async function loadRecordings(): Promise<void> {
  try {
    const response = await fetch(`/api/speech/meetings/${encodeURIComponent(props.id)}/recordings`)
    if (!response.ok) throw new Error('Unable to load recordings')
    const payload = await response.json() as { recordings: Array<{ name: string }> }
    recordingList.value = payload.recordings.map((r) => r.name)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function editSegmentById(id: number): void {
  const segment = segments.value.find((item) => item.id === id)
  if (segment) void correctTranscript(segment)
}

async function correctTranscript(seg: Segment): Promise<void> {
  if (recording.value || status.value === 'stopping') return
  const text = window.prompt('Correct transcript segment', seg.text)
  if (text === null || !text.trim() || text === seg.text) return
  try {
    const response = await fetch(`/api/speech/meetings/${encodeURIComponent(props.id)}/segments/${seg.id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, revision: seg.revision ?? 0 }),
    })
    if (!response.ok) throw new Error(await response.text())
    const result = await response.json() as { text: string, revision: number }
    segments.value = segments.value.map((item) => item.id === seg.id ? { ...item, text: result.text, revision: result.revision } : item)
    showToast('Correction saved')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(async () => {
  await loadMeetingConfig()
  connect()
})

onBeforeUnmount(() => stopInternal())
</script>

<template>
  <div class="meeting-page">
    <div class="meeting-head">
      <div class="meeting-head-left">
        <button class="btn btn-ghost btn-sm" @click="router.push('/')">
          <AppIcon name="arrow-left" :size="16" />
          Back to Home
        </button>
        <div>
          <h1 class="meeting-title">{{ meetingTitle || 'Untitled Meeting' }}</h1>
          <p class="meeting-sub">
            <span class="status-chip" :class="statusClass">
              <span class="status-dot" aria-hidden="true"></span>{{ statusText }}
            </span>
            <span v-if="recording" class="mic-level" :title="`Microphone level ${(micLevel * 100).toFixed(1)}%`">
              <span class="mic-level-bar" :style="{ width: Math.min(100, micLevel * 300) + '%' }"></span>
            </span>
            <span v-if="serverInfo">Buffered {{ (bufferedMs / 1000).toFixed(1) }}s</span>
            <span>Segments {{ transcriptMeta.segments }} \u00b7 Paragraphs {{ transcriptMeta.paragraphs }}</span>
            <span v-if="participantNames.length > 0" class="participant-chip">
              <AppIcon name="users" :size="13" />
              Participants: {{ participantNames.join(', ') }}
            </span>
            <span v-if="hotwordNames.length > 0" class="hotword-chip">
              <AppIcon name="flame" :size="13" />
              Hotword Libraries: {{ hotwordNames.join(', ') }}
            </span>
          </p>
        </div>
      </div>
      <div class="meeting-actions">
        <button class="btn btn-ghost" :disabled="saving" @click="saveMeeting()">
          <AppIcon name="save" :size="16" />
          {{ saving ? 'Saving...' : 'Save Meeting' }}
        </button>
        <button class="btn btn-ghost" @click="exportMeeting('md')">
          <AppIcon name="download" :size="16" />
          Export MD
        </button>
        <button class="btn btn-ghost" @click="exportMeeting('json')">
          <AppIcon name="file-text" :size="16" />
          Export JSON
        </button>
        <button class="btn btn-primary" :class="{ danger: recording }" @click="recording ? stop() : start()">
          <AppIcon :name="recording ? 'stop' : 'mic'" :size="16" />
          {{ recording ? 'Stop Recording' : 'Start Recording' }}
        </button>
      </div>
    </div>

    <Phase07Status :segments="segments" :buffered-ms="bufferedMs"
      :recording="recording" :asr-engine="serverInfo?.asr_engine ?? 'unknown'"
      :asr-model="serverInfo?.asr_model" :diarization-mode="serverInfo?.diarization_mode" />

    <div class="meeting-controls">
      <label class="field">Spoken language
        <select v-model="language" :disabled="recording || status === 'connecting' || status === 'stopping'">
          <option v-for="entry in supportedLanguages" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
        </select>
      </label>
      <label class="field">Display translation
        <select v-model="translationTarget" :disabled="recording || status === 'connecting' || status === 'stopping'">
          <option value="">No translation</option>
          <option v-for="entry in supportedLanguages.filter((item) => item[0] !== 'auto')" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
        </select>
      </label>
      <label class="field"><input type="checkbox" v-model="recordAudio" :disabled="recording" /> Save WAV</label>
      <button class="btn btn-ghost" :disabled="recording" @click="connect()">Apply speech settings</button>
      <button class="btn btn-ghost" :disabled="recording" @click="loadRecordings()">Load recordings</button>
      <label class="field" title="Trigger recognition when this audio duration is buffered.">
        Processing Window (s)
        <input
          v-model.number="vadThreshold"
          type="number"
          min="0.5"
          max="30"
          step="0.5"
          :disabled="recording"
        />
      </label>
      <label class="field" title="Confirm a sentence after this trailing audio delay.">
        Confirmation Delay (s)
        <input
          v-model.number="recThreshold"
          type="number"
          min="0.2"
          max="30"
          step="0.5"
          :disabled="recording"
        />
      </label>
      <button class="btn btn-ghost" :disabled="paragraphs.length === 0" @click="copyTranscript">
        Copy Transcript
      </button>
      <button
        class="btn btn-ghost"
        :title="recording ? 'Stop recording before changing participants' : 'Select enrolled speakers as participants'"
        @click="showParticipants = true"
      >
        <AppIcon name="users" :size="15" />
        Participants{{ selectedParticipantIds.length > 0 ? `(${selectedParticipantIds.length} people \u00b7 ${spkThreshold})` : '' }}
      </button>
      <button
        class="btn btn-ghost"
        :title="recording ? 'Stop recording before changing hotwords' : 'Select hotwords for speech recognition'"
        @click="showHotwords = true"
      >
        <AppIcon name="flame" :size="15" />
        Hotwords{{ selectedHotwordIds.length > 0 ? `(${selectedHotwordIds.length} items)` : '' }}
      </button>
      <button class="btn btn-ghost" :disabled="segments.length === 0" @click="clearTranscript">
        Clear
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>
    <p v-if="audioAnalysis" class="meeting-sub">Audio analysis: {{ audioAnalysis }}</p>
    <p v-if="untranslatedCount > 0" class="hint-text warn-text" role="status">
      {{ untranslatedCount }} segment(s) have no available translation; original text is shown.
    </p>
    <section v-if="partialText" class="meeting-controls" aria-live="polite">
      <strong>Live preview ({{ partialLanguage }})</strong>
      <span>{{ partialText }}</span>
    </section>
    <div v-if="recordingList.length" class="meeting-controls">
      <span>Recorded audio</span>
      <audio v-for="name in recordingList" :key="name" controls :src="`/api/speech/meetings/${encodeURIComponent(props.id)}/recordings/${encodeURIComponent(name)}`"></audio>
    </div>

    <!-- PHASE05_INTELLIGENCE_PATCH -->
    <IntelligencePanel :meeting-id="props.id" :segments="segments" />
    <main ref="transcriptEl" class="transcript" @scroll="onTranscriptScroll">
      <div v-if="segments.length === 0" class="empty-state">
        <div class="empty-icon"><AppIcon name="mic" :size="26" /></div>
        <h3>{{ recording ? 'Listening...' : 'No transcript yet' }}</h3>
        <p v-if="!recording">Start recording and pause at least two seconds after speaking.</p>
        <p v-else>Pause for two seconds after speaking to create a segment.</p>
        <p v-if="recording && micSilent" class="hint-text warn-text" style="margin-top:8px">
          No microphone input detected. Check browser permission and system microphone settings.
        </p>
      </div>

      <template v-else>
        <article v-for="para in paragraphs" :key="para.id" class="paragraph">
          <div class="para-head">
            <span class="speaker-dot" :style="{ background: para.color }"></span>
            <span class="speaker-name">{{ para.speaker_name || `Speaker ${para.speaker_id}` }}</span>
            <span v-if="para.segmentIds.length > 1" class="seg-count">
              {{ para.segmentIds.length }} segments
            </span>
            <span class="para-time">{{ formatTime(para.start_ms) }} \u2013 {{ formatTime(para.end_ms) }}</span>
          </div>
          <p class="para-text">{{ para.text || '(No content in this paragraph)' }}</p>
          <div class="meeting-sub" v-if="!recording">
            <button v-for="segmentId in para.segmentIds" :key="segmentId"
              class="btn btn-ghost btn-sm" :disabled="status === 'stopping'"
              @click="editSegmentById(segmentId)">Edit #{{ segmentId }}</button>
          </div>
        </article>
      </template>
    </main>

    <div v-if="showParticipants" class="dialog-backdrop" @click.self="showParticipants = false">
      <div class="dialog dialog-wide">
        <h3>Meeting Participants</h3>
        <p class="dialog-sub">
          Select enrolled participants. Each segment is matched against their voiceprints.
          Matched segments use participant names. Unmatched segments are clustered.
        </p>
        <div class="participant-list">
          <label
            v-for="spk in speakers"
            :key="spk.id"
            class="participant-row"
            :class="{ checked: selectedParticipantIds.includes(spk.id) }"
          >
            <input
              type="checkbox"
              :checked="selectedParticipantIds.includes(spk.id)"
              @change="toggleParticipant(spk.id)"
            />
            <span class="speaker-avatar">{{ spk.name.slice(0, 1) }}</span>
            <span class="participant-name">{{ spk.name }}</span>
            <span class="muted-text">{{ spk.vector_count }} voiceprints</span>
          </label>
          <div v-if="speakers.length === 0" class="empty-state small">
            <p>No enrolled voiceprints. Add a speaker first.</p>
          </div>
        </div>
        <div class="threshold-row">
          <label class="field">
            Match Threshold
            <input
              v-model.number="spkThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
            />
          </label>
          <span class="muted-text">
            The default threshold is 0.3. Lower values match more readily.
          </span>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          Stop recording before changing participants.
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showParticipants = false">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyParticipants"
          >
            {{ applying ? 'Applying...' : 'Apply and Reconnect' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showHotwords" class="dialog-backdrop" @click.self="showHotwords = false">
      <div class="dialog dialog-wide">
        <h3>Meeting Hotwords</h3>
        <p class="dialog-sub">
          Selected hotwords improve recognition of specialized terms,
          names, and products, up to 200 hotwords per meeting.
        </p>
        <div class="hotword-pick-list">
          <label
            v-for="lib in hotwords"
            :key="lib.id"
            class="hotword-pick-row"
            :class="{ checked: selectedHotwordIds.includes(lib.id) }"
          >
            <input
              type="checkbox"
              :checked="selectedHotwordIds.includes(lib.id)"
              @change="toggleHotword(lib.id)"
            />
            <span class="hotword-pick-icon"><AppIcon name="flame" :size="15" /></span>
            <span class="hotword-pick-info">
              <span class="hotword-pick-name">
                {{ lib.name }}
                <span class="muted-text">{{ lib.word_count }} hotwords</span>
              </span>
              <span v-if="lib.words.length > 0" class="hotword-tags compact">
                <span
                  v-for="w in lib.words.slice(0, 5)"
                  :key="w"
                  class="hotword-tag"
                >
                  {{ w }}
                </span>
                <span v-if="lib.words.length > 5" class="hotword-more">
                  +{{ lib.words.length - 5 }}
                </span>
              </span>
            </span>
          </label>
          <div v-if="hotwords.length === 0" class="empty-state small">
            <p>No hotword libraries yet. Create one on the home page.</p>
          </div>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          Stop recording before changing hotwords.
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showHotwords = false">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyHotwords"
          >
            {{ applying ? 'Applying...' : 'Apply and Reconnect' }}
          </button>
        </div>
      </div>
    </div>

    <Transition name="toast">
      <div v-if="toast" class="toast">
        <AppIcon name="check" :size="16" />
        {{ toast }}
      </div>
    </Transition>
  </div>
</template>
