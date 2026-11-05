<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { meetingsApi, speakersApi, systemApi } from '@/api/client'
import { startMicRecording } from '@/audio/recorder'
import AppIcon from '@/components/AppIcon.vue'
import type { MeetingSegment, MeetingSummary, Speaker, SpeakerVector } from '@/types'

const props = defineProps<{ speaker: Speaker }>()
const emit = defineEmits<{
  added: [vectors: SpeakerVector[]]
  error: [message: string]
}>()

type Tab = 'record' | 'meeting'

const error = ref('')
const tab = ref<Tab>('record')
const minAudioS = ref(5)
const addedVectors = ref<SpeakerVector[]>([])

// ── Live Recording ────────────────────────────────────────────
const recording = ref(false)
const recDurationS = ref(0)
const recReady = ref(false)
const submittingRec = ref(false)
const pcmChunks: Int16Array[] = []
let recorder: Awaited<ReturnType<typeof startMicRecording>> | null = null

// ── From Meeting ──────────────────────────────────────────
const meetings = ref<MeetingSummary[]>([])
const selectedMeetingId = ref('')
const segments = ref<MeetingSegment[]>([])
const selectedSegIds = ref<Set<number>>(new Set())
const loadingSegments = ref(false)
const submittingSeg = ref(false)
const markHumanVerified = ref(false)

const selectedCount = computed(() => selectedSegIds.value.size)
const selectedSegments = computed(() => segments.value.filter(
  (item) => selectedSegIds.value.has(item.id),
))
const canHumanVerify = computed(() => selectedSegments.value.length > 0 &&
  selectedSegments.value.every((item) =>
    item.speaker_profile_id === props.speaker.id &&
    item.speaker_status === 'manual_override' &&
    !item.speaker_uncertain &&
    (item.review_count ?? 0) > 0 &&
    item.duration_s >= minAudioS.value,
  ),
)
const totalSelectedDuration = computed(() =>
  segments.value
    .filter((s) => selectedSegIds.value.has(s.id))
    .reduce((sum, s) => sum + s.duration_s, 0),
)
const selectedShortCount = computed(
  () =>
    segments.value.filter(
      (s) => selectedSegIds.value.has(s.id) && s.duration_s < minAudioS.value,
    ).length,
)

function fmtDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// ── Live Recording ────────────────────────────────────────────
async function startRecord(): Promise<void> {
  if (recording.value) return
  error.value = ''
  pcmChunks.length = 0
  recDurationS.value = 0
  recReady.value = false
  try {
    recorder = await startMicRecording((chunk) => {
      pcmChunks.push(chunk)
      const totalSamples = pcmChunks.reduce((sum, c) => sum + c.length, 0)
      recDurationS.value = totalSamples / 16000
    })
    recording.value = true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function stopRecord(): void {
  if (!recording.value) return
  recording.value = false
  recorder?.stop()
  recorder = null
  recReady.value = recDurationS.value >= 0.2
  if (!recReady.value) {
    error.value = 'Recording too short. Please try again.'
    pcmChunks.length = 0
    recDurationS.value = 0
  }
}

function pcmBlob(): Blob {
  // Copy the exact byte span of the Int16Array view into a Blob.
  return new Blob(
    pcmChunks.map((c) => c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength) as ArrayBuffer),
    { type: 'application/octet-stream' },
  )
}

async function submitRecording(): Promise<void> {
  if (submittingRec.value) return
  submittingRec.value = true
  error.value = ''
  try {
    const { vectors, min_audio_s } = await speakersApi.addRecord(
      props.speaker.id,
      pcmBlob(),
    )
    minAudioS.value = min_audio_s
    addedVectors.value = vectors
    emit('added', vectors)
    pcmChunks.length = 0
    recDurationS.value = 0
    recReady.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    emit('error', error.value)
  } finally {
    submittingRec.value = false
  }
}

// ── From Meeting ──────────────────────────────────────────
async function loadMeetingSegments(): Promise<void> {
  if (!selectedMeetingId.value) {
    segments.value = []
    return
  }
  loadingSegments.value = true
  error.value = ''
  try {
    const { segments: segs } = await meetingsApi.segments(selectedMeetingId.value)
    segments.value = segs
    selectedSegIds.value = new Set()
    markHumanVerified.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    segments.value = []
  } finally {
    loadingSegments.value = false
  }
}

function toggleSegment(id: number): void {
  const next = new Set(selectedSegIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedSegIds.value = next
  markHumanVerified.value = false
}

async function submitMeetingSelection(): Promise<void> {
  if (submittingSeg.value || selectedCount.value === 0) return
  if (markHumanVerified.value && !canHumanVerify.value) {
    error.value = 'Verified enrollment requires prior manual identity review and adequate clean audio'
    return
  }
  if (markHumanVerified.value && !window.confirm(
    'Have you independently listened to each selected segment and confirmed one speaker only? This enrollment will carry human-verified provenance.',
  )) return
  submittingSeg.value = true
  error.value = ''
  try {
    const { vectors, missing, min_audio_s } = await speakersApi.addFromMeeting(
      props.speaker.id,
      selectedMeetingId.value,
      [...selectedSegIds.value],
      markHumanVerified.value,
    )
    minAudioS.value = min_audio_s
    addedVectors.value = vectors
    emit('added', vectors)
    if (missing.length > 0) {
      error.value = `${missing.length} segments have no usable embeddings and were skipped`
      emit('error', error.value)
    }
    selectedSegIds.value = new Set()
    markHumanVerified.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    emit('error', error.value)
  } finally {
    submittingSeg.value = false
  }
}

onMounted(async () => {
  try {
    const health = await systemApi.health()
    minAudioS.value = health.min_speaker_audio_s
  } catch {
    /* Keep the default five-second value. */
  }
  try {
    const { meetings: list } = await meetingsApi.list()
    meetings.value = list
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
})

onBeforeUnmount(() => {
  recorder?.stop()
  recorder = null
})
</script>

<template>
  <div class="vector-adder">
    <div class="tab-bar">
      <button class="tab-btn" :class="{ active: tab === 'record' }" @click="tab = 'record'">
        <AppIcon name="mic" :size="15" />
        Live Recording
      </button>
      <button class="tab-btn" :class="{ active: tab === 'meeting' }" @click="tab = 'meeting'">
        <AppIcon name="file-text" :size="15" />
        From Meeting
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <!-- Live Recording -->
    <div v-if="tab === 'record'" class="record-area">
      <div class="read-card">
        <div class="read-card-head">
          <AppIcon name="mic" :size="15" />
          Read the following passage for at least {{ minAudioS }} seconds.
        </div>
        <blockquote class="read-passage">
          Meeting of Pioneers provides local speech transcription, speaker identification, and meeting archiving. Please read this paragraph clearly for voiceprint enrollment.
        </blockquote>
      </div>
      <button
        class="btn record-btn"
        :class="{ recording }"
        @click="recording ? stopRecord() : startRecord()"
      >
        <AppIcon :name="recording ? 'stop' : 'mic'" :size="18" />
        {{ recording ? 'Stop Recording' : 'Start Recording' }}
      </button>
      <div v-if="recDurationS > 0 || recReady" class="rec-meta">
        <span class="rec-time">{{ recDurationS.toFixed(1) }} s</span>
        <span v-if="recReady && recDurationS < minAudioS" class="warn-badge">
          <AppIcon name="alert" :size="13" />
          Audio too short (&lt; {{ minAudioS }} s). Matching may be unreliable.
        </span>
        <span v-else-if="recReady" class="ok-badge">Recording complete</span>
      </div>
      <p class="hint-text">
        Read the passage before recording. Samples shorter than {{ minAudioS }} seconds may be less reliable, but can still be saved.
      </p>
      <button
        v-if="recReady"
        class="btn btn-primary"
        :disabled="submittingRec"
        @click="submitRecording"
      >
        <AppIcon name="check" :size="16" />
        {{ submittingRec ? 'Extracting...' : 'Extract and Save Voiceprint' }}
      </button>
    </div>

    <!-- From Meeting -->
    <div v-else class="meeting-pick-area">
      <div class="field">
        Select Meeting
        <select v-model="selectedMeetingId" class="text-input" @change="loadMeetingSegments">
          <option value="" disabled>Select a saved meeting...</option>
          <option v-for="m in meetings" :key="m.id" :value="m.id">
            {{ m.title }} ({{ m.segments }} segments)
          </option>
        </select>
      </div>

      <div v-if="loadingSegments" class="empty-state small">
        <p>Loading segments...</p>
      </div>
      <div v-else-if="segments.length > 0" class="seg-select-list">
        <label
          v-for="seg in segments"
          :key="seg.id"
          class="seg-check-row"
          :class="{ checked: selectedSegIds.has(seg.id) }"
        >
          <input
            type="checkbox"
            :checked="selectedSegIds.has(seg.id)"
            @change="toggleSegment(seg.id)"
          />
          <span class="seg-text">{{ seg.text || '(No content in this segment)' }}</span>
          <span class="seg-time">{{ fmtDuration(seg.start_ms) }} – {{ fmtDuration(seg.end_ms) }}</span>
          <span class="seg-duration">{{ seg.duration_s.toFixed(1) }}s</span>
          <span v-if="seg.duration_s < minAudioS" class="warn-badge">
            <AppIcon name="alert" :size="12" />
            Too short
          </span>
        </label>
      </div>
      <div v-else-if="selectedMeetingId" class="empty-state small">
        <p>No confirmed segments are available to import.</p>
      </div>

      <div v-if="selectedCount > 0" class="pick-summary">
        <span>Selected {{ selectedCount }} segments totaling {{ totalSelectedDuration.toFixed(1) }} s</span>
        <span v-if="selectedShortCount > 0" class="warn-badge">
          <AppIcon name="alert" :size="13" />
          {{ selectedShortCount }} samples are shorter than {{ minAudioS }} seconds. Voiceprint accuracy may be lower.
        </span>
      </div>
      <label v-if="selectedCount > 0" class="hint-text">
        <input v-model="markHumanVerified" type="checkbox" :disabled="!canHumanVerify" />
        Human-verified enrollment (requires saved identity corrections,
        adequate duration and independent listening)
      </label>
      <p v-if="selectedCount > 0 && !canHumanVerify" class="hint-text">
        These segments can be saved as unverified candidates. To verify them,
        first correct the speaker identity in the meeting review panel.
      </p>
      <button
        v-if="selectedCount > 0"
        class="btn btn-primary"
        :disabled="submittingSeg"
        @click="submitMeetingSelection"
      >
        <AppIcon name="check" :size="16" />
        {{ submittingSeg ? 'Importing...' : 'Import Voiceprints' }}
      </button>
    </div>

    <div v-if="addedVectors.length > 0" class="adder-results">
      <h3 class="adder-results-title">New Voiceprints</h3>
      <div class="vector-result-list">
        <div v-for="v in addedVectors" :key="v.id" class="vector-result">
          <AppIcon name="check" :size="15" class="vector-ok" />
          <span>{{ v.source === 'recording' ? 'Live Recording' : 'Meeting Segment' }}</span>
          <span class="muted-text">{{ v.duration_s.toFixed(1) }} s</span>
          <span v-if="v.source === 'meeting' && v.segment_ids.length" class="muted-text">
            Segments #{{ v.segment_ids.join(', ') }}
          </span>
          <span v-if="v.human_verified" class="ok-badge">Human verified</span>
          <span v-else class="muted-text">Unverified candidate</span>
          <span v-if="v.short" class="warn-badge">
            <AppIcon name="alert" :size="12" />
            Audio too short
          </span>
          <span v-if="v.text" class="muted-text vector-text">{{ v.text }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
