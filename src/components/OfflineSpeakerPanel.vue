<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface Recording {
  name: string
  bytes: number
  duration_s: number | null
}
interface SpeakerTurn {
  start_ms: number
  end_ms: number
  speaker: string
}
interface Result {
  model: string
  source_filename: string
  relative_to_recording: boolean
  speaker_turns: SpeakerTurn[]
  speaker_count: number
  review_required: boolean
  auto_applied: boolean
  notice: string
}
interface Job {
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  error?: string | null
  result?: Result | null
}
const props = defineProps<{ meetingId: string; recording: boolean }>()
const recordings = ref<Recording[]>([])
const selected = ref('')
const status = ref<'idle' | 'queued' | 'running' | 'completed' | 'failed'>('idle')
const error = ref('')
const result = ref<Result | null>(null)
const jobId = ref('')
let timer: ReturnType<typeof setTimeout> | null = null
let generation = 0
const busy = computed(() => status.value === 'queued' || status.value === 'running')
const canRun = computed(() => !props.recording && !busy.value && Boolean(selected.value))

function endPoll(): void {
  generation += 1
  if (timer !== null) clearTimeout(timer)
  timer = null
}
function reason(errorValue: unknown): string {
  return errorValue instanceof Error ? errorValue.message : String(errorValue)
}
async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const data = await response.json() as Record<string, unknown>
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed')
  }
  return data as T
}
async function refresh(): Promise<void> {
  if (props.recording) return
  try {
    const data = await json<{ recordings: Recording[] }>(
      '/api/speech/meetings/' + encodeURIComponent(props.meetingId) + '/recordings',
    )
    recordings.value = data.recordings
    if (!data.recordings.some((item) => item.name === selected.value)) {
      selected.value = data.recordings[0]?.name ?? ''
    }
  } catch (err) {
    error.value = reason(err)
  }
}
function time(ms: number): string {
  const seconds = Math.floor(Math.max(0, ms) / 1000)
  return String(Math.floor(seconds / 60)).padStart(2, '0') + ':' +
    String(seconds % 60).padStart(2, '0')
}
async function poll(id: string, current: number): Promise<void> {
  if (current !== generation || id !== jobId.value) return
  try {
    const job = await json<Job>('/api/intelligence/jobs/' + encodeURIComponent(id))
    if (current !== generation || id !== jobId.value) return
    if (job.status === 'completed') {
      if (!job.result || job.result.review_required !== true ||
        job.result.relative_to_recording !== true) {
        error.value = 'Invalid acoustic review result'
        status.value = 'failed'
        return
      }
      result.value = job.result
      status.value = 'completed'
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      status.value = 'failed'
      error.value = job.error || 'Second model could not finish the review'
    } else {
      status.value = job.status
      timer = setTimeout(() => { void poll(id, current) }, 2500)
    }
  } catch (err) {
    status.value = 'failed'
    error.value = reason(err)
  }
}
async function run(): Promise<void> {
  if (!canRun.value) return
  endPoll()
  result.value = null
  error.value = ''
  status.value = 'queued'
  try {
    const queued = await json<{ job_id: string }>('/api/intelligence/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: 'speaker_refine',
        meeting_id: props.meetingId,
        payload: { filename: selected.value },
      }),
    })
    jobId.value = queued.job_id
    await poll(queued.job_id, generation)
  } catch (err) {
    status.value = 'failed'
    error.value = reason(err)
  }
}
watch(() => props.meetingId, () => {
  endPoll()
  result.value = null
  recordings.value = []
  selected.value = ''
  status.value = 'idle'
  void refresh()
})
watch(() => props.recording, (active, previous) => {
  if (!active && previous) void refresh()
})
onMounted(() => { void refresh() })
onBeforeUnmount(endPoll)
</script>

<template>
  <section class="speaker-review">
    <h3>Independent speaker review</h3>
    <p class="speaker-note">
      Optional pyannote Community-1 model. Analyze one saved WAV after recording
      to compare its speaker-change timeline with the live transcription.
      Requires a separate Python model environment. No identities are overwritten.
    </p>
    <div class="speaker-controls">
      <label>
        WAV source
        <select v-model="selected" :disabled="props.recording || busy">
          <option v-for="file in recordings" :key="file.name" :value="file.name">
            {{ file.name }} ({{ file.duration_s === null ? 'unknown duration' : time(file.duration_s * 1000) }})
          </option>
        </select>
      </label>
      <button class="btn btn-ghost btn-sm" :disabled="props.recording || busy"
        @click="refresh">Refresh WAV files</button>
      <button class="btn btn-primary btn-sm" :disabled="!canRun" @click="run">
        {{ busy ? 'Analyzing...' : 'Analyze with second speaker model' }}
      </button>
    </div>
    <p v-if="busy" role="status" class="speaker-note">
      {{ status === 'queued' ? 'Waiting for background worker...' : 'Analyzing the saved acoustic recording...' }}
    </p>
    <p v-if="props.recording" class="speaker-note">Stop recording before running the second model.</p>
    <p v-if="error" class="speaker-error" role="alert">{{ error }}</p>
    <div v-if="result">
      <p class="speaker-note">
        {{ result.speaker_count }} acoustic speaker(s) ·
        {{ result.speaker_turns.length }} turns ·
        {{ result.model }}
      </p>
      <p class="speaker-note">{{ result.notice }}</p>
      <div class="speaker-turns">
        <div v-for="(turn, index) in result.speaker_turns" :key="index" class="speaker-turn">
          <span>{{ time(turn.start_ms) }} – {{ time(turn.end_ms) }}</span>
          <strong>{{ turn.speaker }}</strong>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.speaker-review { margin: 12px 0; padding: 16px; border-radius: 12px;
  border: 1px solid var(--border, #d1d5db); }
.speaker-review h3 { margin: 0 0 6px; font-size: 1rem; }
.speaker-controls { display: flex; align-items: flex-end; flex-wrap: wrap; gap: 10px; }
.speaker-controls label { display: grid; gap: 4px; font-size: .8rem; }
.speaker-controls select { max-width: 380px; min-height: 32px; }
.speaker-note { font-size: .82rem; opacity: .8; margin: 8px 0; }
.speaker-error { color: #bd4c3f; font-size: .85rem; }
.speaker-turns { max-height: 300px; overflow-y: auto; display: grid; gap: 4px; }
.speaker-turn { display: flex; gap: 14px; font-size: .85rem; padding: 6px;
  border-bottom: 1px solid var(--border, #ddd); }
</style>
