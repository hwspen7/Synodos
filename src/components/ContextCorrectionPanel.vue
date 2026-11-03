<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { Segment } from '@/types'

type Mode = 'conservative' | 'readable'
type Status = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
interface Suggestion {
  segment_id: number
  original_text: string
  corrected_text: string
  reason: string
  category: string
  confidence: number
  expected_revision: number
}
interface Result {
  suggestions: Suggestion[]
  reviewed_segments: number
  offset: number
  next_offset: number
  total_segments: number
  complete: boolean
  review_required: boolean
  model?: string
}
interface Job {
  status: Status
  error?: string | null
  result?: Result | null
}

const props = defineProps<{ meetingId: string; segments: Segment[]; recording: boolean }>()
const emit = defineEmits<{ applied: [] }>()
const mode = ref<Mode>('conservative')
const jobId = ref<string | null>(null)
const status = ref<Status | null>(null)
const result = ref<Result | null>(null)
const error = ref('')
const notice = ref('')
const accepting = ref<number | null>(null)
const accepted = ref<number[]>([])
const dismissed = ref<number[]>([])
let handle: ReturnType<typeof setTimeout> | null = null
let generation = 0

const busy = computed(() => status.value === 'queued' || status.value === 'running')
const canReview = computed(() => !props.recording && !busy.value &&
  accepting.value === null && props.segments.length > 0)
const suggestions = computed(() => (result.value?.suggestions ?? []).filter(
  (item) => !accepted.value.includes(item.segment_id) &&
    !dismissed.value.includes(item.segment_id),
))

function resetPoll(): void {
  generation += 1
  if (handle !== null) clearTimeout(handle)
  handle = null
}
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  const body = await response.json() as Record<string, unknown>
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string'
      ? body.detail : String(response.status) + ' ' + response.statusText)
  }
  return body as T
}
function jobUrl(id: string): string {
  return '/api/intelligence/jobs/' + encodeURIComponent(id)
}
function messageFor(errorValue: unknown): string {
  return errorValue instanceof Error ? errorValue.message : String(errorValue)
}

async function poll(id: string, current: number): Promise<void> {
  if (generation !== current || jobId.value !== id) return
  try {
    const job = await request<Job>(jobUrl(id))
    if (generation !== current || jobId.value !== id) return
    status.value = job.status
    if (job.status === 'completed') {
      if (!job.result || job.result.review_required !== true) {
        error.value = 'The local AI returned an invalid correction review.'
        return
      }
      result.value = job.result
      notice.value = job.result.suggestions.length
        ? 'Review every proposed edit against the saved audio before accepting.'
        : 'No sufficiently safe changes were proposed in this section.'
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      error.value = job.error || 'Context review ' + job.status
    } else {
      handle = setTimeout(() => { void poll(id, current) }, 2500)
    }
  } catch (err) {
    if (generation === current) {
      status.value = null
      error.value = messageFor(err)
    }
  }
}

async function startReview(next = false): Promise<void> {
  if (!canReview.value) return
  const offset = next ? (result.value?.next_offset ?? 0) : 0
  if (next && result.value?.complete) return
  resetPoll()
  result.value = null
  accepted.value = []
  dismissed.value = []
  error.value = ''
  notice.value = ''
  status.value = 'queued'
  try {
    const queued = await request<{ job_id: string }>('/api/intelligence/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: 'context_correction', meeting_id: props.meetingId,
        payload: { mode: mode.value, offset, limit: 32 },
      }),
    })
    jobId.value = queued.job_id
    await poll(queued.job_id, generation)
  } catch (err) {
    status.value = null
    error.value = messageFor(err)
  }
}

function stale(item: Suggestion): boolean {
  const segment = props.segments.find((s) => s.id === item.segment_id)
  return !segment || segment.text !== item.original_text ||
    (segment.revision ?? 0) !== item.expected_revision
}

async function accept(item: Suggestion): Promise<void> {
  if (props.recording || accepting.value !== null) return
  if (stale(item)) {
    error.value = 'Segment #' + item.segment_id + ' has changed. Review again.'
    return
  }
  accepting.value = item.segment_id
  error.value = ''
  try {
    await request('/api/speech/meetings/' + encodeURIComponent(props.meetingId) +
      '/segments/' + item.segment_id, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: item.corrected_text, revision: item.expected_revision }),
    })
    accepted.value = [...accepted.value, item.segment_id]
    notice.value = 'Correction #' + item.segment_id + ' saved with its original revision history.'
    emit('applied')
  } catch (err) {
    error.value = messageFor(err)
  } finally {
    accepting.value = null
  }
}

function dismiss(item: Suggestion): void {
  dismissed.value = [...dismissed.value, item.segment_id]
}

async function cancel(): Promise<void> {
  if (!jobId.value || !busy.value) return
  try {
    const data = await request<{ cancelled: boolean }>(jobUrl(jobId.value) + '/cancel', {
      method: 'POST',
    })
    notice.value = data.cancelled
      ? 'Queued context review cancelled.' : 'Review already running. Cancellation is unavailable.'
    if (data.cancelled) {
      resetPoll()
      status.value = 'cancelled'
    }
  } catch (err) {
    error.value = messageFor(err)
  }
}

watch(() => props.meetingId, () => {
  resetPoll()
  jobId.value = null
  status.value = null
  result.value = null
  error.value = ''
  notice.value = ''
})
onBeforeUnmount(resetPoll)
</script>

<template>
  <section class="context-review" aria-label="Dialogue-aware transcript correction">
    <div class="context-head">
      <div>
        <h3>Dialogue-aware correction</h3>
        <p class="context-hint">
          Saluki reviews the surrounding conversation. Changes are suggestions only.
          Original speech, timing and speaker labels are never automatically overwritten.
        </p>
      </div>
      <div class="context-actions">
        <label class="context-mode">Review style
          <select v-model="mode" :disabled="busy">
            <option value="conservative">Faithful transcript</option>
            <option value="readable">Improve readability</option>
          </select>
        </label>
        <button class="btn btn-primary btn-sm" :disabled="!canReview"
          @click="startReview(false)">
          {{ busy ? 'Reviewing...' : 'Review with AI' }}
        </button>
        <button v-if="busy" class="btn btn-ghost btn-sm" @click="cancel">
          Cancel queued job
        </button>
      </div>
    </div>
    <p v-if="busy" class="context-hint" role="status">
      {{ status === 'queued' ? 'Waiting for the local model...' : 'Reviewing nearby dialogue...' }}
    </p>
    <p v-if="props.recording" class="context-hint">Stop recording before requesting a review.</p>
    <p v-if="error" class="context-error" role="alert">{{ error }}</p>
    <p v-if="notice" class="context-hint" role="status">{{ notice }}</p>
    <div v-if="result" class="context-results">
      <div class="context-summary">
        <span>{{ result.next_offset }} / {{ result.total_segments }} segments reviewed</span>
        <span>{{ suggestions.length }} suggested changes</span>
        <span v-if="result.model">Model: {{ result.model }}</span>
      </div>
      <article v-for="item in suggestions" :key="item.segment_id" class="context-item">
        <div class="context-summary">
          <strong>Segment #{{ item.segment_id }}</strong>
          <span>{{ item.category }}</span>
          <span>Model estimate {{ Math.round(item.confidence * 100) }}%</span>
        </div>
        <p><strong>Original</strong> {{ item.original_text }}</p>
        <p><strong>Suggestion</strong> {{ item.corrected_text }}</p>
        <p class="context-hint">{{ item.reason }}</p>
        <div class="context-actions">
          <button class="btn btn-primary btn-sm"
            :disabled="props.recording || accepting !== null || stale(item)"
            @click="accept(item)">
            {{ accepting === item.segment_id ? 'Saving...' : 'Accept' }}
          </button>
          <button class="btn btn-ghost btn-sm" :disabled="accepting !== null"
            @click="dismiss(item)">Dismiss</button>
          <span v-if="stale(item)" class="context-error">Transcript changed. Review again.</span>
        </div>
      </article>
      <button v-if="!result.complete" class="btn btn-ghost btn-sm" :disabled="!canReview"
        @click="startReview(true)">Review next 32 segments</button>
    </div>
  </section>
</template>

<style scoped>
.context-review {
  margin: 12px 0;
  padding: 16px;
  border: 1px solid var(--border, #d1d5db);
  border-radius: 12px;
  background: var(--surface, transparent);
}
.context-head, .context-actions, .context-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}
.context-head { justify-content: space-between; align-items: flex-start; }
.context-head h3 { margin: 0 0 6px; font-size: 1rem; }
.context-hint { margin: 6px 0; opacity: .8; font-size: .8rem; }
.context-mode { display: flex; flex-direction: column; gap: 4px; font-size: .8rem; }
.context-results { margin-top: 14px; display: grid; gap: 10px; }
.context-summary { font-size: .8rem; opacity: .85; }
.context-item {
  border: 1px solid var(--border, #ddd);
  border-radius: 8px;
  padding: 12px;
}
.context-item p { margin: 8px 0; overflow-wrap: anywhere; }
.context-error { font-size: .85rem; color: #bd4c3f; }
</style>
