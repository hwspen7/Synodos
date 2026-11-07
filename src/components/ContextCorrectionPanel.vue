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
      <div class="context-actions context-review-toolbar">
        <label class="context-mode"><span>Review style</span>
          <select v-model="mode" :disabled="busy" aria-label="Review style">
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
      <p class="context-review-note">Suggestions only. Changes require your approval.</p>
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
      <button v-if="!result.complete" class="btn btn-ghost btn-sm"
        :disabled="!canReview || suggestions.length > 0"
        @click="startReview(true)">Review next 32 segments</button>
      <p v-if="!result.complete && suggestions.length > 0" class="context-hint">
        Accept or dismiss the current suggestions before reviewing the next section.
      </p>
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

/* Product-level review card */
.context-review{border-color:#e8e6e2;background:#fff;border-radius:14px;padding:20px;margin:0 0 15px;box-shadow:none}
.context-head{gap:17px}
.context-head h3{font-size:15px;font-weight:770;color:#292c31;letter-spacing:-.02em}
.context-hint{opacity:1;color:#77787f;font-size:11px;line-height:1.65}
.context-actions{gap:8px;align-items:flex-end}
.context-mode{font-size:11px;font-weight:640;color:#6e7178}
.context-mode select{min-width:148px;min-height:39px;border:1px solid #e0dfdc;border-radius:9px;padding:7px 10px;background:#fff;color:#30333a}
.context-results{gap:11px}
.context-summary{color:#696b71;background:#faf8f5;padding:10px 12px;border-radius:9px}
.context-item{background:#fff;border:1px solid #e7e5e1;border-radius:11px;padding:15px}
.context-item p{line-height:1.65}

/* Assistant Corrections: a single focused toolbar and clear change comparisons. */
.context-review {
  margin:0;padding:0;border:0;border-radius:0;background:transparent;box-shadow:none;
}
.context-head {
  display:flex;flex-direction:column;align-items:stretch;gap:11px;
}
.context-review-toolbar {
  display:grid;grid-template-columns:minmax(0,1fr) minmax(125px,.8fr);
  align-items:end;gap:10px;
}
.context-review-toolbar .context-mode {
  display:flex;flex-direction:column;min-width:0;gap:7px;
  color:#6a6057;font-size:11px;font-weight:650;
}
.context-review-toolbar .context-mode select {
  box-sizing:border-box;display:block;width:100%;min-width:0;height:41px;
  min-height:41px;padding:0 29px 0 11px;
  border:1px solid #e5ded6;border-radius:9px;
  background:#fff;color:#302a25;font-size:12px;
}
.context-review-toolbar > button.btn {
  display:inline-flex;align-items:center;justify-content:center;
  box-sizing:border-box;height:41px;min-height:41px;
  min-width:0;margin:0;padding:0 12px;border-radius:9px;
  white-space:nowrap;font-size:12px;font-weight:690;
}
.context-review-toolbar > button.btn-primary {
  background:#c4550d;border-color:#c4550d;color:#fff;
}
.context-review-note {
  margin:1px 0 4px;color:#8e8379;font-size:11px;line-height:1.5;
}
.context-results {margin-top:14px;gap:13px;}
.context-results > .context-summary {
  display:flex;gap:5px 13px;flex-wrap:wrap;
  margin:0;padding:9px 0;border-bottom:1px solid #e9e4de;
  border-radius:0;background:transparent;
  color:#736a60;font-size:10px;
}
.context-item {
  min-width:0;padding:15px;border:1px solid #e9e3db;
  border-radius:11px;background:#fff;box-shadow:none;
}
.context-item .context-summary {
  display:flex;justify-content:space-between;gap:8px;
  margin:0 0 12px;padding:0;background:transparent;
  color:#8b8176;font-size:10px;
}
.context-item .context-summary strong {
  color:#322a24;font-size:11px;
}
.context-item > p:not(.context-hint) {
  margin:8px 0;padding:11px 12px;border:1px solid #eee9e3;
  border-radius:8px;background:#faf9f7;color:#443d36;
  font-size:11px;line-height:1.65;
}
.context-item > p:not(.context-hint) strong {
  display:block;margin-bottom:4px;color:#776a5f;font-size:10px;
}
.context-item > p:nth-of-type(2) {
  border-color:#eed9c7;background:#fffaf5;
}
.context-item > p:nth-of-type(2) strong {color:#a65012;}
.context-item > .context-actions {margin-top:12px;align-items:center;}
.context-item > .context-actions button {
  min-height:32px;padding:6px 12px;border-radius:8px;
}
@media(max-width:370px){
  .context-review-toolbar {grid-template-columns:minmax(0,1fr);}
}
</style>
