<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AppIcon from '@/components/AppIcon.vue'
import Phase2IdentityReviewPanel from '@/components/Phase2IdentityReviewPanel.vue'
import type { Segment } from '@/types'

type Tab = 'insights' | 'statements' | 'identity'
type InsightField = 'viewpoints' | 'decisions' | 'actions' | 'questions'
interface EvidenceItem {
  title: string
  source_ids: number[]
  owner?: string | null
  due_date?: string | null
}
interface AnalysisContent {
  speaker_id: number
  speaker_name: string
  summary: string
  summary_source_ids: number[]
  segment_ids: number[]
  viewpoints: EvidenceItem[]
  decisions: EvidenceItem[]
  actions: EvidenceItem[]
  questions: EvidenceItem[]
  human_review_required: boolean
}
interface AnalysisArtifact {
  id: string
  content: AnalysisContent
  stale: boolean
  source_revision: number
  model_name: string
}
interface AnalysisStatus {
  speaker_id: number
  speaker_name: string
  source_revision: number
  segment_ids: number[]
  artifact: AnalysisArtifact | null
  active_job: { id: string; status: string } | null
}
interface TaskStatus { id: string; status: string; error?: string | null }

const props = defineProps<{
  meetingId: string
  speakerId: number
  speakerName: string
  segments: Segment[]
  recording: boolean
  selectedSegmentId: number | null
  tab: Tab
}>()
const emit = defineEmits<{
  (event: 'update:tab', tab: Tab): void
  (event: 'reviewed'): void
  (event: 'jump-to', segmentId: number): void
}>()

const filtered = computed(() => [...props.segments]
  .filter(s => s.speaker_id === props.speakerId && s.text.trim())
  .sort((a, b) => a.start_ms - b.start_ms || a.id - b.id))
const byId = computed(() => new Map(filtered.value.map(s => [s.id, s])))
const identitySegmentId = ref<number | null>(props.selectedSegmentId)
const loaded = ref(false)
const busy = ref(false)
const error = ref('')
const status = ref<AnalysisStatus | null>(null)
const pendingId = ref<string | null>(null)
let refreshGeneration = 0
let pollTimer: ReturnType<typeof setInterval> | null = null

const groupedSections: Array<{ key: InsightField; title: string; description: string }> = [
  { key: 'viewpoints', title: 'Viewpoints', description: 'Positions and arguments they expressed' },
  { key: 'decisions', title: 'Stated decisions', description: 'Decisions they discussed, not necessarily approved by the group' },
  { key: 'actions', title: 'Actions & commitments', description: 'Assignments, commitments and next steps' },
  { key: 'questions', title: 'Questions', description: 'Unresolved points raised by this speaker' },
]
const currentArtifact = computed(() => {
  const artifact = status.value?.artifact
  return artifact && !artifact.stale && artifact.content.speaker_id === props.speakerId ? artifact : null
})
const stale = computed(() => Boolean(status.value?.artifact?.stale))
const analyzing = computed(() => busy.value || Boolean(status.value?.active_job))
const canAnalyze = computed(() =>
  props.speakerId > 0 && filtered.value.length > 0 && !props.recording && loaded.value
  && status.value !== null && !analyzing.value
)
const insightContent = computed(() => currentArtifact.value?.content ?? null)

function timecode(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}
function sourceLabel(id: number): string {
  const item = byId.value.get(id)
  return item ? `#${id} · ${timecode(item.start_ms)}` : `#${id}`
}
async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json() as { detail?: string }
      detail = body.detail || detail
    } catch { /* HTTP status remains available */ }
    throw new Error(detail)
  }
  return await response.json() as T
}
async function refresh(): Promise<void> {
  if (props.speakerId <= 0) { loaded.value = true; status.value = null; return }
  const generation = ++refreshGeneration
  try {
    const endpoint = `/api/intelligence/meetings/${encodeURIComponent(props.meetingId)}/speakers/${props.speakerId}/insights`
    const updated = await fetchJson<AnalysisStatus>(endpoint)
    if (generation !== refreshGeneration) return
    status.value = updated
    error.value = ''
  } catch (cause) {
    if (generation === refreshGeneration) {
      error.value = cause instanceof Error ? cause.message : String(cause)
      status.value = null
    }
  } finally {
    if (generation === refreshGeneration) loaded.value = true
  }
}
async function startAnalysis(): Promise<void> {
  if (!canAnalyze.value || !status.value) return
  busy.value = true
  error.value = ''
  try {
    const result = await fetchJson<{ job_id: string }>('/api/intelligence/jobs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: 'speaker_insights', meeting_id: props.meetingId,
        payload: { speaker_id: props.speakerId, source_revision: status.value.source_revision },
      }),
    })
    pendingId.value = result.job_id
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  } finally { busy.value = false }
}
async function poll(): Promise<void> {
  if (!pendingId.value && !status.value?.active_job) return
  try {
    const jobId = pendingId.value ?? status.value?.active_job?.id
    if (!jobId) return
    const state = await fetchJson<TaskStatus>(`/api/intelligence/jobs/${encodeURIComponent(jobId)}`)
    if (state.status === 'failed' || state.status === 'cancelled') {
      pendingId.value = null
      error.value = state.error || 'Analysis did not complete. Try again.'
      await refresh()
      return
    }
    if (state.status === 'completed') {
      pendingId.value = null
      await refresh()
    } else if (!pendingId.value) {
      pendingId.value = jobId
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

watch(() => [props.meetingId, props.speakerId] as const, () => {
  refreshGeneration += 1
  pendingId.value = null
  loaded.value = false
  error.value = ''
  status.value = null
  identitySegmentId.value = props.selectedSegmentId
  void refresh()
})
watch(() => props.selectedSegmentId, value => { if (value !== null) identitySegmentId.value = value })
watch(() => props.segments.map(s => `${s.id}:${s.revision ?? 0}:${s.speaker_id}:${s.text}`).join('|'), () => {
  // Existing AI artifacts are invalidated by the backend revision check.
  void refresh()
})
onMounted(() => {
  void refresh()
  pollTimer = setInterval(() => { void poll() }, 3000)
})
onBeforeUnmount(() => {
  refreshGeneration += 1
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <section class="speaker-contribution" aria-label="Individual speaker review">
    <header class="speaker-contribution-header">
      <div>
        <h3>{{ speakerName }}</h3>
        <p>{{ filtered.length }} attributed {{ filtered.length === 1 ? 'segment' : 'segments' }} · Results depend on speaker attribution.</p>
      </div>
    </header>
    <nav class="speaker-contribution-tabs" aria-label="Speaker detail sections">
      <button type="button" :class="{ active: tab === 'insights' }"
        :aria-current="tab === 'insights' ? 'page' : undefined" @click="emit('update:tab', 'insights')">Insights</button>
      <button type="button" :class="{ active: tab === 'statements' }"
        :aria-current="tab === 'statements' ? 'page' : undefined" @click="emit('update:tab', 'statements')">Statements</button>
      <button type="button" :class="{ active: tab === 'identity' }"
        :aria-current="tab === 'identity' ? 'page' : undefined" @click="emit('update:tab', 'identity')">Identity</button>
    </nav>

    <div v-if="tab === 'insights'" class="speaker-insights-view">
      <p class="speaker-review-disclaimer">AI analysis uses only this speaker's saved transcript turns. Speaker attribution and extracted claims require human review.</p>
      <p v-if="speakerId <= 0" class="speaker-review-message">Unidentified turns cannot be reliably attributed to one person. Review speaker identities first.</p>
      <template v-else>
        <button type="button" class="speaker-analyze-button" :disabled="!canAnalyze"
          @click="startAnalysis">
          <AppIcon name="file-text" :size="16" />
          {{ analyzing ? 'Analyzing…' : currentArtifact ? 'Regenerate analysis' : 'Analyze this speaker' }}
        </button>
        <p v-if="!loaded" class="speaker-review-message" role="status">Loading saved analysis...</p>
        <p v-if="status?.active_job" class="speaker-review-message" role="status">Analysis queued or running. It may wait until live recording stops.</p>
        <p v-if="recording" class="speaker-review-message">Stop and save recording before generating an individual analysis.</p>
        <p v-if="error" class="speaker-review-error" role="alert">{{ error }}</p>
        <p v-if="stale" class="speaker-review-message">The transcript changed since this analysis. Regenerate to use the corrected speaker turns.</p>
        <template v-if="insightContent">
          <section class="speaker-insight-section">
            <h4>Summary</h4>
            <p v-if="insightContent.summary" class="speaker-summary-text">{{ insightContent.summary }}</p>
            <p v-else class="speaker-review-message">No evidence-backed summary could be produced.</p>
            <div v-if="insightContent.summary_source_ids.length" class="speaker-evidence-links">
              <button v-for="sourceId in insightContent.summary_source_ids" :key="sourceId"
                type="button" @click="emit('jump-to', sourceId)">{{ sourceLabel(sourceId) }}</button>
            </div>
          </section>
          <section v-for="group in groupedSections" :key="group.key" class="speaker-insight-section">
            <h4>{{ group.title }} <span>{{ insightContent[group.key].length }}</span></h4>
            <p class="speaker-section-description">{{ group.description }}</p>
            <p v-if="!insightContent[group.key].length" class="speaker-review-message">No supported items found.</p>
            <div v-for="(item, index) in insightContent[group.key]" :key="group.key + index" class="speaker-insight-item">
              <p>{{ item.title }}</p>
              <small v-if="group.key === 'actions' && (item.owner || item.due_date)">
                {{ item.owner || 'Owner unspecified' }}<template v-if="item.due_date"> · {{ item.due_date }}</template>
              </small>
              <div class="speaker-evidence-links">
                <button v-for="sourceId in item.source_ids" :key="sourceId" type="button"
                  @click="emit('jump-to', sourceId)">{{ sourceLabel(sourceId) }}</button>
              </div>
            </div>
          </section>
          <small class="speaker-insight-footnote">Generated with {{ currentArtifact?.model_name || 'local AI' }} · Source-linked suggestions, not verified facts.</small>
        </template>
        <p v-else-if="loaded && !analyzing && !error" class="speaker-review-message">No analysis yet. Generate one to examine what this speaker contributed.</p>
      </template>
    </div>

    <div v-else-if="tab === 'statements'" class="speaker-statements-view">
      <p class="speaker-review-disclaimer">Original attributed turns in chronological order. Edits and speaker corrections are reflected here.</p>
      <p v-if="!filtered.length" class="speaker-review-message">No attributed statements.</p>
      <article v-for="segment in filtered" :key="segment.id" class="speaker-statement">
        <div class="speaker-statement-time">
          <span>{{ timecode(segment.start_ms) }} – {{ timecode(segment.end_ms) }}</span>
          <button type="button" @click="emit('jump-to', segment.id)">Go to transcript</button>
        </div>
        <p>{{ segment.text }}</p>
      </article>
    </div>

    <div v-else class="speaker-identity-view">
      <p class="speaker-review-disclaimer">Inspect and correct a specific segment's speaker identity. An automatic voice match is not proof of identity.</p>
      <label for="speaker-identity-segment" class="speaker-segment-label">Segment to verify</label>
      <select id="speaker-identity-segment" v-model="identitySegmentId" :disabled="recording || !filtered.length">
        <option v-for="segment in filtered" :key="segment.id" :value="segment.id">
          #{{ segment.id }} · {{ timecode(segment.start_ms) }} – {{ timecode(segment.end_ms) }}
        </option>
      </select>
      <Phase2IdentityReviewPanel v-if="identitySegmentId !== null"
        :meeting-id="meetingId" :segment-id="identitySegmentId"
        :revision="segments.find(segment => segment.id === identitySegmentId)?.revision ?? null"
        :recording="recording" @reviewed="emit('reviewed')" />
    </div>
  </section>
</template>
<style scoped>
.speaker-contribution{display:flex;flex-direction:column;gap:14px;min-width:0}
.speaker-contribution-header h3{font-size:15px;font-weight:770;margin:0;color:#282a30}
.speaker-contribution-header p{font-size:11px;line-height:1.6;color:#81838a;margin:5px 0 0}
.speaker-contribution-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;padding:4px;border-radius:11px;background:#f6f5f3;border:1px solid #eae7e2}
.speaker-contribution-tabs button{border:0;border-radius:8px;background:transparent;padding:9px 3px;color:#777980;font-size:11px;font-weight:700}
.speaker-contribution-tabs button.active{background:white;color:#a44f00;box-shadow:0 2px 5px rgba(50,30,10,.08)}
.speaker-review-disclaimer{margin:0;color:#747780;font-size:11px;line-height:1.65}
.speaker-insights-view,.speaker-statements-view,.speaker-identity-view{display:flex;flex-direction:column;gap:13px}
.speaker-analyze-button{display:flex;justify-content:center;align-items:center;gap:8px;background:#aa5100;color:#fff;border:0;border-radius:9px;padding:11px;font-size:12px;font-weight:730}
.speaker-analyze-button:disabled{opacity:.5;cursor:not-allowed}
.speaker-review-message{font-size:11px;line-height:1.6;color:#82848b;margin:0;padding:11px 12px;background:#faf9f7;border-radius:9px}
.speaker-review-error{color:#a83e22;font-size:11px;line-height:1.6;padding:10px;background:#fff5ef;border:1px solid #f2d4c2;border-radius:9px}
.speaker-insight-section{padding:15px;border:1px solid #e8e6e3;border-radius:11px;background:#fff}
.speaker-insight-section h4{display:flex;justify-content:space-between;align-items:center;margin:0 0 9px;font-size:12px;font-weight:770}
.speaker-insight-section h4 span{padding:2px 7px;border-radius:7px;background:#fff0e1;color:#a34f08;font-size:10px}
.speaker-section-description{color:#83858b;margin:0 0 11px;font-size:10px;line-height:1.6}
.speaker-summary-text{font-size:12px;line-height:1.75;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
.speaker-insight-item{padding:10px 0;border-top:1px solid #f0eeeb}
.speaker-insight-item:first-of-type{border-top:0}
.speaker-insight-item p{font-size:12px;line-height:1.6;margin:0 0 7px}
.speaker-insight-item small{font-size:10px;color:#8b8c92}
.speaker-evidence-links{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}
.speaker-evidence-links button{font-size:10px;border:1px solid #f1d9be;background:#fff7ed;color:#94500d;border-radius:7px;padding:5px 8px}
.speaker-evidence-links button:hover{background:#ffebd3}
.speaker-insight-footnote{color:#898b92;font-size:10px;line-height:1.6}
.speaker-statement{border:1px solid #e9e6e2;padding:13px;border-radius:10px}
.speaker-statement-time{display:flex;justify-content:space-between;gap:8px;color:#999ba0;font-size:10px}
.speaker-statement-time button{background:transparent;border:0;color:#a14b00;font-size:10px;text-decoration:underline}
.speaker-statement p{font-size:12px;line-height:1.7;white-space:pre-wrap;margin:9px 0 0;overflow-wrap:anywhere}
.speaker-segment-label{font-size:11px;color:#6f7178;font-weight:690}
.speaker-identity-view select{width:100%;padding:9px 10px;border:1px solid #e1ded9;background:#fff;border-radius:9px;color:#303139;font-size:11px}
button:focus-visible,select:focus-visible{outline:3px solid rgba(235,113,0,.3);outline-offset:2px}
</style>
