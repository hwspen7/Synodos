<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AppIcon from './AppIcon.vue'
import { PHASE1_LANGUAGES } from '@/config/phase1Languages'

type Mode = 'meeting' | 'research' | 'classroom' | 'interview' | 'standup' | 'sales'
type View = 'overview' | 'translation'
type ReviewKind = 'actions' | 'decisions'
type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string

interface Segment {
  id: number
  text: string
  start_ms: number
  end_ms: number
  speaker_name?: string
}
interface EvidenceItem {
  title: string
  source_ids?: number[]
  owner?: string | null
  due_date?: string | null
}
interface ArtifactContent {
  summary?: string
  actions?: EvidenceItem[]
  decisions?: EvidenceItem[]
  open_questions?: EvidenceItem[]
  evidence_audit?: { claim_count: number; claims_with_valid_references: number; independent_fact_check_performed: boolean }
}
interface Job {
  id: string
  kind: string
  status: JobStatus
  error?: string | null
  result?: Record<string, unknown> | null
  created_at?: string
  updated_at?: string
  attempts?: number
}
interface Artifact {
  id: string
  kind: string
  mode: string
  content: ArtifactContent
  model_name?: string
  source_revision?: number
  stale?: boolean
  created_at?: string
}
interface Review {
  id: string
  title: string
  status: string
  revision: number
  evidence: number[]
  owner?: string | null
  due_date?: string | null
}
interface WorkspaceTranslation {
  segments: Segment[]
  translations: Array<{ segment_id: number; text: string }>
  active_jobs: number
  failed_jobs: number
  last_error: string | null
}
const props = defineProps<{ meetingId: string; segments: Segment[]; compact?: boolean; section?: View }>()
const modes: Array<{ id: Mode; name: string; description: string }> = [
  { id: 'meeting', name: 'Meeting', description: 'Summaries and decisions' },
  { id: 'research', name: 'Research', description: 'Experiments and findings' },
  { id: 'classroom', name: 'Classroom', description: 'Lectures and study notes' },
  { id: 'interview', name: 'Interview', description: 'Questions and answers' },
  { id: 'standup', name: 'Standup', description: 'Progress and blockers' },
  { id: 'sales', name: 'Sales', description: 'Client discussions' },
]
const mode = ref<Mode>('meeting')
const view = computed<View>(() => props.section ?? 'overview')
const language = ref('en')
const notice = ref('')
const error = ref('')
const jobs = ref<Job[]>([])
const artifacts = ref<Artifact[]>([])
const actions = ref<Review[]>([])
const decisions = ref<Review[]>([])
const translation = ref<WorkspaceTranslation | null>(null)
const translationLoading = ref(false)
const translationSubmitting = ref(false)
const translationError = ref('')
let translationScope = 0
const selectedArtifactId = ref<string | null>(null)
const busyKind = ref<string | null>(null)
const aiEnabled = ref(false)
const loading = ref(true)
const provider = ref('')
const modelName = ref('')
const showAllJobs = ref(false)
let pollId: ReturnType<typeof setInterval> | null = null
let refreshPending = false

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/intelligence${path}`, init)
  let raw: Record<string, unknown>
  try { raw = await response.json() as Record<string, unknown> }
  catch { throw new Error(`Unexpected server response (${response.status})`) }
  if (!response.ok) {
    throw new Error(typeof raw.detail === 'string' ? raw.detail : `Request failed (${response.status})`)
  }
  return raw as T
}
function jsonPost(data: unknown): RequestInit {
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }
}
function messageFromError(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}
function clearMessages(): void {
  error.value = ''
  notice.value = ''
}
function formatTime(ms: number): string {
  const safe = Math.max(0, Math.floor(Number(ms || 0) / 1000))
  const minutes = Math.floor(safe / 60)
  return `${String(minutes).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`
}
function formatDate(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? '' : date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
function kindLabel(kind: string): string {
  return ({ summary: 'Meeting notes', extract: 'Action extraction', index_meeting: 'Knowledge index',
    live_translate: 'Translation', translate: 'Translation', ask: 'Knowledge answer',
    analyze_media: 'Media analysis', index_media: 'Media index', ask_evidence: 'Evidence-only answer' } as Record<string, string>)[kind] ?? kind.replace(/_/g, ' ')
}
function evidenceText(ids?: number[]): string {
  return ids?.length ? `Segments ${ids.join(', ')}` : 'No source reference'
}
const latestArtifacts = computed(() => artifacts.value.slice(0, 15))
const selectedArtifact = computed(() => latestArtifacts.value.find(a => a.id === selectedArtifactId.value) ?? latestArtifacts.value[0] ?? null)
const activeJobs = computed(() => jobs.value.filter(j => j.status === 'queued' || j.status === 'running'))
const recentJobs = computed(() => showAllJobs.value ? jobs.value.slice(0, 15) : jobs.value.slice(0, 5))
const hasSegments = computed(() => props.segments.some(s => s.text.trim().length > 0))
const translationVisible = computed(() => aiEnabled.value && view.value === 'translation')
const translationRows = computed(() => translation.value?.segments ?? [])
const translated = computed(() => Object.fromEntries(
  (translation.value?.translations ?? []).map(item => [item.segment_id, item.text]),
))
const translatedCount = computed(() => Object.keys(translated.value).length)
const translationPending = computed(() => (translation.value?.active_jobs ?? 0) > 0)
const translationFailed = computed(() => (translation.value?.failed_jobs ?? 0) > 0)
const translationPath = computed(() =>
  `/meetings/${encodeURIComponent(props.meetingId)}/workspace-translation/${encodeURIComponent(language.value)}`,
)
const selectedContent = computed<ArtifactContent>(() => selectedArtifact.value?.content ?? {})
const selectedActionItems = computed(() => selectedContent.value.actions ?? [])
const selectedDecisionItems = computed(() => selectedContent.value.decisions ?? [])
const selectedOpenItems = computed(() => selectedContent.value.open_questions ?? [])
const briefSummary = computed(() =>
  artifacts.value.find(item => item.kind === 'summary' && !item.stale)
  ?? artifacts.value.find(item => item.kind === 'summary')
  ?? null)
const briefExtraction = computed(() =>
  artifacts.value.find(item => item.kind === 'extract' && !item.stale)
  ?? artifacts.value.find(item => item.kind === 'extract')
  ?? null)
const briefFindings = computed<ArtifactContent>(() =>
  briefSummary.value?.content ?? briefExtraction.value?.content ?? {})

async function refresh(silent = false): Promise<void> {
  if (!props.meetingId || refreshPending) return
  refreshPending = true
  try {
    const meetingId = props.meetingId
    const overview = await request<{ jobs: Job[]; artifacts: Artifact[]; actions: Review[]; decisions: Review[] }>(
      `/meetings/${encodeURIComponent(meetingId)}/overview`,
    )
    if (meetingId !== props.meetingId) return
    jobs.value = overview.jobs ?? []
    artifacts.value = overview.artifacts ?? []
    actions.value = overview.actions ?? []
    decisions.value = overview.decisions ?? []
    if (!silent) error.value = ''
  } catch (e) {
    error.value = messageFromError(e)
  } finally {
    refreshPending = false
  }
}
async function perform(kind: string, payload: Record<string, unknown> = {}): Promise<boolean> {
  if (busyKind.value) return false
  busyKind.value = kind
  clearMessages()
  try {
    await request<{ job_id: string }>('/jobs', jsonPost({ kind, meeting_id: props.meetingId, payload }))
    notice.value = `${kindLabel(kind)} queued. The AI worker will process it in the background.`
    await refresh()
    return true
  } catch (e) {
    error.value = messageFromError(e)
    return false
  } finally {
    busyKind.value = null
  }
}
async function analyze(): Promise<void> {
  await perform('summary', { mode: mode.value })
}
async function review(kind: ReviewKind, item: Review, status: string): Promise<void> {
  if (busyKind.value) return
  busyKind.value = `review-${item.id}`
  clearMessages()
  try {
    await request(`/review/${kind}/${item.id}`, jsonPost({ revision: item.revision, status }))
    notice.value = `${kind === 'actions' ? 'Action' : 'Decision'} updated.`
    await refresh()
  } catch (e) { error.value = messageFromError(e) }
  finally { busyKind.value = null }
}
async function cancelJob(job: Job): Promise<void> {
  if (job.status !== 'queued' || busyKind.value) return
  busyKind.value = `cancel-${job.id}`
  clearMessages()
  try {
    const result = await request<{ cancelled: boolean }>(`/jobs/${encodeURIComponent(job.id)}/cancel`, jsonPost({}))
    notice.value = result.cancelled ? 'Queued task cancelled.' : 'This task is already running and cannot be cancelled.'
    await refresh()
  } catch (e) { error.value = messageFromError(e) }
  finally { busyKind.value = null }
}
async function fetchTranslations(): Promise<void> {
  if (!translationVisible.value || translationLoading.value) return
  const scope = translationScope
  translationLoading.value = true
  try {
    const data = await request<WorkspaceTranslation>(translationPath.value)
    if (scope !== translationScope) return
    translation.value = data
    translationError.value = ''
  } catch (e) {
    if (scope === translationScope) translationError.value = messageFromError(e)
  } finally {
    if (scope === translationScope) translationLoading.value = false
  }
}
async function queueTranslations(): Promise<void> {
  if (!translationVisible.value || translationSubmitting.value || translationPending.value) return
  const scope = translationScope
  translationSubmitting.value = true
  translationError.value = ''
  try {
    await request(translationPath.value, { method: 'POST' })
    if (scope === translationScope) await fetchTranslations()
  } catch (e) {
    if (scope === translationScope) translationError.value = messageFromError(e)
  } finally {
    if (scope === translationScope) translationSubmitting.value = false
  }
}
watch([translationPath, translationVisible], () => {
  translationScope += 1
  translation.value = null
  translationLoading.value = false
  translationSubmitting.value = false
  translationError.value = ''
  void fetchTranslations()
})
watch(() => props.meetingId, () => {
  jobs.value = []; artifacts.value = []; actions.value = []; decisions.value = []
  selectedArtifactId.value = null
  void refresh()
})
onMounted(async () => {
  try {
    const capabilities = await request<{ enabled: boolean; provider?: string; text_model?: string }>('/capabilities')
    aiEnabled.value = capabilities.enabled
    provider.value = capabilities.provider ?? ''
    modelName.value = capabilities.text_model ?? ''
    if (aiEnabled.value) await refresh()
  } catch (e) { error.value = messageFromError(e) }
  finally { loading.value = false }
  pollId = setInterval(() => {
    if (aiEnabled.value) void refresh(true)
    if (!translationSubmitting.value) void fetchTranslations()
  }, 4000)
})
onBeforeUnmount(() => {
  translationScope += 1
  if (pollId) clearInterval(pollId)
})
</script>

<template>
  <section class="review-insights intelligence-inline" aria-label="Meeting AI tools">
    <template v-if="view === 'overview'">
    <p v-if="loading" class="review-insights-hint" role="status">Loading meeting insights...</p>
    <div v-else-if="!aiEnabled" class="review-insights-empty">
      <strong>AI analysis is unavailable</strong>
      <p>Enable the local AI worker to generate summaries. Transcript editing and speaker review remain available.</p>
    </div>
    <template v-else>
      <div class="review-insights-controls">
        <label class="review-inline-mode"><span class="visually-hidden">Analysis type</span>
          <select v-model="mode" aria-label="Analysis type" title="Analysis type">
            <option v-for="item in modes" :key="item.id" :value="item.id">{{ item.name }}</option>
          </select>
        </label>
        <button type="button" class="review-insights-primary"
          :disabled="!!busyKind || !hasSegments || activeJobs.some(j => j.kind === 'summary')"
          @click="analyze()">
          <AppIcon name="file-text" :size="16" />
          {{ busyKind === 'summary' ? 'Preparing...' : 'Generate summary' }}
        </button>
      </div>
      <p v-if="!hasSegments" class="review-insights-hint">No transcript available.</p>
      <p v-if="activeJobs.length" class="review-insights-hint" role="status">
        {{ activeJobs.length }} analysis {{ activeJobs.length === 1 ? 'task is' : 'tasks are' }} processing.
      </p>
      <div v-if="error" class="review-insights-error" role="alert">{{ error }}</div>
      <p v-if="notice" class="review-insights-hint" role="status">{{ notice }}</p>

      <div class="review-insights-results" aria-label="Meeting analysis results">
      <section class="review-insights-section">
        <h4>Summary</h4>
        <p v-if="briefSummary?.content.summary || briefFindings.summary" class="review-insights-narrative">{{ briefSummary?.content.summary || briefFindings.summary }}</p>
        <p v-else class="review-insights-empty-text">No summary yet.</p>
        <small v-if="briefSummary?.stale" class="review-insights-stale">Transcript changed since this summary was generated.</small>
      </section>
      <section class="review-insights-section">
        <h4>Decisions <span v-if="briefFindings.decisions?.length">{{ briefFindings.decisions.length }}</span></h4>
        <ul v-if="briefFindings.decisions?.length" class="review-insights-list">
          <li v-for="(item, i) in briefFindings.decisions" :key="'decision-' + i">
            <strong>{{ item.title }}</strong><small v-if="item.source_ids?.length">{{ evidenceText(item.source_ids) }}</small>
          </li>
        </ul>
        <p v-else class="review-insights-empty-text">No decisions.</p>
      </section>
      <section class="review-insights-section">
        <h4>Next steps <span v-if="briefFindings.actions?.length">{{ briefFindings.actions.length }}</span></h4>
        <ul v-if="briefFindings.actions?.length" class="review-insights-list">
          <li v-for="(item, i) in briefFindings.actions" :key="'action-' + i">
            <strong>{{ item.title }}</strong>
            <small>{{ item.owner || 'No owner assigned' }}<template v-if="item.due_date"> · {{ item.due_date }}</template></small>
          </li>
        </ul>
        <p v-else class="review-insights-empty-text">No next steps.</p>
      </section>
      <section v-if="briefFindings.open_questions?.length" class="review-insights-section">
        <h4>Open questions <span>{{ briefFindings.open_questions.length }}</span></h4>
        <ul class="review-insights-list">
          <li v-for="(item, i) in briefFindings.open_questions" :key="'open-' + i">
            <strong>{{ item.title }}</strong>
            <small v-if="item.source_ids?.length">{{ evidenceText(item.source_ids) }}</small>
          </li>
        </ul>
      </section>
      </div>
      <p v-if="briefSummary?.model_name" class="review-inline-provenance" :title="briefSummary.model_name">Generated by {{ briefSummary.model_name }}</p>
      <p v-if="briefFindings.evidence_audit" class="review-inline-provenance">
        {{ briefFindings.evidence_audit.claims_with_valid_references }} / {{ briefFindings.evidence_audit.claim_count }} claims linked to sources. This is not independent fact-checking.
      </p>
      <details v-if="latestArtifacts.length > 1" class="review-inline-details">
        <summary>Saved analyses ({{ latestArtifacts.length }})</summary>
        <label class="review-inline-mode">Select result
          <select v-model="selectedArtifactId" aria-label="Saved analysis result">
            <option v-for="artifact in latestArtifacts" :key="artifact.id" :value="artifact.id">{{ kindLabel(artifact.kind) }} · {{ formatDate(artifact.created_at) }}</option>
          </select>
        </label>
        <div v-if="selectedArtifact" class="review-inline-history-content">
          <p v-if="selectedArtifact.stale" class="review-insights-stale">Transcript changed since this result was generated.</p>
          <p v-if="selectedContent.summary" class="review-insights-narrative">{{ selectedContent.summary }}</p>
          <p v-else class="review-insights-empty-text">No narrative summary available.</p>
          <p class="review-inline-provenance">{{ selectedActionItems.length }} actions · {{ selectedDecisionItems.length }} decisions · {{ selectedOpenItems.length }} open questions</p>
        </div>
      </details>
      <details v-if="actions.length || decisions.length" class="review-inline-details">
        <summary>Review action items and decisions ({{ actions.length + decisions.length }})</summary>
        <div v-if="actions.length || decisions.length" class="review-grid">
          <div class="content-card review-card"><div class="card-header"><h3>Action tracker</h3><span class="count-pill">{{ actions.length }}</span></div>
            <div v-for="item in actions" :key="item.id" class="review-item">
              <strong>{{ item.title }}</strong><p>{{ item.owner || 'Unassigned' }} · {{ item.due_date || 'No due date' }} · {{ evidenceText(item.evidence) }}</p>
              <div class="review-footer"><span class="status-pill">{{ item.status }}</span><span class="button-cluster">
                <button v-if="item.status === 'proposed'" type="button" class="ui-btn ui-btn-mini" :disabled="!!busyKind" @click="review('actions', item, 'approved')">Approve</button>
                <button v-if="item.status !== 'done'" type="button" class="ui-btn ui-btn-mini ui-btn-solid" :disabled="!!busyKind" @click="review('actions', item, 'done')">Mark done</button>
              </span></div>
            </div>
          </div>
          <div class="content-card review-card"><div class="card-header"><h3>Decision tracker</h3><span class="count-pill">{{ decisions.length }}</span></div>
            <div v-for="item in decisions" :key="item.id" class="review-item">
              <strong>{{ item.title }}</strong><p>{{ evidenceText(item.evidence) }}</p>
              <div class="review-footer"><span class="status-pill">{{ item.status }}</span><span class="button-cluster">
                <button v-if="item.status === 'proposed'" type="button" class="ui-btn ui-btn-mini ui-btn-solid" :disabled="!!busyKind" @click="review('decisions', item, 'confirmed')">Confirm</button>
                <button v-if="item.status !== 'rejected'" type="button" class="ui-btn ui-btn-mini" :disabled="!!busyKind" @click="review('decisions', item, 'rejected')">Reject</button>
              </span></div>
            </div>
          </div>
        </div>
      </details>
      <details v-if="jobs.length" class="review-inline-details">
        <summary :title="[provider, modelName].filter(Boolean).join(' · ')">
          Tasks<span v-if="activeJobs.length"> ({{ activeJobs.length }} active)</span>
        </summary>
          <aside class="content-card activity-card" aria-label="Background task activity">
            <div v-if="!jobs.length" class="empty-card compact-empty"><strong>No tasks yet</strong></div>
            <div v-else class="job-list">
              <div v-for="job in recentJobs" :key="job.id" class="job-item">
                <span :class="['job-indicator', `is-${job.status}`]" aria-hidden="true">
                  <AppIcon v-if="job.status === 'completed'" name="check" :size="13" />
                  <AppIcon v-else-if="job.status === 'failed'" name="alert" :size="13" />
                  <AppIcon v-else name="clock" :size="13" />
                </span>
                <div class="job-main"><strong>{{ kindLabel(job.kind) }}</strong>
                  <div class="job-sub"><span :class="['job-state', `text-${job.status}`]">{{ job.status }}</span><span v-if="job.created_at">· {{ formatDate(job.created_at) }}</span></div>
                  <p v-if="job.error && job.status === 'failed'" class="job-error" :title="job.error">{{ job.error }}</p>
                </div>
                <button v-if="job.status === 'queued'" type="button" class="job-cancel" :disabled="!!busyKind" @click="cancelJob(job)">Cancel</button>
              </div>
            </div>
            <button v-if="jobs.length > 5" type="button" class="text-button view-all" @click="showAllJobs = !showAllJobs">{{ showAllJobs ? 'Show fewer tasks' : `View more (${jobs.length})` }}</button>
          </aside>
      </details>
    </template>
    </template>
    <p v-if="view !== 'overview' && loading" class="review-insights-hint" role="status">Checking AI availability...</p>
    <div v-else-if="view !== 'overview' && !aiEnabled" class="review-insights-empty">
      <strong>AI worker unavailable</strong>
      <p>Enable the local AI worker to use this tool. Existing transcripts remain accessible.</p>
    </div>
    <template v-else-if="view === 'translation'">
      <div v-if="error" class="review-insights-error" role="alert">{{ error }}</div>
      <div class="workspace-body">
        <div class="section-heading"><div><h3>Bilingual transcript</h3><p>Translate the saved meeting. Results are stored separately from the source transcript.</p></div></div>
        <div class="content-card translation-card">
          <div class="translation-toolbar">
            <label class="select-label">Target language <select v-model="language" class="ui-select" aria-label="Translation language">
              <option v-for="entry in PHASE1_LANGUAGES" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
            </select></label>
            <button type="button" class="ui-btn ui-btn-solid"
              :disabled="translationLoading || translationSubmitting || translationPending || !translationRows.length || translatedCount === translationRows.length"
              @click="queueTranslations">
              {{ translationSubmitting ? 'Queueing…' : translationPending ? 'Translating…' : translationFailed ? 'Retry remaining' : 'Translate saved meeting' }}
            </button>
          </div>
          <p v-if="translationError" class="job-error" role="alert">{{ translationError }}</p>
          <p v-else-if="translationFailed && !translationPending && translatedCount < translationRows.length" class="job-error" role="alert">
            {{ translation?.last_error || 'Some batches did not finish. Retry the remaining segments.' }}
          </p>
          <p v-if="translationRows.length" class="card-description" role="status">
            {{ translatedCount }} / {{ translationRows.length }} segments translated.
            <template v-if="translationPending"> Background work waits for active recording to finish.</template>
          </p>
          <div v-if="!translationRows.length" class="empty-card">
            <strong>{{ translationLoading ? 'Loading saved transcript…' : 'No saved speech yet' }}</strong>
            <p v-if="!translationLoading && !translationError">Save a meeting with transcript content to translate it.</p>
          </div>
          <div v-else class="translation-list"><div class="translation-head"><span>Original</span><span>Translation</span></div>
            <div v-for="seg in translationRows" :key="seg.id" class="translation-entry"><div><span class="segment-time">{{ formatTime(seg.start_ms) }}</span><p>{{ seg.text }}</p></div>
              <div><span :class="['translation-status', { ready: !!translated[seg.id] }]">{{ translated[seg.id] ? 'Translated' : 'Pending' }}</span><p :class="{ 'placeholder-copy': !translated[seg.id] }">{{ translated[seg.id] || 'Waiting for translation' }}</p></div></div>
          </div>
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.intelligence-panel {
  --ai-bg: var(--mop-surface, #fff);
  --ai-paper: var(--mop-paper, #fffdf9);
  --ai-text: var(--mop-text-strong, #261d17);
  --ai-muted: var(--mop-text-muted, #6e6259);
  --ai-border: var(--mop-border, rgba(32, 26, 22, .12));
  --ai-primary: var(--mop-primary, #da751d);
  --ai-primary-dark: var(--mop-primary-strong, #b85d10);
  --ai-teal: var(--mop-secondary, #167c80);
  margin: 20px 0 26px;
  border: 1px solid var(--ai-border);
  border-radius: 17px;
  background: var(--ai-bg);
  box-shadow: 0 8px 30px rgba(42, 29, 17, .04);
  color: var(--ai-text);
  overflow: hidden;
  font-size: 13px;
  line-height: 1.5;
}
.intelligence-panel * { box-sizing: border-box; }
.intelligence-panel button,.intelligence-panel input,.intelligence-panel select { font: inherit; }
.intelligence-panel button { cursor: pointer; }
.intelligence-panel button:disabled { cursor: not-allowed; opacity: .5; }
.intelligence-panel button:focus-visible,.intelligence-panel input:focus-visible,.intelligence-panel select:focus-visible { outline: 3px solid rgba(218,117,29,.28); outline-offset: 2px; }
.workspace-head { display:flex; justify-content:space-between; gap:20px; align-items:center; padding:24px 26px 12px; background:linear-gradient(115deg,#fffcf7, var(--ai-bg) 70%); }
.workspace-identity { display:flex; gap:14px; align-items:center; min-width:0; }
.workspace-emblem { display:flex; width:47px; height:47px; flex-shrink:0; border-radius:14px; align-items:center; justify-content:center; background:linear-gradient(140deg,#e98b32,#c46214); color:#fff; box-shadow:0 7px 18px rgba(218,117,29,.2); }
.workspace-heading { min-width:0; }
.eyebrow-small { display:block; color:var(--ai-primary-dark); font-size:10px; font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
.workspace-heading h2 { margin:2px 0 0; font-size:23px; line-height:1.2; font-weight:800; letter-spacing:-.4px; }
.workspace-heading p { margin:5px 0 0; font-size:12px; color:var(--ai-muted); }
.workspace-tools { display:flex; gap:10px; align-items:center; flex-shrink:0; }
.availability { display:inline-flex; align-items:center; gap:7px; font-size:11px; font-weight:700; color:#277243; }
.availability.offline { color:#a85c17; }
.availability-dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
.workspace-meta { display:flex; align-items:center; flex-wrap:wrap; gap:11px; padding:3px 26px 19px 87px; color:var(--ai-muted); font-size:11px; }
.model-pill { display:inline-flex; align-items:center; gap:6px; padding:3px 8px; border-radius:30px; background:rgba(22,124,128,.08); color:var(--ai-teal); font-weight:650; text-transform:capitalize; }
.model-dot { height:6px; width:6px; background:var(--ai-teal); border-radius:50%; }
.meta-divider { height:11px; border-left:1px solid var(--ai-border); }
.workspace-nav { display:flex; gap:4px; padding:0 26px; border-top:1px solid var(--ai-border); border-bottom:1px solid var(--ai-border); background:#fffdfa; overflow:auto; }
.nav-tab { position:relative; display:inline-flex; flex-shrink:0; align-items:center; min-height:47px; padding:0 16px; border:0; color:var(--ai-muted); background:transparent; font-size:12.5px; font-weight:650; }
.nav-tab:hover { color:var(--ai-primary-dark); }
.nav-tab.selected { color:var(--ai-primary-dark); }
.nav-tab.selected:after { content:''; position:absolute; height:3px; bottom:0; left:14px; right:14px; border-radius:4px 4px 0 0; background:var(--ai-primary); }
.workspace-body { padding:25px 26px 28px; }
.section-heading { display:flex; align-items:end; justify-content:space-between; gap:12px; margin-bottom:13px; }
.section-heading h3 { font-size:15px; font-weight:800; margin:0; }
.section-heading p { margin:3px 0 0; color:var(--ai-muted); font-size:12px; }
.action-heading { margin-top:25px; }
.mode-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }
.mode-card { position:relative; display:flex; flex-direction:column; align-items:flex-start; gap:2px; min-width:0; padding:10px 13px; text-align:left; border:1px solid var(--ai-border); background:var(--ai-bg); border-radius:10px; transition:background .15s,border-color .15s,box-shadow .15s; }
.mode-card:hover { border-color:rgba(218,117,29,.42); background:#fffcf8; }
.mode-card.chosen { background:#fff8ef; border-color:#db9c62; box-shadow:inset 0 0 0 1px rgba(218,117,29,.13); }
.mode-name { font-weight:750; font-size:12.5px; }
.mode-description { color:var(--ai-muted); font-size:10.5px; }
.mode-check { position:absolute; right:10px; top:9px; color:var(--ai-primary-dark); }
.action-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.action-card { display:flex; align-items:center; gap:12px; min-width:0; padding:15px 13px; border:1px solid var(--ai-border); border-radius:12px; background:#fff; text-align:left; color:var(--ai-text); transition:border-color .15s,box-shadow .15s,transform .15s; }
.action-card:not(:disabled):hover { transform:translateY(-1px); border-color:rgba(218,117,29,.38); box-shadow:0 7px 19px rgba(91,54,21,.09); }
.action-icon { display:flex; flex:none; align-items:center; justify-content:center; width:36px; height:36px; border-radius:11px; background:rgba(218,117,29,.09); color:var(--ai-primary-dark); }
.action-primary { background:linear-gradient(130deg,#fffbf4,#fff 90%); border-color:#edd8bd; }
.action-primary .action-icon { background:#e27b22; color:#fff; }
.action-content { display:flex; flex:1; min-width:0; flex-direction:column; gap:3px; }
.action-content strong { font-size:12px; line-height:1.3; font-weight:760; }
.action-content small { font-size:10.5px; line-height:1.35; color:var(--ai-muted); }
.action-arrow { color:#ad8a69; font-size:15px; align-self:flex-start; }
.inline-hint { color:#9b5a1b; margin:9px 0 0; font-size:11px; }
.overview-grid { display:grid; grid-template-columns:minmax(0,1.65fr) minmax(245px,1fr); align-items:start; gap:14px; margin-top:26px; }
.content-card { min-width:0; overflow:hidden; border:1px solid var(--ai-border); border-radius:12px; background:#fff; }
.card-header { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:17px 18px 14px; }
.card-header h3 { margin:2px 0 0; font-size:15px; font-weight:800; }
.count-pill { padding:4px 8px; background:#f4f2ee; color:var(--ai-muted); border-radius:30px; font-size:10.5px; white-space:nowrap; font-weight:650; }
.empty-card { display:flex; align-items:center; flex-direction:column; text-align:center; justify-content:center; padding:38px 20px; color:var(--ai-muted); }
.empty-card strong { color:var(--ai-text); font-size:13px; margin-top:6px; }
.empty-card p { font-size:11.5px; max-width:265px; margin:5px 0 0; }
.empty-illustration { width:46px; height:46px; display:flex; align-items:center; justify-content:center; color:var(--ai-primary); border-radius:13px; background:rgba(218,117,29,.08); }
.compact-empty { padding:26px 19px; }
.artifact-switcher { display:flex; gap:6px; padding:0 17px 14px; overflow-x:auto; }
.artifact-choice { display:inline-flex; gap:8px; align-items:center; flex-shrink:0; padding:6px 9px; font-size:11px; color:var(--ai-muted); border:1px solid var(--ai-border); border-radius:7px; background:#fff; }
.artifact-choice span { font-size:10px; opacity:.65; }
.artifact-choice.active { border-color:rgba(218,117,29,.48); background:#fff6ec; color:var(--ai-primary-dark); font-weight:750; }
.artifact-body { padding:0 18px 18px; }
.artifact-topline { display:flex; gap:8px; justify-content:space-between; align-items:center; margin-bottom:8px; }
.type-label { color:var(--ai-muted); font-size:11px; text-transform:capitalize; }
.summary-text { white-space:pre-wrap; overflow-wrap:anywhere; margin:0; font-size:13px; line-height:1.9; color:var(--ai-text); }
.empty-inline { color:var(--ai-muted); font-size:12px; }
.insights-summary { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:18px; padding:14px 11px; border-radius:10px; background:#faf7f2; }
.insights-summary div { display:flex; flex-direction:column; align-items:center; gap:2px; }
.insights-summary strong { font-size:20px; font-weight:800; }
.insights-summary span { font-size:10px; color:var(--ai-muted); }
.finding-section { margin-top:18px; }
.finding-section h4 { margin:0 0 9px; font-size:12.5px; }
.finding { display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-top:1px solid var(--ai-border); }
.finding-marker { display:flex; flex:none; justify-content:center; align-items:center; width:24px; height:24px; border-radius:8px; background:#fff0df; color:var(--ai-primary-dark); font-size:11px; font-weight:800; }
.marker-teal { color:var(--ai-teal); background:#e8f5f4; }
.marker-muted { color:#76685b; background:#f0eeeb; }
.finding div { min-width:0; }
.finding strong { display:block; font-weight:650; font-size:12px; overflow-wrap:anywhere; }
.finding p,.finding small { display:block; margin:3px 0 0; color:var(--ai-muted); font-size:10.5px; }
.artifact-provenance { margin:20px 0 0; padding-top:9px; border-top:1px solid var(--ai-border); color:var(--ai-muted); font-size:10px; overflow-wrap:anywhere; }
.job-list { padding:0 15px 4px; }
.job-item { display:flex; align-items:flex-start; gap:10px; padding:13px 2px; border-top:1px solid var(--ai-border); }
.job-indicator { display:flex; flex:none; align-items:center; justify-content:center; margin-top:1px; width:25px; height:25px; border-radius:50%; color:#897a69; background:#f1f0ee; }
.job-indicator.is-completed { background:#e8f4eb; color:#277b42; }
.job-indicator.is-running { color:#bb6b14; background:#fff2df; }
.job-indicator.is-failed { color:#b3261e; background:#fceceb; }
.job-main { flex:1; min-width:0; }
.job-main strong { font-size:11.5px; font-weight:700; }
.job-sub { display:flex; gap:5px; align-items:center; color:var(--ai-muted); font-size:10px; margin-top:2px; }
.job-state { text-transform:capitalize; font-weight:700; }
.text-running,.text-queued { color:#ae691d; }
.text-completed { color:#277b42; }
.text-failed { color:#b3261e; }
.job-error { color:#b3261e; font-size:11px; line-height:1.45; overflow-wrap:anywhere; margin:7px 0 0; }
.job-cancel { flex:none; margin-left:auto; background:transparent; padding:4px; border:0; color:var(--ai-muted); font-size:10px; }
.job-cancel:hover { color:var(--ai-primary-dark); }
.text-button { background:transparent; border:0; color:var(--ai-primary-dark); font-weight:650; font-size:11px; padding:9px; }
.view-all { width:100%; text-align:center; border-top:1px solid var(--ai-border); }
.review-grid { margin-top:14px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.review-item { padding:13px 18px; border-top:1px solid var(--ai-border); }
.review-item strong { font-size:12px; font-weight:670; }
.review-item p { font-size:10.5px; color:var(--ai-muted); margin:4px 0 8px; }
.review-footer { display:flex; gap:10px; justify-content:space-between; align-items:center; }
.button-cluster { display:flex; gap:6px; }
.status-pill { display:inline-flex; padding:3px 8px; border-radius:999px; font-weight:650; font-size:10px; text-transform:capitalize; background:#f1f0ee; color:#766b5f; }
.status-completed { background:#e8f4eb; color:#267641; }
.status-running,.status-queued { background:#fff2df; color:#9c5915; }
.status-failed { background:#fceceb; color:#a7251f; }
.ui-btn { display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:8px 12px; border:1px solid var(--ai-border); border-radius:8px; font-size:11.5px; font-weight:700; color:var(--ai-text); background:#fff; transition:background .15s,border-color .15s; }
.ui-btn:hover:not(:disabled) { background:#fff9f1; border-color:rgba(218,117,29,.4); }
.ui-btn-quiet { color:var(--ai-muted); }
.ui-btn-solid { background:var(--ai-primary); border-color:var(--ai-primary); color:#fff; }
.ui-btn-solid:hover:not(:disabled) { background:var(--ai-primary-dark); color:#fff; }
.ui-btn-mini { font-size:10px; padding:5px 9px; }
.feedback { display:flex; align-items:center; gap:9px; padding:10px 14px; margin:0 26px 14px; border-radius:9px; font-size:11.5px; line-height:1.4; overflow-wrap:anywhere; }
.feedback span { flex:1; }
.feedback-info { background:#eef7f6; color:#13686b; }
.feedback-error { background:#fff0ee; color:#aa291e; }
.feedback-close { border:0; background:transparent; color:inherit; font-size:20px; line-height:1; }
.disabled-state { margin:18px 26px 26px; display:flex; gap:13px; padding:16px; border:1px solid #e9cba2; border-radius:10px; background:#fff9ef; }
.disabled-state strong { font-size:12px; }
.disabled-state p { margin:4px 0 0; color:var(--ai-muted); font-size:11px; }
.card-description { margin:0 18px 12px; font-size:11px; color:var(--ai-muted); }
.ui-select { width:100%; min-width:0; padding:10px 12px; border:1px solid #dcd6cc; border-radius:8px; background:#fff; color:var(--ai-text); font-size:12px; }
.translation-card { padding:17px 18px; }
.translation-toolbar { display:flex; align-items:center; gap:18px; flex-wrap:wrap; margin-bottom:14px; }
.select-label { display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:650; }
.select-label { margin-left:auto; color:var(--ai-muted); }
.ui-select { width:auto; padding:7px 10px; }
.translation-head,.translation-entry { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; }
.translation-head { background:#faf8f4; padding:11px 14px; font-size:11px; font-weight:750; color:var(--ai-muted); }
.translation-entry { border-bottom:1px solid var(--ai-border); padding:13px 14px; }
.translation-entry p { font-size:12px; margin:6px 0 0; line-height:1.75; white-space:pre-wrap; overflow-wrap:anywhere; }
.segment-time { font-size:10px; color:var(--ai-muted); }
.translation-status { font-size:10px; color:#a5743e; }
.translation-status.ready { color:var(--ai-teal); }
.placeholder-copy { color:var(--ai-muted); font-style:italic; }
.visually-hidden { position:absolute; height:1px; width:1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; }
@media (max-width:820px) {
  .workspace-head { padding:20px 18px 12px; }
  .workspace-meta { padding:4px 18px 17px; }
  .workspace-nav { padding:0 13px; }
  .workspace-body { padding:20px 18px 24px; }
  .overview-grid { grid-template-columns:minmax(0,1fr); }
  .feedback { margin-left:18px; margin-right:18px; }
}
@media (max-width:640px) {
  .workspace-head { flex-direction:column; align-items:stretch; gap:13px; }
  .workspace-tools { justify-content:space-between; }
  .workspace-heading h2 { font-size:20px; }
  .mode-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .action-grid,.review-grid { grid-template-columns:1fr; }
  .action-card { padding:12px 13px; }
  .workspace-body { padding:17px 13px 22px; }
  .workspace-nav { padding:0 4px; }
  .nav-tab { padding:0 12px; }
  .translation-head,.translation-entry { grid-template-columns:1fr; gap:6px; }
  .translation-head span:last-child { display:none; }
  .translation-toolbar { gap:12px; }
  .select-label { margin-left:0; }
}
@media (prefers-reduced-motion:reduce) { .intelligence-panel *, .intelligence-panel *::before, .intelligence-panel *::after { transition-duration:0s !important; animation-duration:0s !important; } }

/* Workspace 2.0 visual integration */
.intelligence-panel{--ai-primary:#eb7100;--ai-primary-dark:#a74f00;--ai-teal:#ab5100;--ai-border:#e8e6e2;margin:0 0 15px;border-radius:15px;background:#fff;box-shadow:none}
.workspace-head{background:#fff;padding:19px 20px 13px}
.workspace-emblem{background:#eb7100;box-shadow:none;width:39px;height:39px;border-radius:11px}
.workspace-heading h2{font-size:18px;font-weight:780;letter-spacing:-.03em}
.workspace-heading p{font-size:11px}
.workspace-meta{padding:0 20px 15px;gap:8px}
.model-pill{background:#fff2e6;color:#a75100}
.workspace-nav{background:#fff;padding:0 13px}
.nav-tab{min-height:43px;padding:0 10px;font-size:11px}
.nav-tab.selected:after{background:#eb7100}
.workspace-body{padding:17px}
.mode-card,.action-card,.content-card{border-color:#e8e6e2;border-radius:11px}
.mode-card:hover,.mode-card.chosen{border-color:#f3c89e;background:#fff8f1}
.action-card:not(:disabled):hover{border-color:#f1c396;box-shadow:0 4px 14px rgba(55,34,11,.06)}
.action-primary{background:#fff9f1;border-color:#efd3b2}
.action-primary .action-icon{background:#eb7100}
.action-icon{background:#fff2e5;color:#a45100}
.workspace-heading,.section-heading{color:#292b30}


/* Compact meeting review: a direct output, not a second nested application. */
.review-insights{display:flex;flex-direction:column;gap:15px;min-width:0;color:#292b30}
.review-insights-header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.review-insights-header h3{margin:0;font-size:16px;letter-spacing:-.025em;font-weight:770}
.review-insights-header p{margin:6px 0 0;font-size:12px;line-height:1.6;color:#777981}
.review-insights-actions{display:flex;flex-wrap:wrap;gap:8px}
.review-insights-actions button{display:inline-flex;align-items:center;justify-content:center;gap:7px;min-height:39px;padding:9px 12px;border-radius:9px;font-size:12px;font-weight:700;flex:1 1 155px}
.review-insights-primary{background:#a94f00;border:1px solid #a94f00;color:#fff}
.review-insights-primary:hover:not(:disabled){background:#894000}
.review-insights-secondary{background:#fff;border:1px solid #e7e3dc;color:#695648}
.review-insights-secondary:hover:not(:disabled){background:#fff7f0}
.review-insights-actions button:disabled{opacity:.52;cursor:not-allowed}
.review-insights-section{border:1px solid #e9e7e3;border-radius:12px;background:#fff;padding:15px}
.review-insights-section h4{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 12px;font-size:13px;color:#282a30;font-weight:770}
.review-insights-section h4 span{display:flex;align-items:center;justify-content:center;min-width:21px;padding:2px 6px;border-radius:7px;background:#fff1e4;color:#a5530b;font-size:10px}
.review-insights-narrative{margin:0;line-height:1.75;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#3c4047}
.review-insights-empty-text,.review-insights-hint{font-size:11px;color:#86868c;line-height:1.65;margin:0}
.review-insights-stale{display:block;margin-top:10px;font-size:11px;color:#a75b1b}
.review-insights-list{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:11px}
.review-insights-list li{display:flex;flex-direction:column;gap:3px;padding:0 0 10px;border-bottom:1px solid #efeeeb}
.review-insights-list li:last-child{border:0;padding-bottom:0}
.review-insights-list strong{font-size:12px;font-weight:680;color:#383b42;line-height:1.55}
.review-insights-list small{font-size:10px;color:#8a8b91;line-height:1.6}
.review-insights-error{padding:11px 12px;border:1px solid #f0d0bd;background:#fff8f4;border-radius:9px;font-size:11px;line-height:1.6;color:#a34420}
.review-insights-empty{padding:20px;border:1px dashed #e9dbca;background:#fffbf7;border-radius:12px}
.review-insights-empty strong{font-size:13px;color:#4b3c32}
.review-insights-empty p{font-size:11px;line-height:1.6;color:#777981;margin:6px 0 0}
.review-insights-advanced{display:flex;align-items:center;justify-content:space-between;gap:12px;width:100%;padding:12px 2px;border:0;border-top:1px solid #ece9e4;background:transparent;color:#8c552b;text-align:left;font-size:11px;font-weight:730}
.review-insights-advanced:hover{color:#a44e00}


/* Integrated Assistant sections with streamlined inline controls. */
.intelligence-inline {
  --ai-bg:#fff;
  --ai-paper:#fcfbf9;
  --ai-text:#302a25;
  --ai-muted:#83776c;
  --ai-border:#e9e2da;
  --ai-primary:#eb7100;
  --ai-primary-dark:#a44906;
  --ai-teal:#995220;
  display:flex;flex-direction:column;gap:13px;min-width:0;
  color:#302a25;font-size:12px;
}
.intelligence-inline * {box-sizing:border-box;}
.intelligence-inline button,.intelligence-inline select,.intelligence-inline input {font:inherit;}
.intelligence-inline button:disabled {cursor:not-allowed;opacity:.55;}
.intelligence-inline .review-insights-header h3 {font-size:14px;}
.intelligence-inline .review-insights-header p {font-size:11px;}
.review-inline-mode {
  display:flex;align-items:center;justify-content:space-between;gap:9px;
  min-width:0;color:#73695f;font-size:11px;font-weight:650;
}
.review-inline-mode select {
  min-width:0;max-width:175px;min-height:31px;flex:1;
  padding:5px 8px;border:1px solid #e6ddd3;border-radius:8px;
  background:#fff;color:#463b31;
}
.review-inline-provenance {
  margin:3px 0 0;color:#978b7e;line-height:1.6;
  font-size:10.5px;overflow-wrap:anywhere;
}
.review-inline-details {min-width:0;border-top:1px solid #eee9e3;padding-top:11px;}
.review-inline-details>summary {
  display:flex;justify-content:space-between;gap:8px;
  color:#62574c;font-size:11px;font-weight:720;cursor:pointer;
  list-style:none;padding:5px 0 10px;
}
.review-inline-details>summary::-webkit-details-marker {display:none;}
.review-inline-details>summary:after {content:"⌄";margin-left:auto;color:#a24c14;font-size:15px;}
.review-inline-details[open]>summary:after {content:"⌃";}
.review-inline-history-content {padding:8px 0;}
.review-inline-index {
  display:flex;justify-content:space-between;align-items:center;gap:9px;
  padding:8px 0 12px;border-bottom:1px solid #ede8e2;
  color:#84776a;font-size:11px;
}
.review-inline-index .ui-btn {flex:none;white-space:nowrap;}
.intelligence-inline .workspace-body {min-width:0;padding:0;background:transparent;}
.intelligence-inline .section-heading {margin:0 0 11px;align-items:stretch;}
.intelligence-inline .section-heading h3 {font-size:13px;font-weight:760;}
.intelligence-inline .section-heading p {font-size:11px;line-height:1.55;}
.intelligence-inline .review-grid,
.intelligence-inline .overview-grid {
  display:grid;grid-template-columns:minmax(0,1fr);gap:12px;margin:0;
}
.intelligence-inline .content-card {
  min-width:0;background:#fff;border:1px solid #ede7df;
  border-radius:10px;box-shadow:none;
}
.intelligence-inline .card-header {padding:12px 12px 10px;}
.intelligence-inline .card-header h3 {font-size:12px;}
.intelligence-inline .card-description {margin:0 12px 12px;font-size:11px;line-height:1.55;}
.intelligence-inline .ui-btn {cursor:pointer;}
.intelligence-inline .ui-btn-solid {color:#fff;background:#b85710;border-color:#b85710;}
.intelligence-inline .translation-card {padding:12px;}
.intelligence-inline .translation-toolbar {align-items:flex-start;gap:8px;}
.intelligence-inline .translation-toolbar .select-label {margin:0;flex-wrap:wrap;}
.intelligence-inline .translation-head {display:none;}
.intelligence-inline .translation-entry {
  display:grid;grid-template-columns:minmax(0,1fr);gap:7px;padding:12px 0;
}
.intelligence-inline .translation-entry>div:last-child {
  border-left:2px solid #f0d2b5;padding-left:10px;
}
.intelligence-inline .translation-entry p {font-size:11px;line-height:1.55;}
.intelligence-inline .review-grid {margin-top:8px;}
.intelligence-inline .review-card {min-width:0;}
.intelligence-inline .review-item {padding:11px 12px;}
.intelligence-inline .review-footer {flex-wrap:wrap;}
.intelligence-inline .job-list {padding:0;}
.intelligence-inline .job-item {padding:11px 0;}
.intelligence-inline .activity-card {border:0;background:transparent;}
.intelligence-inline .empty-card {padding:16px 12px;}
.intelligence-inline .view-all {padding:10px;}
.intelligence-inline .review-insights-section {background:transparent;padding:12px 0;border:0;border-bottom:1px solid #eeeae5;border-radius:0;}
.intelligence-inline .review-insights-actions button {min-height:34px;padding:8px 9px;}
@media(max-width:560px) {
  .review-inline-mode {flex-wrap:wrap;}
  .review-inline-mode select {max-width:none;}
  .intelligence-inline .review-insights-actions button {flex:1 1 100%;}
}

/* One compact command row, with evidence below it. */
.intelligence-inline .review-insights-controls {
  display:grid;grid-template-columns:minmax(104px,.95fr) minmax(148px,1.05fr);
  align-items:center;gap:8px;min-width:0;
}
.intelligence-inline .review-insights-controls .review-inline-mode {
  height:36px;min-height:36px;gap:7px;white-space:nowrap;
}
.intelligence-inline .review-insights-controls .review-inline-mode select {
  flex:1;min-width:0;max-width:none;height:36px;min-height:36px;
  padding:0 22px 0 7px;font-size:11px;
}
.intelligence-inline .review-insights-controls .review-insights-primary {
  display:flex;align-items:center;justify-content:center;gap:6px;
  height:36px;min-height:36px;min-width:0;width:100%;
  padding:0 7px;border-radius:9px;
  font-size:11px;font-weight:700;white-space:nowrap;
}
.intelligence-inline .review-insights-section {
  padding:11px 0;border-bottom:1px solid #eeeae5;
}
.intelligence-inline .review-insights-section h4 {
  margin:0 0 7px;font-size:12px;
}
@media(max-width:340px){
  .intelligence-inline .review-insights-controls {grid-template-columns:minmax(0,1fr);}
}

/* Assistant overview: aligned controls and a single editorial results surface. */
.intelligence-inline.review-insights {gap:13px;}
.intelligence-inline.review-insights .review-insights-controls {
  display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);
  align-items:center;gap:10px;margin:0 0 3px;
}
.intelligence-inline.review-insights .review-insights-controls .review-inline-mode {
  display:flex;width:100%;min-width:0;height:40px;min-height:40px;margin:0;
}
.intelligence-inline.review-insights .review-insights-controls .review-inline-mode select {
  display:block;width:100%;height:40px;min-height:40px;
  padding:0 30px 0 13px;border:1px solid #e5ddd3;
  border-radius:9px;background-color:#fff;
  color:#332d28;font-size:12px;font-weight:610;
}
.intelligence-inline.review-insights .review-insights-controls .review-insights-primary {
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
  width:100%;min-width:0;height:40px;min-height:40px;
  padding:0 10px;border-radius:9px;font-size:12px;font-weight:680;
}
.intelligence-inline.review-insights .review-insights-results {
  min-width:0;overflow:hidden;background:#fff;
  border:1px solid #e9e4de;border-radius:13px;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-section {
  display:block;margin:0;padding:18px 19px 19px;
  border:0;border-bottom:1px solid #eeeae6;border-radius:0;
  background:#fff;box-shadow:none;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-section:last-child {
  border-bottom:0;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-section h4 {
  display:flex;align-items:center;justify-content:flex-start;
  gap:8px;margin:0 0 10px;color:#292724;
  font-size:12px;font-weight:760;line-height:1.3;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-section h4:before {
  content:none;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-section h4 span {
  margin-left:auto;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-narrative,
.intelligence-inline.review-insights .review-insights-results .review-insights-list {
  margin:0;color:#4f4841;line-height:1.75;font-size:12px;
}
.intelligence-inline.review-insights .review-insights-results .review-insights-empty-text {
  margin:0;color:#9b9289;font-size:11px;line-height:1.55;
}
.intelligence-inline.review-insights .review-inline-details {
  margin-top:0;padding:11px 2px 0;
}
@media(max-width:340px){
  .intelligence-inline.review-insights .review-insights-controls {grid-template-columns:minmax(0,1fr);}
}
</style>
