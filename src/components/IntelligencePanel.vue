<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AppIcon from './AppIcon.vue'
import { PHASE1_LANGUAGES } from '@/config/phase1Languages'

type Mode = 'meeting' | 'research' | 'classroom' | 'interview' | 'standup' | 'sales'
type View = 'overview' | 'knowledge' | 'translation' | 'media'
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
interface SearchResult {
  text: string
  source_id: string
  source_kind: string
  evidence: Record<string, unknown>
}
interface TranslatedFragment {
  segment_id: number
  source_text: string
  text: string
}
interface TranslationJobResult {
  target?: string
  translations?: TranslatedFragment[]
}
interface AskResult {
  answer?: string
  sources?: Array<SearchResult>
}

const props = defineProps<{ meetingId: string; segments: Segment[] }>()
const modes: Array<{ id: Mode; name: string; description: string }> = [
  { id: 'meeting', name: 'Meeting', description: 'Summaries and decisions' },
  { id: 'research', name: 'Research', description: 'Experiments and findings' },
  { id: 'classroom', name: 'Classroom', description: 'Lectures and study notes' },
  { id: 'interview', name: 'Interview', description: 'Questions and answers' },
  { id: 'standup', name: 'Standup', description: 'Progress and blockers' },
  { id: 'sales', name: 'Sales', description: 'Client discussions' },
]
const views: Array<{ id: View; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'translation', label: 'Translation' },
  { id: 'media', label: 'Media' },
]
const mode = ref<Mode>('meeting')
const view = ref<View>('overview')
const language = ref('en')
const question = ref('')
const searchText = ref('')
const notice = ref('')
const error = ref('')
const translationEnabled = ref(false)
const jobs = ref<Job[]>([])
const artifacts = ref<Artifact[]>([])
const actions = ref<Review[]>([])
const decisions = ref<Review[]>([])
const searchResults = ref<SearchResult[]>([])
const translated = ref<Record<number, string>>({})
const selectedArtifactId = ref<string | null>(null)
const askJobId = ref<string | null>(null)
const askJob = ref<Job | null>(null)
const mediaJobId = ref<string | null>(null)
const mediaJob = ref<Job | null>(null)
const mediaName = ref('')
const mediaInput = ref<HTMLInputElement | null>(null)
const busyKind = ref<string | null>(null)
const refreshing = ref(false)
const searching = ref(false)
const aiEnabled = ref(false)
const loading = ref(true)
const provider = ref('')
const modelName = ref('')
const showAllJobs = ref(false)
const queuedTranslation = new Set<string>()
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
    analyze_media: 'Media analysis', index_media: 'Media index' } as Record<string, string>)[kind] ?? kind.replace(/_/g, ' ')
}
function sourceLabel(hit: SearchResult): string {
  const ms = hit.evidence?.start_ms
  return `${hit.source_kind === 'meeting' ? 'Meeting' : 'Media'} · ${typeof ms === 'number' ? formatTime(ms) : hit.source_id.slice(0, 12)}`
}
function evidenceText(ids?: number[]): string {
  return ids?.length ? `Segments ${ids.join(', ')}` : 'No source reference'
}
const latestArtifacts = computed(() => artifacts.value.slice(0, 15))
const selectedArtifact = computed(() => latestArtifacts.value.find(a => a.id === selectedArtifactId.value) ?? latestArtifacts.value[0] ?? null)
const activeJobs = computed(() => jobs.value.filter(j => j.status === 'queued' || j.status === 'running'))
const recentJobs = computed(() => showAllJobs.value ? jobs.value.slice(0, 15) : jobs.value.slice(0, 5))
const latestCompleted = computed(() => jobs.value.filter(j => j.status === 'completed').length)
const currentMode = computed(() => modes.find(m => m.id === mode.value) ?? modes[0]!)
const hasSegments = computed(() => props.segments.some(s => s.text.trim().length > 0))
const translationRows = computed(() => props.segments.filter(s => s.text.trim()).slice(-12))
const selectedContent = computed<ArtifactContent>(() => selectedArtifact.value?.content ?? {})
const selectedActionItems = computed(() => selectedContent.value.actions ?? [])
const selectedDecisionItems = computed(() => selectedContent.value.decisions ?? [])
const selectedOpenItems = computed(() => selectedContent.value.open_questions ?? [])
const currentAskAnswer = computed(() => (askJob.value?.result as AskResult | null)?.answer ?? '')
const currentAskSources = computed(() => (askJob.value?.result as AskResult | null)?.sources ?? [])

async function refresh(silent = false): Promise<void> {
  if (!props.meetingId || refreshPending) return
  refreshPending = true
  if (!silent) refreshing.value = true
  try {
    const meetingId = props.meetingId
    const [overview, activeAsk, activeMedia] = await Promise.all([
      request<{ jobs: Job[]; artifacts: Artifact[]; actions: Review[]; decisions: Review[] }>(`/meetings/${encodeURIComponent(meetingId)}/overview`),
      askJobId.value ? request<Job>(`/jobs/${encodeURIComponent(askJobId.value)}`) : Promise.resolve(null),
      mediaJobId.value ? request<Job>(`/jobs/${encodeURIComponent(mediaJobId.value)}`) : Promise.resolve(null),
    ])
    if (meetingId !== props.meetingId) return
    jobs.value = overview.jobs ?? []
    artifacts.value = overview.artifacts ?? []
    actions.value = overview.actions ?? []
    decisions.value = overview.decisions ?? []
    if (activeAsk) askJob.value = activeAsk
    if (activeMedia) mediaJob.value = activeMedia
    if (translationEnabled.value) {
      await fetchTranslations()
      for (const job of jobs.value) {
        if (job.kind !== 'live_translate' || job.status !== 'completed') continue
        const payload = job.result as TranslationJobResult | null
        if (payload?.target !== language.value) continue
        for (const item of payload.translations ?? []) {
          const original = props.segments.find(s => s.id === item.segment_id)
          if (original?.text === item.source_text) translated.value[item.segment_id] = item.text
        }
      }
    }
    if (!silent) error.value = ''
  } catch (e) {
    error.value = messageFromError(e)
  } finally {
    refreshPending = false
    refreshing.value = false
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
async function analyze(kind: 'summary' | 'extract'): Promise<void> {
  await perform(kind, { mode: mode.value })
}
async function indexMeeting(): Promise<void> { await perform('index_meeting') }
async function ask(): Promise<void> {
  if (!question.value.trim() || busyKind.value) return
  busyKind.value = 'ask'
  clearMessages()
  try {
    const result = await request<{ job_id: string }>('/ask', jsonPost({ question: question.value.trim() }))
    askJobId.value = result.job_id
    askJob.value = { id: result.job_id, kind: 'ask', status: 'queued' }
    notice.value = 'Question submitted. Your answer will appear here when ready.'
    view.value = 'knowledge'
  } catch (e) { error.value = messageFromError(e) }
  finally { busyKind.value = null }
}
async function search(): Promise<void> {
  if (!searchText.value.trim()) return
  searching.value = true
  clearMessages()
  try {
    const data = await request<{ results: SearchResult[] }>('/search', jsonPost({ query: searchText.value.trim(), limit: 12 }))
    searchResults.value = data.results ?? []
  } catch (e) { error.value = messageFromError(e) }
  finally { searching.value = false }
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
async function upload(ev: Event): Promise<void> {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file || busyKind.value) return
  busyKind.value = 'upload'
  clearMessages()
  mediaName.value = file.name
  try {
    const data = await request<{ media_id: string; job_id: string }>('/media/upload', {
      method: 'POST',
      headers: { 'X-File-Name': encodeURIComponent(file.name), 'Content-Type': file.type || 'application/octet-stream' },
      body: file,
    })
    mediaJobId.value = data.job_id
    mediaJob.value = { id: data.job_id, kind: 'analyze_media', status: 'queued' }
    notice.value = `${file.name} uploaded. Analysis queued.`
    view.value = 'media'
  } catch (e) { error.value = messageFromError(e) }
  finally { busyKind.value = null; input.value = '' }
}
async function fetchTranslations(): Promise<void> {
  const data = await request<{ translations: Array<{ segment_id: number; text: string }> }>(
    `/meetings/${encodeURIComponent(props.meetingId)}/translations/${encodeURIComponent(language.value)}`,
  )
  const next: Record<number, string> = {}
  for (const item of data.translations ?? []) next[item.segment_id] = item.text
  translated.value = { ...translated.value, ...next }
}
async function queueTranslations(): Promise<void> {
  if (!translationEnabled.value || busyKind.value) return
  const untranslated = props.segments.filter(s => s.text.trim() && !translated.value[s.id])
  const fresh = untranslated.filter(s => !queuedTranslation.has(`${language.value}/${s.id}/${s.text}`)).slice(0, 8)
  if (!fresh.length) {
    notice.value = 'No new finalized segments to translate.'
    return
  }
  const keys = fresh.map(s => `${language.value}/${s.id}/${s.text}`)
  keys.forEach(k => queuedTranslation.add(k))
  const ok = await perform('live_translate', {
    target: language.value,
    segments: fresh.map(s => ({ id: s.id, text: s.text })),
  })
  if (!ok) keys.forEach(k => queuedTranslation.delete(k))
}
watch(() => props.meetingId, () => {
  jobs.value = []; artifacts.value = []; actions.value = []; decisions.value = []
  selectedArtifactId.value = null
  searchResults.value = []
  askJobId.value = null; askJob.value = null
  mediaJobId.value = null; mediaJob.value = null; mediaName.value = ''
  translated.value = {}; queuedTranslation.clear()
  void refresh()
})
watch(language, () => { translated.value = {}; queuedTranslation.clear(); if (translationEnabled.value) { void fetchTranslations().catch(e => { error.value = messageFromError(e) }); void queueTranslations() } })
watch(() => props.segments.map(s => `${s.id}:${s.text}`).join('|'), () => {
  if (translationEnabled.value) void queueTranslations()
})
watch(translationEnabled, enabled => {
  if (enabled) { void fetchTranslations().catch(e => { error.value = messageFromError(e) }); void queueTranslations() }
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
  pollId = setInterval(() => { if (aiEnabled.value) void refresh(true) }, 4000)
})
onBeforeUnmount(() => { if (pollId) clearInterval(pollId) })
</script>

<template>
  <section class="intelligence-panel" aria-label="AI intelligence workspace">
    <div class="workspace-head">
      <div class="workspace-identity">
        <span class="workspace-emblem" aria-hidden="true"><AppIcon name="file-text" :size="23" /></span>
        <div class="workspace-heading">
          <span class="eyebrow-small">MEETING INTELLIGENCE</span>
          <h2>AI Workspace</h2>
          <p>Turn conversations into decisions, answers, and next steps.</p>
        </div>
      </div>
      <div class="workspace-tools">
        <span class="availability" :class="{ offline: !aiEnabled }">
          <span class="availability-dot" aria-hidden="true"></span>
          {{ loading ? 'Checking' : aiEnabled ? 'AI enabled' : 'AI unavailable' }}
        </span>
        <button type="button" class="ui-btn ui-btn-quiet" :disabled="refreshing || !aiEnabled" @click="refresh()">
          <AppIcon name="clock" :size="15" /> {{ refreshing ? 'Refreshing' : 'Refresh' }}
        </button>
      </div>
    </div>

    <div v-if="aiEnabled" class="workspace-meta">
      <span class="model-pill" :title="modelName"><span class="model-dot"></span>{{ provider || 'Local' }} provider</span>
      <span class="meta-divider" aria-hidden="true"></span>
      <span>{{ segments.length }} transcript segments</span>
      <span class="meta-divider" aria-hidden="true"></span>
      <span>{{ latestCompleted }} completed {{ latestCompleted === 1 ? 'task' : 'tasks' }}</span>
    </div>

    <div v-if="error" role="alert" class="feedback feedback-error">
      <AppIcon name="alert" :size="17" /> <span>{{ error }}</span>
      <button type="button" class="feedback-close" aria-label="Dismiss error" @click="error = ''">×</button>
    </div>
    <div v-if="notice" role="status" class="feedback feedback-info">
      <AppIcon name="check" :size="16" /> <span>{{ notice }}</span>
      <button type="button" class="feedback-close" aria-label="Dismiss notice" @click="notice = ''">×</button>
    </div>

    <div v-if="!loading && !aiEnabled" class="disabled-state">
      <AppIcon name="alert" :size="23" />
      <div><strong>Intelligence is not enabled</strong><p>Enable MEETING_AI_ENABLED=1 in the backend and start the separate AI worker to use this workspace.</p></div>
    </div>

    <template v-if="aiEnabled">
      <nav class="workspace-nav" aria-label="AI workspace sections">
        <button v-for="item in views" :key="item.id" type="button" :class="['nav-tab', { selected: view === item.id }]"
          :aria-current="view === item.id ? 'page' : undefined" @click="view = item.id">{{ item.label }}</button>
      </nav>

      <div v-if="view === 'overview'" class="workspace-body">
        <div class="section-heading">
          <div><h3>Choose a workflow</h3><p>Analysis is generated from saved meeting transcripts.</p></div>
        </div>
        <div class="mode-grid" role="group" aria-label="Analysis mode">
          <button v-for="item in modes" :key="item.id" type="button" :class="['mode-card', { chosen: mode === item.id }]"
            :aria-pressed="mode === item.id" @click="mode = item.id">
            <span class="mode-name">{{ item.name }}</span><span class="mode-description">{{ item.description }}</span>
            <span v-if="mode === item.id" class="mode-check" aria-hidden="true"><AppIcon name="check" :size="12" /></span>
          </button>
        </div>

        <div class="section-heading action-heading">
          <div><h3>Quick actions</h3><p>{{ currentMode.name }} workflow · {{ currentMode.description }}</p></div>
        </div>
        <div class="action-grid">
          <button type="button" class="action-card action-primary" :disabled="!!busyKind || !hasSegments || activeJobs.some(j => j.kind === 'summary')" @click="analyze('summary')">
            <span class="action-icon"><AppIcon name="file-text" :size="19" /></span>
            <span class="action-content"><strong>Generate notes</strong><small>Structured summary and highlights</small></span>
            <span class="action-arrow" aria-hidden="true">↗</span>
          </button>
          <button type="button" class="action-card" :disabled="!!busyKind || !hasSegments || activeJobs.some(j => j.kind === 'extract')" @click="analyze('extract')">
            <span class="action-icon"><AppIcon name="list" :size="19" /></span>
            <span class="action-content"><strong>Extract next steps</strong><small>Action items and decisions</small></span>
            <span class="action-arrow" aria-hidden="true">↗</span>
          </button>
          <button type="button" class="action-card" :disabled="!!busyKind || !hasSegments || activeJobs.some(j => j.kind === 'index_meeting')" @click="indexMeeting">
            <span class="action-icon"><AppIcon name="save" :size="19" /></span>
            <span class="action-content"><strong>Index meeting</strong><small>Make content searchable</small></span>
            <span class="action-arrow" aria-hidden="true">↗</span>
          </button>
        </div>
        <p v-if="!hasSegments" class="inline-hint">Save a meeting with transcript content to enable analysis.</p>

        <div class="overview-grid">
          <div class="content-card insights-card">
            <div class="card-header"><div><span class="eyebrow-small">YOUR OUTPUT</span><h3>Latest insights</h3></div>
              <span v-if="latestArtifacts.length" class="count-pill">{{ latestArtifacts.length }} saved</span>
            </div>
            <div v-if="!latestArtifacts.length" class="empty-card">
              <span class="empty-illustration"><AppIcon name="file-text" :size="25" /></span>
              <strong>Nothing generated yet</strong>
              <p>Choose a workflow and generate notes to see your meeting insights here.</p>
            </div>
            <template v-else>
              <div v-if="latestArtifacts.length > 1" class="artifact-switcher" role="group" aria-label="Saved analysis">
                <button v-for="artifact in latestArtifacts.slice(0, 6)" :key="artifact.id" type="button"
                  :class="['artifact-choice', { active: selectedArtifact?.id === artifact.id }]"
                  @click="selectedArtifactId = artifact.id">
                  {{ kindLabel(artifact.kind) }} <span>{{ formatDate(artifact.created_at) }}</span>
                </button>
              </div>
              <div v-if="selectedArtifact" class="artifact-body">
                <div class="artifact-topline"><span class="type-label">{{ selectedArtifact.mode }} · {{ kindLabel(selectedArtifact.kind) }}</span>
                  <span v-if="selectedArtifact.stale" class="status-pill status-failed">Outdated transcript</span></div>
                <p v-if="selectedContent.summary" class="summary-text">{{ selectedContent.summary }}</p>
                <p v-else class="empty-inline">No narrative summary was returned for this analysis.</p>
                <div class="insights-summary">
                  <div><strong>{{ selectedActionItems.length }}</strong><span>Action items</span></div>
                  <div><strong>{{ selectedDecisionItems.length }}</strong><span>Decisions</span></div>
                  <div><strong>{{ selectedOpenItems.length }}</strong><span>Open questions</span></div>
                </div>
                <div v-if="selectedActionItems.length" class="finding-section">
                  <h4>Action items</h4>
                  <div v-for="(item, i) in selectedActionItems" :key="`action-${i}`" class="finding">
                    <span class="finding-marker">{{ i + 1 }}</span>
                    <div><strong>{{ item.title }}</strong><p>{{ item.owner || 'Owner not assigned' }}<span v-if="item.due_date"> · Due {{ item.due_date }}</span></p><small>{{ evidenceText(item.source_ids) }}</small></div>
                  </div>
                </div>
                <div v-if="selectedDecisionItems.length" class="finding-section">
                  <h4>Decisions</h4>
                  <div v-for="(item, i) in selectedDecisionItems" :key="`decision-${i}`" class="finding">
                    <span class="finding-marker marker-teal"><AppIcon name="check" :size="13" /></span>
                    <div><strong>{{ item.title }}</strong><small>{{ evidenceText(item.source_ids) }}</small></div>
                  </div>
                </div>
                <div v-if="selectedOpenItems.length" class="finding-section">
                  <h4>Open questions</h4>
                  <div v-for="(item, i) in selectedOpenItems" :key="`question-${i}`" class="finding">
                    <span class="finding-marker marker-muted">?</span>
                    <div><strong>{{ item.title }}</strong><small>{{ evidenceText(item.source_ids) }}</small></div>
                  </div>
                </div>
                <p v-if="selectedArtifact.model_name" class="artifact-provenance" :title="selectedArtifact.model_name">Generated by {{ selectedArtifact.model_name }}</p>
                <p v-if="selectedContent.evidence_audit" class="artifact-provenance">
                  Source-linked claims {{ selectedContent.evidence_audit.claims_with_valid_references }}/{{ selectedContent.evidence_audit.claim_count }} - References are not independently fact-checked
                </p>
              </div>
            </template>
          </div>

          <aside class="content-card activity-card" aria-label="Background task activity">
            <div class="card-header"><div><span class="eyebrow-small">BACKGROUND WORK</span><h3>Task activity</h3></div>
              <span class="count-pill">{{ activeJobs.length }} active</span></div>
            <div v-if="!jobs.length" class="empty-card compact-empty"><strong>No tasks yet</strong><p>New analysis jobs will appear here.</p></div>
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
        </div>

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
      </div>

      <div v-else-if="view === 'knowledge'" class="workspace-body">
        <div class="section-heading"><div><h3>Knowledge center</h3><p>Find evidence across saved meetings and indexed media.</p></div></div>
        <div class="knowledge-grid">
          <div class="content-card knowledge-card"><div class="card-header"><h3>Search your library</h3></div>
            <div class="input-row"><input v-model="searchText" class="ui-input" type="search" placeholder="Search meetings, topics, or files" aria-label="Knowledge search" @keyup.enter="search" />
              <button type="button" class="ui-btn ui-btn-solid" :disabled="searching || !searchText.trim()" @click="search">{{ searching ? 'Searching' : 'Search' }}</button></div>
            <div v-if="searchResults.length" class="search-list"><div v-for="(hit, i) in searchResults" :key="`${hit.source_id}-${i}`" class="search-hit">
              <span class="source-chip">{{ sourceLabel(hit) }}</span><p>{{ hit.text }}</p>
            </div></div>
            <div v-else class="empty-card compact-empty"><strong>No search results yet</strong><p>Index a meeting, then search for a topic or phrase.</p></div>
          </div>
          <div class="content-card knowledge-card"><div class="card-header"><h3>Ask your knowledge</h3></div>
            <p class="card-description">Answers are grounded in retrieved sources when available.</p>
            <div class="input-row"><input v-model="question" class="ui-input" type="text" placeholder="What decisions did we make?" aria-label="Ask a knowledge question" @keyup.enter="ask" />
              <button type="button" class="ui-btn ui-btn-solid" :disabled="!!busyKind || !question.trim()" @click="ask">Ask</button></div>
            <div v-if="askJob" class="answer-card"><span :class="['status-pill', `status-${askJob.status}`]">{{ askJob.status }}</span>
              <p v-if="currentAskAnswer" class="answer-text">{{ currentAskAnswer }}</p>
              <p v-else-if="askJob.error" class="job-error">{{ askJob.error }}</p>
              <p v-else class="card-description">{{ askJob.status === 'queued' ? 'Waiting for the worker to pick up your question.' : 'Processing your question.' }}</p>
              <div v-if="currentAskSources.length" class="answer-sources"><strong>Sources</strong><div v-for="(source, i) in currentAskSources" :key="i" class="answer-source"><span class="source-chip">{{ sourceLabel(source) }}</span><span>{{ source.text }}</span></div></div>
            </div>
            <div v-else class="empty-card compact-empty"><strong>Start with a question</strong><p>Ask about decisions, key ideas, or follow-up items.</p></div>
          </div>
        </div>
      </div>

      <div v-else-if="view === 'translation'" class="workspace-body">
        <div class="section-heading"><div><h3>Bilingual transcript</h3><p>Translate finalized segments without modifying the original transcript.</p></div></div>
        <div class="content-card translation-card">
          <div class="translation-toolbar"><label class="switch-label"><input v-model="translationEnabled" type="checkbox" /> <span>Enable translation</span></label>
            <label class="select-label">Target language <select v-model="language" class="ui-select" aria-label="Translation language">
              <option v-for="entry in PHASE1_LANGUAGES" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
            </select></label>
            <button type="button" class="ui-btn ui-btn-solid" :disabled="!translationEnabled || !!busyKind || !translationRows.length" @click="queueTranslations">Translate new segments</button>
          </div>
          <div v-if="!translationEnabled" class="empty-card"><strong>Translation is off</strong><p>Enable translation to queue confirmed segments for bilingual viewing.</p></div>
          <div v-else-if="!translationRows.length" class="empty-card"><strong>No finalized speech yet</strong><p>Record or open a meeting with transcript content.</p></div>
          <div v-else class="translation-list"><div class="translation-head"><span>Original</span><span>Translation</span></div>
            <div v-for="seg in translationRows" :key="seg.id" class="translation-entry"><div><span class="segment-time">{{ formatTime(seg.start_ms) }}</span><p>{{ seg.text }}</p></div>
              <div><span :class="['translation-status', { ready: !!translated[seg.id] }]">{{ translated[seg.id] ? 'Translated' : 'Pending' }}</span><p :class="{ 'placeholder-copy': !translated[seg.id] }">{{ translated[seg.id] || 'Waiting for translation' }}</p></div></div>
          </div>
        </div>
      </div>

      <div v-else-if="view === 'media'" class="workspace-body">
        <div class="section-heading"><div><h3>Media analysis</h3><p>Bring documents, audio, and video into your searchable knowledge base.</p></div></div>
        <div class="content-card media-card"><span class="media-mark"><AppIcon name="file-text" :size="27" /></span>
          <h3>Analyze a file</h3><p>Upload a PDF, document, slide deck, image, or recording. Processing runs in the background.</p>
          <input ref="mediaInput" class="visually-hidden" type="file" accept=".txt,.md,.pdf,.docx,.pptx,.png,.jpg,.jpeg,.wav,.mp3,.m4a,.mp4,.mov" aria-label="Select media file" :disabled="!!busyKind" @change="upload" />
          <button type="button" class="ui-btn ui-btn-solid media-pick" :disabled="!!busyKind" @click="mediaInput?.click()">{{ busyKind === 'upload' ? 'Uploading...' : 'Choose file' }}</button>
          <small>Accepted: PDF, DOCX, PPTX, TXT, images, WAV, MP3, M4A, MP4, MOV</small>
          <div v-if="mediaJob" class="media-progress"><div><strong>{{ mediaName || 'Uploaded file' }}</strong><span :class="['status-pill', `status-${mediaJob.status}`]">{{ mediaJob.status }}</span></div>
            <p v-if="mediaJob.error" class="job-error">{{ mediaJob.error }}</p>
            <p v-else-if="mediaJob.status === 'completed' && mediaJob.result?.summary" class="answer-text">{{ String(mediaJob.result.summary) }}</p>
            <p v-else class="card-description">{{ mediaJob.status === 'completed' ? 'Media indexing completed.' : 'The AI worker is processing your file.' }}</p>
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
.knowledge-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:15px; }
.knowledge-card { padding-bottom:14px; }
.card-description { margin:0 18px 12px; font-size:11px; color:var(--ai-muted); }
.input-row { display:flex; align-items:center; gap:8px; margin:0 18px 15px; }
.ui-input,.ui-select { width:100%; min-width:0; padding:10px 12px; border:1px solid #dcd6cc; border-radius:8px; background:#fff; color:var(--ai-text); font-size:12px; }
.input-row .ui-input { flex:1; }
.ui-input::placeholder { color:#a39b91; }
.search-list { padding:0 18px; }
.search-hit { padding:12px 0; border-top:1px solid var(--ai-border); }
.search-hit p { margin:7px 0 0; line-height:1.75; font-size:12px; white-space:pre-wrap; overflow-wrap:anywhere; }
.source-chip { display:inline-flex; padding:3px 7px; border-radius:5px; color:var(--ai-teal); background:rgba(22,124,128,.09); font-size:10px; font-weight:650; }
.answer-card { padding:12px 18px 15px; margin:0 18px; background:#fbfaf8; border-radius:10px; }
.answer-text { font-size:12.5px; white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.85; }
.answer-card .card-description { margin:8px 0 0; }
.answer-sources { padding-top:14px; border-top:1px solid var(--ai-border); font-size:11px; }
.answer-source { display:flex; flex-direction:column; gap:6px; padding-top:10px; line-height:1.65; overflow-wrap:anywhere; }
.translation-card { padding:17px 18px; }
.translation-toolbar { display:flex; align-items:center; gap:18px; flex-wrap:wrap; margin-bottom:14px; }
.switch-label,.select-label { display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:650; }
.switch-label input { accent-color:var(--ai-primary); width:15px; height:15px; }
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
.media-card { text-align:center; display:flex; align-items:center; flex-direction:column; padding:34px 22px; }
.media-mark { display:flex; align-items:center; justify-content:center; border-radius:15px; height:58px; width:58px; background:rgba(218,117,29,.09); color:var(--ai-primary-dark); }
.media-card h3 { margin:13px 0 4px; font-size:17px; }
.media-card > p { color:var(--ai-muted); font-size:12px; max-width:430px; margin:0 0 20px; }
.media-pick { padding:10px 20px; }
.media-card > small { color:var(--ai-muted); font-size:10.5px; margin-top:15px; }
.media-progress { margin-top:22px; padding:14px 16px; border:1px solid var(--ai-border); border-radius:10px; text-align:left; width:min(100%,550px); }
.media-progress > div { display:flex; justify-content:space-between; gap:10px; }
.media-progress strong { overflow-wrap:anywhere; }
.media-progress .card-description { margin:6px 0 0; }
.visually-hidden { position:absolute; height:1px; width:1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; }
@media (max-width:820px) {
  .workspace-head { padding:20px 18px 12px; }
  .workspace-meta { padding:4px 18px 17px; }
  .workspace-nav { padding:0 13px; }
  .workspace-body { padding:20px 18px 24px; }
  .overview-grid,.knowledge-grid { grid-template-columns:minmax(0,1fr); }
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
</style>
