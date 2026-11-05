<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { meetingsApi, speakersApi } from '@/api/client'

const props = defineProps<{
  meetingId: string
  segmentId: number | null
  revision: number | null
  recording: boolean
}>()
const emit = defineEmits<{ (event: 'reviewed'): void }>()

type Candidate = {
  status: string
  profile_id: string | null
  similarity: number | null
  margin: number | null
  score_kind?: string
  embedding_model?: string
  enrollment_human_verified?: boolean
  evidence_level?: string
  human_review_required?: boolean
}
type RevisionEntry = {
  revision?: number
  timestamp?: string
  updated_at?: string
  reason?: string
  source?: string
  before?: { text?: string; speaker_name?: string }
  after?: { text?: string; speaker_name?: string }
}
const loading = ref(false)
const saving = ref(false)
const error = ref('')
const identity = ref<Candidate | null>(null)
const names = ref<Record<string, string>>({})
const history = ref<RevisionEntry[]>([])
const initialText = ref('')
const currentText = ref('')
const proposedText = ref('')
const currentRevision = ref(0)
const currentProfileId = ref<string | null>(null)
const selectedProfileId = ref('')
let loadGeneration = 0

const identityName = computed(() => identity.value?.profile_id
  ? (names.value[identity.value.profile_id] ?? 'Enrolled speaker')
  : 'Unknown')
const matchedCandidate = computed(() => identity.value?.status === 'matched'
  && Boolean(identity.value.profile_id))
const canSaveText = computed(() => !props.recording && !saving.value &&
  props.segmentId !== null && proposedText.value !== currentText.value)
const canSaveIdentity = computed(() => !props.recording && !saving.value &&
  props.segmentId !== null && selectedProfileId.value !== (currentProfileId.value ?? ''))

async function refresh(): Promise<void> {
  const generation = ++loadGeneration
  error.value = ''
  identity.value = null
  history.value = []
  initialText.value = ''
  currentText.value = ''
  proposedText.value = ''
  selectedProfileId.value = ''
  currentProfileId.value = null
  if (props.segmentId === null || props.recording) return
  loading.value = true
  try {
    const [candidate, review, profiles, segmentResponse] = await Promise.all([
      meetingsApi.identityCandidate(props.meetingId, props.segmentId),
      meetingsApi.revisions(props.meetingId, props.segmentId),
      speakersApi.list(),
      meetingsApi.segments(props.meetingId),
    ])
    if (generation !== loadGeneration) return
    const segment = segmentResponse.segments.find(item => item.id === props.segmentId)
    if (!segment) throw new Error('Saved segment is no longer available')
    identity.value = candidate.candidate
    initialText.value = review.original_asr_text
    currentText.value = segment.text
    proposedText.value = segment.text
    currentRevision.value = segment.revision ?? review.revision
    currentProfileId.value = segment.speaker_profile_id ?? null
    selectedProfileId.value = currentProfileId.value ?? ''
    history.value = review.corrections as RevisionEntry[]
    names.value = Object.fromEntries(profiles.speakers.map(item => [item.id, item.name]))
  } catch (cause) {
    if (generation === loadGeneration) {
      error.value = cause instanceof Error ? cause.message : String(cause)
    }
  } finally {
    if (generation === loadGeneration) loading.value = false
  }
}

watch(
  () => [props.meetingId, props.segmentId, props.revision, props.recording] as const,
  () => { void refresh() },
  { immediate: true },
)

async function acceptCandidate(): Promise<void> {
  if (props.recording || props.segmentId === null || saving.value
      || !matchedCandidate.value || !identity.value?.profile_id) return
  const confirmed = window.confirm(
    'Only confirm this identity after independently listening to the original audio. ' +
    'Cosine similarity is not an identity probability. Save this manual correction?',
  )
  if (!confirmed) return
  saving.value = true
  error.value = ''
  try {
    await meetingsApi.reviewSegment(props.meetingId, props.segmentId, {
      expected_revision: currentRevision.value,
      speaker_profile_id: identity.value.profile_id,
      reason: 'Operator reviewed original audio before confirming speaker',
    })
    emit('reviewed')
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    saving.value = false
  }
}

async function saveTextCorrection(): Promise<void> {
  if (props.segmentId === null || !canSaveText.value) return
  const confirmed = window.confirm(
    'Confirm this transcript correction against the source audio? The prior text and revision will remain in history.',
  )
  if (!confirmed) return
  saving.value = true
  error.value = ''
  try {
    await meetingsApi.reviewSegment(props.meetingId, props.segmentId, {
      expected_revision: currentRevision.value,
      text: proposedText.value,
      reason: 'Operator reviewed source audio and corrected transcript',
    })
    emit('reviewed')
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    saving.value = false
  }
}

async function saveManualIdentity(): Promise<void> {
  if (props.segmentId === null || !canSaveIdentity.value) return
  const profile = selectedProfileId.value || null
  if (profile !== null && !names.value[profile]) {
    error.value = 'Selected speaker profile no longer exists'
    return
  }
  const confirmed = window.confirm(
    'Confirm the speaker correction only after listening to the original recording? This will create a versioned manual review event.',
  )
  if (!confirmed) return
  saving.value = true
  error.value = ''
  try {
    await meetingsApi.reviewSegment(props.meetingId, props.segmentId, {
      expected_revision: currentRevision.value,
      speaker_profile_id: profile,
      reason: 'Operator reviewed source audio and corrected speaker identity',
    })
    emit('reviewed')
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <section class="phase2-evidence" aria-label="Speaker identity evidence and revision history">
    <h3>Speaker identity evidence</h3>
    <p class="quiet">A candidate is a suggestion only, not a verified identity. Original ASR text is retained.</p>
    <p v-if="segmentId === null" class="quiet">Choose an Evidence button on a saved transcript segment.</p>
    <p v-else-if="recording" class="quiet">Stop recording before reviewing stored identities.</p>
    <template v-else>
      <p v-if="loading" role="status">Loading saved review evidence...</p>
      <p v-if="error" class="phase2-error" role="alert">{{ error }}</p>
      <template v-if="identity">
        <p><strong>Segment #{{ segmentId }}</strong> · {{ identityName }}</p>
        <p class="quiet">Status {{ identity.status }} · {{ identity.embedding_model || 'unavailable model' }}</p>
        <p class="quiet">Enrollment source: {{ identity.enrollment_human_verified
          ? 'Prior human-verified source' : 'Not independently verified' }}.
          A cosine match is never automatic identity confirmation.</p>
        <p v-if="identity.similarity !== null" class="quiet">
          Raw cosine {{ identity.similarity.toFixed(3) }}
          <template v-if="identity.margin !== null"> · margin {{ identity.margin.toFixed(3) }}</template>
        </p>
        <button class="btn btn-ghost btn-sm" :disabled="saving || !matchedCandidate"
          @click="acceptCandidate">
          {{ saving ? 'Saving review...' : 'Confirm after listening' }}
        </button>
      </template>
      <h4>Human correction</h4>
      <label class="review-field">
        Confirmed transcript text
        <textarea v-model="proposedText" :disabled="saving || loading"
          rows="3" maxlength="30000" aria-label="Edit saved transcript text"></textarea>
      </label>
      <button class="btn btn-primary btn-sm" :disabled="!canSaveText"
        @click="saveTextCorrection">Save versioned text correction</button>
      <label class="review-field">
        Speaker identity after listening
        <select v-model="selectedProfileId" :disabled="saving || loading"
          aria-label="Review speaker identity">
          <option value="">Unidentified Speaker</option>
          <option v-for="(name, profileId) in names" :key="profileId" :value="profileId">
            {{ name }}
          </option>
        </select>
      </label>
      <button class="btn btn-primary btn-sm" :disabled="!canSaveIdentity"
        @click="saveManualIdentity">Save versioned identity correction</button>
      <p class="quiet">Edits are stored after recording only. A stale revision
        is rejected rather than overwriting another review.</p>
      <h4>Transcript provenance</h4>
      <p v-if="initialText" class="initial-text">{{ initialText }}</p>
      <p v-else class="quiet">No original ASR text is available.</p>
      <p class="quiet">{{ history.length }} saved revision events</p>
      <div v-for="(item, index) in history" :key="index" class="phase2-event">
        <strong>Revision {{ item.revision ?? index + 1 }}</strong>
        <span>{{ item.reason || item.source || 'Manual change' }}</span>
        <p v-if="item.before?.speaker_name || item.after?.speaker_name" class="quiet">
          {{ item.before?.speaker_name || 'Unknown' }} → {{ item.after?.speaker_name || 'Unknown' }}
        </p>
      </div>
    </template>
  </section>
</template>

<style scoped>
.phase2-evidence { padding: 16px; border-top: 1px solid var(--border, #ddd); }
.phase2-evidence h3 { margin: 0 0 8px; font-size: 15px; }
.phase2-evidence h4 { margin-top: 18px; margin-bottom: 8px; font-size: 13px; }
.phase2-evidence .quiet { color: var(--text-muted, #777); font-size: 12px; line-height: 1.5; }
.review-field { display: grid; gap: 5px; margin: 12px 0 6px;
  font-size: 12px; color: var(--text-muted, #777); }
.review-field textarea, .review-field select { width: 100%; box-sizing: border-box;
  padding: 8px; border: 1px solid var(--border, #ddd); border-radius: 6px;
  background: var(--card, #fff); color: var(--text, #333); }
.initial-text { padding: 8px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 180px; overflow: auto; }
.phase2-error { color: #b93e27; overflow-wrap: anywhere; }
.phase2-event { border-top: 1px solid var(--border, #ddd); padding: 8px 0; font-size: 12px; display: grid; gap: 4px; }
</style>
