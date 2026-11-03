<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { downloadUrl, hotwordsApi, meetingsApi, speakersApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'
import type { HotwordLibrary, MeetingSummary, Speaker } from '@/types'

const router = useRouter()

const meetings = ref<MeetingSummary[]>([])
const speakers = ref<Speaker[]>([])
const hotwords = ref<HotwordLibrary[]>([])
const loading = ref(true)
const error = ref('')
const showCreate = ref(false)
const newTitle = ref('')
const creating = ref(false)

const totalDurationMs = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.duration_ms ?? 0), 0),
)
const totalSegments = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.segments ?? 0), 0),
)
const totalSpeakers = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.speakers ?? 0), 0),
)
const totalSpeakerVectors = computed(() =>
  speakers.value.reduce((sum, s) => sum + (s.vector_count ?? 0), 0),
)
const totalHotwordWords = computed(() =>
  hotwords.value.reduce((sum, h) => sum + (h.word_count ?? 0), 0),
)

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    meetings.value = (await meetingsApi.list()).meetings
    speakers.value = (await speakersApi.list()).speakers
    hotwords.value = (await hotwordsApi.list()).hotwords
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function createMeeting(): Promise<void> {
  const title = newTitle.value.trim()
  if (!title || creating.value) return
  creating.value = true
  error.value = ''
  try {
    const { meeting } = await meetingsApi.create(title)
    showCreate.value = false
    newTitle.value = ''
    await router.push({ name: 'meeting', params: { id: meeting.id }, query: { autostart: '1' } })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}

function viewMeeting(meeting: MeetingSummary): void {
  void router.push(`/meeting/${meeting.id}`)
}

function resumeMeeting(meeting: MeetingSummary): void {
  void router.push(`/meeting/${meeting.id}?autostart=1`)
}

function exportMeeting(meeting: MeetingSummary, format: 'md' | 'json'): void {
  downloadUrl(meetingsApi.exportUrl(meeting.id, format))
}

async function removeMeeting(meeting: MeetingSummary): Promise<void> {
  if (!window.confirm(`Delete meeting "${meeting.title}"? This action cannot be undone.`)) return
  error.value = ''
  try {
    await meetingsApi.remove(meeting.id)
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function fmtDuration(ms: number): string {
  const total = Math.max(0, Math.floor((ms ?? 0) / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h} h ${m} min`
  if (m > 0) return `${m} min ${s} s`
  return `${s} s`
}

function fmtDate(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

onMounted(load)
</script>

<template>
  <div class="home">
    <section class="home-compact-hero">
      <div class="home-compact-hero-inner">
        <div>
          <p class="home-eyebrow">MEETING OF PIONEERS · WORKSPACE</p>
          <h1>Your meetings, organized.</h1>
          <p>Record conversations, follow speakers and review a clear transcript.</p>
        </div>
        <button class="btn btn-primary btn-lg" @click="showCreate = true">
          <AppIcon name="plus" :size="18" /> Start a meeting
        </button>
      </div>
    </section>
    <main class="page-shell">
      <div class="metric-grid">
        <div class="metric-tile">
          <div class="metric-icon tone-primary"><AppIcon name="file-text" :size="20" /></div>
          <div>
            <div class="metric-label">Meetings</div>
            <div class="metric-value">{{ meetings.length }}</div>
          </div>
        </div>
        <div class="metric-tile">
          <div class="metric-icon tone-secondary"><AppIcon name="clock" :size="20" /></div>
          <div>
            <div class="metric-label">Total Duration</div>
            <div class="metric-value">{{ fmtDuration(totalDurationMs) }}</div>
          </div>
        </div>
        <div class="metric-tile">
          <div class="metric-icon tone-accent"><AppIcon name="mic" :size="20" /></div>
          <div>
            <div class="metric-label">Audio Segments</div>
            <div class="metric-value">{{ totalSegments }}</div>
          </div>
        </div>
      <div class="metric-tile">
        <div class="metric-icon tone-success"><AppIcon name="users" :size="20" /></div>
        <div>
          <div class="metric-label">Total Speakers</div>
          <div class="metric-value">{{ totalSpeakers }}</div>
        </div>
      </div>
      </div>

      <section class="home-quick-library" aria-label="Meeting resources">
        <RouterLink class="home-library-shortcut" to="/speakers">
          <span class="shortcut-icon"><AppIcon name="users" :size="19" /></span>
          <span class="shortcut-copy"><strong>Speakers</strong><small>{{ speakers.length }} people · {{ totalSpeakerVectors }} voiceprints</small></span>
          <span class="shortcut-arrow">→</span>
        </RouterLink>
        <RouterLink class="home-library-shortcut" to="/hotwords">
          <span class="shortcut-icon"><AppIcon name="flame" :size="19" /></span>
          <span class="shortcut-copy"><strong>Terminology</strong><small>{{ hotwords.length }} libraries · {{ totalHotwordWords }} terms</small></span>
          <span class="shortcut-arrow">→</span>
        </RouterLink>
      </section>

      <p v-if="error" class="error-banner">{{ error }}</p>

      <section id="meetings" class="meetings-section">
        <div class="toolbar-row">
          <div>
            <h2>Saved Meetings</h2>
            <p class="section-sub">Saved transcripts can be resumed, viewed, or exported.</p>
          </div>
          <button class="btn btn-primary" @click="showCreate = true">
            <AppIcon name="plus" :size="16" />
            New Meeting
          </button>
        </div>

        <div v-if="loading" class="empty-state">
          <p>Loading meetings...</p>
        </div>
        <div v-else-if="meetings.length === 0" class="empty-state">
          <div class="empty-icon"><AppIcon name="mic" :size="26" /></div>
          <h3>No meetings yet</h3>
          <p>Create a meeting and enable microphone access.</p>
          <button class="btn btn-primary" @click="showCreate = true">
            <AppIcon name="plus" :size="16" />
            Create Meeting
          </button>
        </div>
        <div v-else class="meeting-list">
          <article v-for="meeting in meetings" :key="meeting.id" class="meeting-card">
            <div class="meeting-main" @click="viewMeeting(meeting)">
              <h3 class="meeting-title">{{ meeting.title }}</h3>
              <p class="meeting-meta">
                Updated {{ fmtDate(meeting.updated_at) }}
                <span class="dot-sep">·</span>
                Duration {{ fmtDuration(meeting.duration_ms) }}
                <span class="dot-sep">·</span>
                {{ meeting.segments }} segments
                <span class="dot-sep">·</span>
                {{ meeting.speakers }} speakers
              </p>
            </div>
            <div class="meeting-actions">
              <button class="btn btn-ghost btn-sm" title="View Transcript" @click="viewMeeting(meeting)">
                <AppIcon name="edit" :size="15" />
                View
              </button>
              <button
                class="btn btn-primary btn-sm"
                title="Resume from End"
                @click="resumeMeeting(meeting)"
              >
                <AppIcon name="play" :size="14" />
                Resume
              </button>
              <div class="action-group">
                <button class="btn btn-ghost btn-sm" title="Export Markdown" @click="exportMeeting(meeting, 'md')">
                  <AppIcon name="download" :size="15" />
                  MD
                </button>
                <button class="btn btn-ghost btn-sm" title="Export JSON" @click="exportMeeting(meeting, 'json')">
                  <AppIcon name="file-text" :size="15" />
                  JSON
                </button>
                <button class="btn btn-danger btn-sm" title="Delete" @click="removeMeeting(meeting)">
                  <AppIcon name="trash" :size="15" />
                </button>
              </div>
            </div>
          </article>
        </div>
      </section>
    </main>

    <div v-if="showCreate" class="dialog-backdrop" @click.self="showCreate = false">
      <div class="dialog">
        <h3>Create Meeting</h3>
        <p class="dialog-sub">Name your meeting. You can resume it later.</p>
        <input
          v-model="newTitle"
          class="text-input"
          placeholder="e.g. Weekly Product Meeting / Project Review"
          autofocus
          @keyup.enter="createMeeting"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showCreate = false">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="!newTitle.trim() || creating"
            @click="createMeeting"
          >
            {{ creating ? 'Creating...' : 'Create and Start' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style src="../assets/home-refresh.css"></style>
