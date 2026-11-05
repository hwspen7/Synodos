<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { speakersApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'
import SpeakerVectorAdder from '@/components/SpeakerVectorAdder.vue'
import type { Speaker, SpeakerVector } from '@/types'

const router = useRouter()

const speakers = ref<Speaker[]>([])
const loading = ref(true)
const error = ref('')
const toast = ref('')

const renameTarget = ref<Speaker | null>(null)
const renameName = ref('')
const renaming = ref(false)
const addTarget = ref<Speaker | null>(null)

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 2600)
}

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    speakers.value = (await speakersApi.list()).speakers
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
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

function openRename(speaker: Speaker): void {
  renameTarget.value = speaker
  renameName.value = speaker.name
}

async function renameSpeaker(): Promise<void> {
  if (!renameTarget.value || !renameName.value.trim() || renaming.value) return
  renaming.value = true
  error.value = ''
  try {
    await speakersApi.rename(renameTarget.value.id, renameName.value.trim())
    renameTarget.value = null
    await load()
    showToast('Renamed')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    renaming.value = false
  }
}

async function removeVector(speaker: Speaker, vectorId: string): Promise<void> {
  if (!window.confirm('Delete this voiceprint? This action cannot be undone.')) return
  error.value = ''
  try {
    await speakersApi.deleteVector(speaker.id, vectorId)
    await load()
    showToast('Voiceprint deleted')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

async function onVectorsAdded(vectors: SpeakerVector[]): Promise<void> {
  await load()
  showToast(`Added ${vectors.length} voiceprints for ${addTarget.value?.name ?? ''}`)
}

async function removeSpeaker(speaker: Speaker): Promise<void> {
  if (!window.confirm(`Delete speaker "${speaker.name}" and all ${speaker.vector_count} voiceprints? This cannot be undone.`)) {
    return
  }
  error.value = ''
  try {
    await speakersApi.remove(speaker.id)
    await load()
    showToast('Speaker deleted')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(load)
</script>

<template>
  <div class="speakers page-shell">
    <div class="speaker-head">
      <div class="speaker-head-left">
        <span class="page-section-kicker">PEOPLE / VOICE IDENTITIES</span>
        <div>
          <h1 class="page-title">Speaker Voiceprint Library</h1>
          <p class="page-sub">Store multiple voiceprints for each speaker to improve meeting identification.</p>
        </div>
      </div>
      <button class="btn btn-primary" @click="router.push('/speakers/new')">
        <AppIcon name="user-plus" :size="16" />
        Create Speaker
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <div v-if="loading" class="empty-state">
      <p>Loading speakers...</p>
    </div>
    <div v-else-if="speakers.length === 0" class="empty-state">
      <div class="empty-icon"><AppIcon name="users" :size="26" /></div>
      <h3>No speakers yet</h3>
      <p>Create a speaker and enroll a recording or meeting segment.</p>
      <button class="btn btn-primary" @click="router.push('/speakers/new')">
        <AppIcon name="user-plus" :size="16" />
        Create Speaker
      </button>
    </div>
    <div v-else class="speaker-list">
      <article v-for="speaker in speakers" :key="speaker.id" class="speaker-card">
        <div class="speaker-main">
          <span class="speaker-avatar">{{ speaker.name.slice(0, 1) }}</span>
          <div class="speaker-info">
            <h3 class="speaker-card-title">{{ speaker.name }}</h3>
            <p class="speaker-meta">
              {{ speaker.vector_count }} voiceprints
              <span class="dot-sep">·</span>
              Updated {{ fmtDate(speaker.updated_at) }}
            </p>
            <span v-if="speaker.vectors.some(vector => vector.embedding_model !== 'ecapa' && !vector.short && vector.duration_s >= 2)"
              class="voice-ready-indicator" title="Has at least one eligible CAM++ voiceprint. A future match still requires sufficient acoustic similarity.">
              <AppIcon name="check" :size="12" /> Auto-match eligible
            </span>
            <span v-else class="voice-needs-sample" title="Automatic cross-meeting matching requires a compatible, clean CAM++ sample of at least two seconds.">
              Needs a clean CAM++ sample
            </span>
          </div>
        </div>
        <div class="speaker-actions">
          <button
            class="btn btn-ghost btn-sm"
            title="Add another voiceprint for this speaker"
            @click="addTarget = speaker"
          >
            <AppIcon name="mic" :size="14" />
            Add Voiceprint
          </button>
          <button class="btn btn-ghost btn-sm" title="Rename" @click="openRename(speaker)">
            <AppIcon name="edit" :size="14" />
            Rename
          </button>
          <button class="btn btn-danger btn-sm" title="Delete" @click="removeSpeaker(speaker)">
            <AppIcon name="trash" :size="14" />
          </button>
        </div>

        <div v-if="speaker.vectors.length > 0" class="vector-list">
          <div v-for="v in speaker.vectors" :key="v.id" class="vector-row">
            <AppIcon name="list" :size="13" class="vector-icon" />
            <span class="vector-src">
              {{ v.source === 'recording' ? 'Live Recording' : 'Meeting Segment' }}
            </span>
            <span class="muted-text">{{ v.duration_s.toFixed(1) }} s</span>
            <span v-if="v.source === 'meeting' && v.meeting_id" class="muted-text">
              Meeting {{ v.meeting_id.slice(0, 8) }} · segments #{{ v.segment_ids.join(', ') }}
            </span>
            <span v-if="v.text" class="muted-text vector-text">{{ v.text }}</span>
            <span v-if="v.short" class="warn-badge" title="Short audio may lead to unreliable voiceprint matching.">
              <AppIcon name="alert" :size="12" />
              Too short
            </span>
            <span class="vector-date">{{ fmtDate(v.created_at) }}</span>
            <button
              class="icon-btn danger"
              title="Delete this voiceprint"
              @click="removeVector(speaker, v.id)"
            >
              <AppIcon name="trash" :size="13" />
            </button>
          </div>
        </div>
        <p v-else class="hint-text">No voiceprints yet. Add a sample.</p>
      </article>
    </div>

    <div v-if="renameTarget" class="dialog-backdrop" @click.self="renameTarget = null">
      <div class="dialog">
        <h3>Rename Speaker</h3>
        <p class="dialog-sub">The updated name will also appear in saved meetings.</p>
        <input
          v-model="renameName"
          class="text-input"
          placeholder="Speaker Name"
          autofocus
          @keyup.enter="renameSpeaker"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="renameTarget = null">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="!renameName.trim() || renaming"
            @click="renameSpeaker"
          >
            {{ renaming ? 'Saving...' : 'Save' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="addTarget" class="dialog-backdrop" @click.self="addTarget = null">
      <div class="dialog dialog-wide">
        <h3>Add Voiceprints for {{ addTarget.name }}</h3>
        <p class="dialog-sub">Add voiceprints from recordings or meeting segments.</p>
        <SpeakerVectorAdder
          :key="addTarget.id"
          :speaker="addTarget"
          @added="onVectorsAdded"
          @error="error = $event"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="addTarget = null">Close</button>
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
