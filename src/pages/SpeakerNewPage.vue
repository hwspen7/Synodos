<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { speakersApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'
import SpeakerVectorAdder from '@/components/SpeakerVectorAdder.vue'
import type { Speaker, SpeakerVector } from '@/types'

const router = useRouter()

const error = ref('')
const toast = ref('')

// Step 1: Speaker name
const name = ref('')
const creating = ref(false)
const speaker = ref<Speaker | null>(null)

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 3200)
}

// ── Create Speaker ───────────────────────────
async function createSpeaker(): Promise<void> {
  const title = name.value.trim()
  if (!title || creating.value) return
  creating.value = true
  error.value = ''
  try {
    const { speaker: spk } = await speakersApi.create(title)
    speaker.value = spk
    showToast(`Speaker "${spk.name}" created. Add a voiceprint.`)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}

function onVectorsAdded(vectors: SpeakerVector[]): void {
  if (speaker.value) speaker.value.vector_count += vectors.length
  showToast(`Added ${vectors.length} voiceprints`)
}

function finish(): void {
  void router.push('/speakers')
}
</script>

<template>
  <div class="speaker-new page-shell">
    <div class="speaker-head">
      <div class="speaker-head-left">
        <RouterLink class="page-inline-back" to="/speakers"><AppIcon name="arrow-left" :size="15" /> Speaker library</RouterLink>
        <div>
          <h1 class="page-title">Create Speaker</h1>
          <p class="page-sub">Enroll voiceprints from recordings or saved meeting segments.</p>
        </div>
      </div>
      <button v-if="speaker" class="btn btn-primary" @click="finish">
        <AppIcon name="check" :size="16" />
        Done
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <section class="panel">
      <h2 class="panel-title">1 · Speaker Name</h2>
      <div v-if="!speaker" class="inline-form">
        <input
          v-model="name"
          class="text-input grow"
          placeholder="e.g. Alex / Product Manager"
          :disabled="creating"
          @keyup.enter="createSpeaker"
        />
        <button
          class="btn btn-primary"
          :disabled="!name.trim() || creating"
          @click="createSpeaker"
        >
          <AppIcon name="user-plus" :size="16" />
          {{ creating ? 'Creating...' : 'Create Speaker' }}
        </button>
      </div>
      <div v-else class="speaker-created">
        <span class="speaker-avatar">{{ speaker.name.slice(0, 1) }}</span>
        <div>
          <strong>{{ speaker.name }}</strong>
          <span class="muted-text">{{ speaker.vector_count }} voiceprints enrolled. You can add more samples.</span>
        </div>
      </div>
    </section>

    <section v-if="speaker" class="panel">
      <h2 class="panel-title">2 · Add Voiceprint Samples</h2>
      <SpeakerVectorAdder :speaker="speaker" @added="onVectorsAdded" @error="error = $event" />
    </section>

    <Transition name="toast">
      <div v-if="toast" class="toast">
        <AppIcon name="check" :size="16" />
        {{ toast }}
      </div>
    </Transition>
  </div>
</template>
