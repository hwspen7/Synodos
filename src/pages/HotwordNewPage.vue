<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { hotwordsApi } from '@/api/client'
import { parseHotwords } from '@/config/hotwords'
import AppIcon from '@/components/AppIcon.vue'

const router = useRouter()

const error = ref('')
const toast = ref('')
const name = ref('')
const wordsText = ref('')
const creating = ref(false)

const words = computed(() => parseHotwords(wordsText.value))

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 3200)
}

async function createLibrary(): Promise<void> {
  const title = name.value.trim()
  if (!title || words.value.length === 0 || creating.value) return
  creating.value = true
  error.value = ''
  try {
    await hotwordsApi.create(title, words.value)
    showToast(`Created hotword library "${title}".`)
    window.setTimeout(() => void router.push('/hotwords'), 500)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}
</script>

<template>
  <div class="hotword-new page-shell">
    <div class="speaker-head">
      <div class="speaker-head-left">
        <RouterLink class="page-inline-back" to="/hotwords"><AppIcon name="arrow-left" :size="15" /> Terminology</RouterLink>
        <div>
          <h1 class="page-title">Create Hotword Library</h1>
          <p class="page-sub">Collect frequently used names and terminology for meetings.</p>
        </div>
      </div>
      <button
        class="btn btn-primary"
        :disabled="!name.trim() || words.length === 0 || creating"
        @click="createLibrary"
      >
        <AppIcon name="check" :size="16" />
        {{ creating ? 'Creating...' : 'Create Hotword Library' }}
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <section class="panel">
      <h2 class="panel-title">Library Name</h2>
      <input
        v-model="name"
        class="text-input"
        placeholder="e.g. Product Terms / Project Codes / Medical Terms"
        :disabled="creating"
        @keyup.enter="createLibrary"
      />
    </section>

    <section class="panel">
      <div class="hotword-form-head">
        <h2 class="panel-title">Hotword List</h2>
        <span class="hotword-count">{{ words.length }} hotwords</span>
      </div>
      <textarea
        v-model="wordsText"
        class="text-area"
        rows="8"
        placeholder="One term per line, e.g. &#10;Research Institute&#10;ModelScope&#10;Large Language Model"
        :disabled="creating"
      ></textarea>
      <p class="hint-text">
        Enter one term per line or separate terms with commas. Spaces within a term are preserved.
        Prioritize selected hotwords during recognition, up to 200 per meeting.
      </p>
      <div v-if="words.length > 0" class="hotword-tags">
        <span v-for="w in words" :key="w" class="hotword-tag">{{ w }}</span>
      </div>
    </section>

    <Transition name="toast">
      <div v-if="toast" class="toast">
        <AppIcon name="check" :size="16" />
        {{ toast }}
      </div>
    </Transition>
  </div>
</template>
