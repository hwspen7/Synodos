<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { hotwordsApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'
import type { HotwordLibrary } from '@/types'

const router = useRouter()

const libraries = ref<HotwordLibrary[]>([])
const loading = ref(true)
const error = ref('')
const toast = ref('')

const editTarget = ref<HotwordLibrary | null>(null)
const editName = ref('')
const editWordsText = ref('')
const editing = ref(false)

/** Preview the first twelve hotwords. */
const PREVIEW_LIMIT = 12

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
    libraries.value = (await hotwordsApi.list()).hotwords
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

function parseWords(text: string): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of text.replace(/[\uFF0C\u3001;\uFF1B]/g, ' ').split(/\s+/)) {
    const w = item.trim()
    if (w && !seen.has(w)) {
      seen.add(w)
      out.push(w)
    }
  }
  return out
}

const editWordCount = computed(() => parseWords(editWordsText.value).length)

function openEdit(lib: HotwordLibrary): void {
  editTarget.value = lib
  editName.value = lib.name
  editWordsText.value = lib.words.join('\n')
}

async function saveEdit(): Promise<void> {
  if (!editTarget.value || !editName.value.trim() || editWordCount.value === 0 || editing.value) {
    return
  }
  editing.value = true
  error.value = ''
  try {
    await hotwordsApi.update(editTarget.value.id, {
      name: editName.value.trim(),
      words: parseWords(editWordsText.value),
    })
    editTarget.value = null
    await load()
    showToast('Library saved')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    editing.value = false
  }
}

async function removeLibrary(lib: HotwordLibrary): Promise<void> {
  if (!window.confirm(`Delete library "${lib.name}" and its ${lib.word_count} terms? This cannot be undone.`)) {
    return
  }
  error.value = ''
  try {
    await hotwordsApi.remove(lib.id)
    await load()
    showToast('Library deleted')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(load)
</script>

<template>
  <div class="hotwords page-shell">
    <div class="speaker-head">
      <div class="speaker-head-left">
        <button class="btn btn-ghost btn-sm" @click="router.push('/')">
          <AppIcon name="arrow-left" :size="16" />
          Back to Home
        </button>
        <div>
          <h1 class="page-title">Hotword Libraries</h1>
          <p class="page-sub">Manage libraries of specialized terminology.</p>
        </div>
      </div>
      <button class="btn btn-primary" @click="router.push('/hotwords/new')">
        <AppIcon name="flame" :size="16" />
        Create Hotword Library
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <div v-if="loading" class="empty-state">
      <p>Loading hotword libraries...</p>
    </div>
    <div v-else-if="libraries.length === 0" class="empty-state">
      <div class="empty-icon"><AppIcon name="flame" :size="26" /></div>
      <h3>No hotword libraries yet</h3>
      <p>Create a library of frequently used names and terms.</p>
      <button class="btn btn-primary" @click="router.push('/hotwords/new')">
        <AppIcon name="flame" :size="16" />
        Create Hotword Library
      </button>
    </div>
    <div v-else class="hotword-list">
      <article v-for="lib in libraries" :key="lib.id" class="hotword-card">
        <div class="hotword-card-head">
          <div class="hotword-card-title-wrap">
            <span class="hotword-card-icon"><AppIcon name="flame" :size="16" /></span>
            <h3 class="hotword-card-title">{{ lib.name }}</h3>
            <span class="hotword-count-badge">{{ lib.word_count }} hotwords</span>
          </div>
          <div class="hotword-card-actions">
            <button class="btn btn-ghost btn-sm" title="Edit Library" @click="openEdit(lib)">
              <AppIcon name="edit" :size="14" />
              Edit
            </button>
            <button class="btn btn-danger btn-sm" title="Delete Library" @click="removeLibrary(lib)">
              <AppIcon name="trash" :size="14" />
            </button>
          </div>
        </div>
        <div v-if="lib.words.length > 0" class="hotword-tags">
          <span
            v-for="w in lib.words.slice(0, PREVIEW_LIMIT)"
            :key="w"
            class="hotword-tag"
          >
            {{ w }}
          </span>
          <span v-if="lib.words.length > PREVIEW_LIMIT" class="hotword-more">
            +{{ lib.words.length - PREVIEW_LIMIT }} more
          </span>
        </div>
        <p v-else class="hint-text">No hotwords in this library.</p>
        <p class="hotword-meta">Updated {{ fmtDate(lib.updated_at) }}</p>
      </article>
    </div>

    <div v-if="editTarget" class="dialog-backdrop" @click.self="editTarget = null">
      <div class="dialog dialog-wide">
        <h3>Edit Hotword Library</h3>
        <p class="dialog-sub">Changes take effect when the meeting reconnects.</p>
        <label class="field dialog-field">
          Library Name
          <input v-model="editName" class="text-input" placeholder="Library Name" />
        </label>
        <label class="field dialog-field">
          Hotwords ({{ editWordCount }})
          <textarea
            v-model="editWordsText"
            class="text-area"
            rows="7"
            placeholder="One hotword per line"
          ></textarea>
        </label>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="editTarget = null">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="!editName.trim() || editWordCount === 0 || editing"
            @click="saveEdit"
          >
            {{ editing ? 'Saving...' : 'Save' }}
          </button>
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
