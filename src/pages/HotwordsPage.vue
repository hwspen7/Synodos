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

/** 展示前 12 个热词，其余折叠。 */
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
  for (const item of text.replace(/[，、;；]/g, ' ').split(/\s+/)) {
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
    showToast('已保存热词库')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    editing.value = false
  }
}

async function removeLibrary(lib: HotwordLibrary): Promise<void> {
  if (!window.confirm(`确定删除热词库「${lib.name}」及其 ${lib.word_count} 个热词？此操作不可恢复。`)) {
    return
  }
  error.value = ''
  try {
    await hotwordsApi.remove(lib.id)
    await load()
    showToast('已删除热词库')
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
          返回首页
        </button>
        <div>
          <h1 class="page-title">热词库</h1>
          <p class="page-sub">维护一组组热词，开会时选中热词库即可提升专有名词识别率。</p>
        </div>
      </div>
      <button class="btn btn-primary" @click="router.push('/hotwords/new')">
        <AppIcon name="flame" :size="16" />
        新建热词库
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <div v-if="loading" class="empty-state">
      <p>正在加载热词库…</p>
    </div>
    <div v-else-if="libraries.length === 0" class="empty-state">
      <div class="empty-icon"><AppIcon name="flame" :size="26" /></div>
      <h3>还没有热词库</h3>
      <p>新建一个热词库，把会议中常出现的专有名词放进去，开会时按需选择。</p>
      <button class="btn btn-primary" @click="router.push('/hotwords/new')">
        <AppIcon name="flame" :size="16" />
        新建热词库
      </button>
    </div>
    <div v-else class="hotword-list">
      <article v-for="lib in libraries" :key="lib.id" class="hotword-card">
        <div class="hotword-card-head">
          <div class="hotword-card-title-wrap">
            <span class="hotword-card-icon"><AppIcon name="flame" :size="16" /></span>
            <h3 class="hotword-card-title">{{ lib.name }}</h3>
            <span class="hotword-count-badge">{{ lib.word_count }} 个热词</span>
          </div>
          <div class="hotword-card-actions">
            <button class="btn btn-ghost btn-sm" title="编辑热词库" @click="openEdit(lib)">
              <AppIcon name="edit" :size="14" />
              编辑
            </button>
            <button class="btn btn-danger btn-sm" title="删除热词库" @click="removeLibrary(lib)">
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
            +{{ lib.words.length - PREVIEW_LIMIT }} 个
          </span>
        </div>
        <p v-else class="hint-text">热词列表为空。</p>
        <p class="hotword-meta">更新于 {{ fmtDate(lib.updated_at) }}</p>
      </article>
    </div>

    <div v-if="editTarget" class="dialog-backdrop" @click.self="editTarget = null">
      <div class="dialog dialog-wide">
        <h3>编辑热词库</h3>
        <p class="dialog-sub">修改名称或热词列表；已选择该库的会议会在下次连接时生效。</p>
        <label class="field dialog-field">
          热词库名称
          <input v-model="editName" class="text-input" placeholder="热词库名称" />
        </label>
        <label class="field dialog-field">
          热词列表（{{ editWordCount }} 个）
          <textarea
            v-model="editWordsText"
            class="text-area"
            rows="7"
            placeholder="每行一个热词"
          ></textarea>
        </label>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="editTarget = null">取消</button>
          <button
            class="btn btn-primary"
            :disabled="!editName.trim() || editWordCount === 0 || editing"
            @click="saveEdit"
          >
            {{ editing ? '保存中…' : '保存' }}
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
