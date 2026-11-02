<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { hotwordsApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'

const router = useRouter()

const error = ref('')
const toast = ref('')
const name = ref('')
const wordsText = ref('')
const creating = ref(false)

/** 按空白/中英文逗号/顿号切分热词。 */
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

const words = computed(() => parseWords(wordsText.value))

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
    showToast(`已创建热词库「${title}」`)
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
        <button class="btn btn-ghost btn-sm" @click="router.push('/hotwords')">
          <AppIcon name="arrow-left" :size="16" />
          返回热词库
        </button>
        <div>
          <h1 class="page-title">新建热词库</h1>
          <p class="page-sub">把会议中经常出现的专有名词收集起来，开会时按需选择。</p>
        </div>
      </div>
      <button
        class="btn btn-primary"
        :disabled="!name.trim() || words.length === 0 || creating"
        @click="createLibrary"
      >
        <AppIcon name="check" :size="16" />
        {{ creating ? '创建中…' : '创建热词库' }}
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <section class="panel">
      <h2 class="panel-title">热词库名称</h2>
      <input
        v-model="name"
        class="text-input"
        placeholder="例如：产品术语 / 项目代号 / 医学名词"
        :disabled="creating"
        @keyup.enter="createLibrary"
      />
    </section>

    <section class="panel">
      <div class="hotword-form-head">
        <h2 class="panel-title">热词列表</h2>
        <span class="hotword-count">{{ words.length }} 个热词</span>
      </div>
      <textarea
        v-model="wordsText"
        class="text-area"
        rows="8"
        placeholder="每行一个热词，例如：&#10;达摩院&#10;魔搭&#10;大模型"
        :disabled="creating"
      ></textarea>
      <p class="hint-text">
        建议使用单行词语（可包含中文、英文、数字）；多个热词用换行、空格或逗号分隔。
        识别时会优先输出这些热词，同一会议最多注入 200 个。
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
