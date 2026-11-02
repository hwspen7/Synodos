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

// 步骤 1：说话人姓名
const name = ref('')
const creating = ref(false)
const speaker = ref<Speaker | null>(null)

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 3200)
}

// ── 创建说话人 ──────────────────────────────────────────
async function createSpeaker(): Promise<void> {
  const title = name.value.trim()
  if (!title || creating.value) return
  creating.value = true
  error.value = ''
  try {
    const { speaker: spk } = await speakersApi.create(title)
    speaker.value = spk
    showToast(`已创建说话人「${spk.name}」，请添加声纹`)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}

function onVectorsAdded(vectors: SpeakerVector[]): void {
  if (speaker.value) speaker.value.vector_count += vectors.length
  showToast(`已添加 ${vectors.length} 条声纹`)
}

function finish(): void {
  void router.push('/speakers')
}
</script>

<template>
  <div class="speaker-new page-shell">
    <div class="speaker-head">
      <div class="speaker-head-left">
        <button class="btn btn-ghost btn-sm" @click="router.push('/speakers')">
          <AppIcon name="arrow-left" :size="16" />
          返回说话人库
        </button>
        <div>
          <h1 class="page-title">新建说话人</h1>
          <p class="page-sub">现场录音提取声纹，或从已保存会议中选取片段作为声纹样本。</p>
        </div>
      </div>
      <button v-if="speaker" class="btn btn-primary" @click="finish">
        <AppIcon name="check" :size="16" />
        完成
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <section class="panel">
      <h2 class="panel-title">1 · 说话人姓名</h2>
      <div v-if="!speaker" class="inline-form">
        <input
          v-model="name"
          class="text-input grow"
          placeholder="例如：张三 / 产品经理小李"
          :disabled="creating"
          @keyup.enter="createSpeaker"
        />
        <button
          class="btn btn-primary"
          :disabled="!name.trim() || creating"
          @click="createSpeaker"
        >
          <AppIcon name="user-plus" :size="16" />
          {{ creating ? '创建中…' : '创建说话人' }}
        </button>
      </div>
      <div v-else class="speaker-created">
        <span class="speaker-avatar">{{ speaker.name.slice(0, 1) }}</span>
        <div>
          <strong>{{ speaker.name }}</strong>
          <span class="muted-text">已有 {{ speaker.vector_count }} 条声纹，可继续添加多条声纹样本。</span>
        </div>
      </div>
    </section>

    <section v-if="speaker" class="panel">
      <h2 class="panel-title">2 · 添加声纹样本</h2>
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
