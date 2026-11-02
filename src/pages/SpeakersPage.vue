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
    showToast('已重命名')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    renaming.value = false
  }
}

async function removeVector(speaker: Speaker, vectorId: string): Promise<void> {
  if (!window.confirm(`确定删除这条声纹？此操作不可恢复。`)) return
  error.value = ''
  try {
    await speakersApi.deleteVector(speaker.id, vectorId)
    await load()
    showToast('已删除声纹')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

async function onVectorsAdded(vectors: SpeakerVector[]): Promise<void> {
  await load()
  showToast(`已为「${addTarget.value?.name ?? ''}」添加 ${vectors.length} 条声纹`)
}

async function removeSpeaker(speaker: Speaker): Promise<void> {
  if (!window.confirm(`确定删除说话人「${speaker.name}」及其全部 ${speaker.vector_count} 条声纹？此操作不可恢复。`)) {
    return
  }
  error.value = ''
  try {
    await speakersApi.remove(speaker.id)
    await load()
    showToast('已删除说话人')
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
        <button class="btn btn-ghost btn-sm" @click="router.push('/')">
          <AppIcon name="arrow-left" :size="16" />
          返回首页
        </button>
        <div>
          <h1 class="page-title">说话人声纹库</h1>
          <p class="page-sub">每个说话人可保存多条声纹特征向量，供会议识别参与者使用。</p>
        </div>
      </div>
      <button class="btn btn-primary" @click="router.push('/speakers/new')">
        <AppIcon name="user-plus" :size="16" />
        新建说话人
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <div v-if="loading" class="empty-state">
      <p>正在加载说话人…</p>
    </div>
    <div v-else-if="speakers.length === 0" class="empty-state">
      <div class="empty-icon"><AppIcon name="users" :size="26" /></div>
      <h3>还没有保存的说话人</h3>
      <p>新建说话人并添加声纹样本（现场录音或会议片段），之后可在会议中标注参与者。</p>
      <button class="btn btn-primary" @click="router.push('/speakers/new')">
        <AppIcon name="user-plus" :size="16" />
        新建说话人
      </button>
    </div>
    <div v-else class="speaker-list">
      <article v-for="speaker in speakers" :key="speaker.id" class="speaker-card">
        <div class="speaker-main">
          <span class="speaker-avatar">{{ speaker.name.slice(0, 1) }}</span>
          <div class="speaker-info">
            <h3 class="speaker-card-title">{{ speaker.name }}</h3>
            <p class="speaker-meta">
              {{ speaker.vector_count }} 条声纹
              <span class="dot-sep">·</span>
              更新于 {{ fmtDate(speaker.updated_at) }}
            </p>
          </div>
        </div>
        <div class="speaker-actions">
          <button
            class="btn btn-ghost btn-sm"
            title="为这个说话人添加新的声纹样本"
            @click="addTarget = speaker"
          >
            <AppIcon name="mic" :size="14" />
            添加声纹
          </button>
          <button class="btn btn-ghost btn-sm" title="重命名" @click="openRename(speaker)">
            <AppIcon name="edit" :size="14" />
            重命名
          </button>
          <button class="btn btn-danger btn-sm" title="删除" @click="removeSpeaker(speaker)">
            <AppIcon name="trash" :size="14" />
          </button>
        </div>

        <div v-if="speaker.vectors.length > 0" class="vector-list">
          <div v-for="v in speaker.vectors" :key="v.id" class="vector-row">
            <AppIcon name="list" :size="13" class="vector-icon" />
            <span class="vector-src">
              {{ v.source === 'recording' ? '现场录音' : '会议片段' }}
            </span>
            <span class="muted-text">{{ v.duration_s.toFixed(1) }} 秒</span>
            <span v-if="v.source === 'meeting' && v.meeting_id" class="muted-text">
              会议 {{ v.meeting_id.slice(0, 8) }} · 片段 #{{ v.segment_ids.join(', ') }}
            </span>
            <span v-if="v.text" class="muted-text vector-text">{{ v.text }}</span>
            <span v-if="v.short" class="warn-badge" title="语音过短，声纹特征可能不准确">
              <AppIcon name="alert" :size="12" />
              过短
            </span>
            <span class="vector-date">{{ fmtDate(v.created_at) }}</span>
            <button
              class="icon-btn danger"
              title="删除这条声纹"
              @click="removeVector(speaker, v.id)"
            >
              <AppIcon name="trash" :size="13" />
            </button>
          </div>
        </div>
        <p v-else class="hint-text">还没有声纹样本，建议尽快添加。</p>
      </article>
    </div>

    <div v-if="renameTarget" class="dialog-backdrop" @click.self="renameTarget = null">
      <div class="dialog">
        <h3>重命名说话人</h3>
        <p class="dialog-sub">修改后，会议记录中已标注的姓名会随之更新。</p>
        <input
          v-model="renameName"
          class="text-input"
          placeholder="说话人姓名"
          autofocus
          @keyup.enter="renameSpeaker"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="renameTarget = null">取消</button>
          <button
            class="btn btn-primary"
            :disabled="!renameName.trim() || renaming"
            @click="renameSpeaker"
          >
            {{ renaming ? '保存中…' : '保存' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="addTarget" class="dialog-backdrop" @click.self="addTarget = null">
      <div class="dialog dialog-wide">
        <h3>为「{{ addTarget.name }}」添加声纹</h3>
        <p class="dialog-sub">现场录音或从已保存会议中选取片段，为这个说话人追加声纹样本。</p>
        <SpeakerVectorAdder
          :key="addTarget.id"
          :speaker="addTarget"
          @added="onVectorsAdded"
          @error="error = $event"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="addTarget = null">关闭</button>
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
