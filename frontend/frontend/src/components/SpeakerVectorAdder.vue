<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { meetingsApi, speakersApi, systemApi } from '@/api/client'
import { startMicRecording } from '@/audio/recorder'
import AppIcon from '@/components/AppIcon.vue'
import type { MeetingSegment, MeetingSummary, Speaker, SpeakerVector } from '@/types'

const props = defineProps<{ speaker: Speaker }>()
const emit = defineEmits<{
  added: [vectors: SpeakerVector[]]
  error: [message: string]
}>()

type Tab = 'record' | 'meeting'

const error = ref('')
const tab = ref<Tab>('record')
const minAudioS = ref(5)
const addedVectors = ref<SpeakerVector[]>([])

// ── 现场录音 ────────────────────────────────────────────
const recording = ref(false)
const recDurationS = ref(0)
const recReady = ref(false)
const submittingRec = ref(false)
const pcmChunks: Int16Array[] = []
let recorder: Awaited<ReturnType<typeof startMicRecording>> | null = null

// ── 从会议选取 ──────────────────────────────────────────
const meetings = ref<MeetingSummary[]>([])
const selectedMeetingId = ref('')
const segments = ref<MeetingSegment[]>([])
const selectedSegIds = ref<Set<number>>(new Set())
const loadingSegments = ref(false)
const submittingSeg = ref(false)

const selectedCount = computed(() => selectedSegIds.value.size)
const totalSelectedDuration = computed(() =>
  segments.value
    .filter((s) => selectedSegIds.value.has(s.id))
    .reduce((sum, s) => sum + s.duration_s, 0),
)
const selectedShortCount = computed(
  () =>
    segments.value.filter(
      (s) => selectedSegIds.value.has(s.id) && s.duration_s < minAudioS.value,
    ).length,
)

function fmtDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// ── 现场录音 ────────────────────────────────────────────
async function startRecord(): Promise<void> {
  if (recording.value) return
  error.value = ''
  pcmChunks.length = 0
  recDurationS.value = 0
  recReady.value = false
  try {
    recorder = await startMicRecording((chunk) => {
      pcmChunks.push(chunk)
      const totalSamples = pcmChunks.reduce((sum, c) => sum + c.length, 0)
      recDurationS.value = totalSamples / 16000
    })
    recording.value = true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function stopRecord(): void {
  if (!recording.value) return
  recording.value = false
  recorder?.stop()
  recorder = null
  recReady.value = recDurationS.value >= 0.2
  if (!recReady.value) {
    error.value = '录音太短，无法提取声纹特征，请重新录制'
    pcmChunks.length = 0
    recDurationS.value = 0
  }
}

function pcmBlob(): Blob {
  // Int16Array 的 buffer 可能带额外字节，按视图实际字节范围构造 Blob
  return new Blob(
    pcmChunks.map((c) => c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength) as ArrayBuffer),
    { type: 'application/octet-stream' },
  )
}

async function submitRecording(): Promise<void> {
  if (submittingRec.value) return
  submittingRec.value = true
  error.value = ''
  try {
    const { vectors, min_audio_s } = await speakersApi.addRecord(
      props.speaker.id,
      pcmBlob(),
    )
    minAudioS.value = min_audio_s
    addedVectors.value = vectors
    emit('added', vectors)
    pcmChunks.length = 0
    recDurationS.value = 0
    recReady.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    emit('error', error.value)
  } finally {
    submittingRec.value = false
  }
}

// ── 从会议选取 ──────────────────────────────────────────
async function loadMeetingSegments(): Promise<void> {
  if (!selectedMeetingId.value) {
    segments.value = []
    return
  }
  loadingSegments.value = true
  error.value = ''
  try {
    const { segments: segs } = await meetingsApi.segments(selectedMeetingId.value)
    segments.value = segs
    selectedSegIds.value = new Set()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    segments.value = []
  } finally {
    loadingSegments.value = false
  }
}

function toggleSegment(id: number): void {
  const next = new Set(selectedSegIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedSegIds.value = next
}

async function submitMeetingSelection(): Promise<void> {
  if (submittingSeg.value || selectedCount.value === 0) return
  submittingSeg.value = true
  error.value = ''
  try {
    const { vectors, missing, min_audio_s } = await speakersApi.addFromMeeting(
      props.speaker.id,
      selectedMeetingId.value,
      [...selectedSegIds.value],
    )
    minAudioS.value = min_audio_s
    addedVectors.value = vectors
    emit('added', vectors)
    if (missing.length > 0) {
      error.value = `有 ${missing.length} 个片段没有可用的声纹向量，已跳过`
      emit('error', error.value)
    }
    selectedSegIds.value = new Set()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    emit('error', error.value)
  } finally {
    submittingSeg.value = false
  }
}

onMounted(async () => {
  try {
    const health = await systemApi.health()
    minAudioS.value = health.min_speaker_audio_s
  } catch {
    /* 保留默认 5 秒 */
  }
  try {
    const { meetings: list } = await meetingsApi.list()
    meetings.value = list
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
})

onBeforeUnmount(() => {
  recorder?.stop()
  recorder = null
})
</script>

<template>
  <div class="vector-adder">
    <div class="tab-bar">
      <button class="tab-btn" :class="{ active: tab === 'record' }" @click="tab = 'record'">
        <AppIcon name="mic" :size="15" />
        现场录音
      </button>
      <button class="tab-btn" :class="{ active: tab === 'meeting' }" @click="tab = 'meeting'">
        <AppIcon name="file-text" :size="15" />
        从会议选取
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <!-- 现场录音 -->
    <div v-if="tab === 'record'" class="record-area">
      <div class="read-card">
        <div class="read-card-head">
          <AppIcon name="mic" :size="15" />
          请朗读以下内容（建议 {{ minAudioS }} 秒以上）
        </div>
        <blockquote class="read-passage">
          网络开拓者协会（简称“网协”），成立于1997年，是目前北京理工大学校内规模最大的校级学生组织之一，同时也是校团委领导的唯一校级IT 类学生组织。本着“网络无限，开拓不懈”的精神，网协不断助力校园网络和信息化建设。
        </blockquote>
      </div>
      <button
        class="btn record-btn"
        :class="{ recording }"
        @click="recording ? stopRecord() : startRecord()"
      >
        <AppIcon :name="recording ? 'stop' : 'mic'" :size="18" />
        {{ recording ? '停止录音' : '开始录音' }}
      </button>
      <div v-if="recDurationS > 0 || recReady" class="rec-meta">
        <span class="rec-time">{{ recDurationS.toFixed(1) }} 秒</span>
        <span v-if="recReady && recDurationS < minAudioS" class="warn-badge">
          <AppIcon name="alert" :size="13" />
          语音过短（&lt; {{ minAudioS }} 秒），声纹特征可能不准确
        </span>
        <span v-else-if="recReady" class="ok-badge">录音完成</span>
      </div>
      <p class="hint-text">
        建议先通读一遍上方内容再开始录音；低于 {{ minAudioS }} 秒会提示声纹可能不准确，但也可以保存。
      </p>
      <button
        v-if="recReady"
        class="btn btn-primary"
        :disabled="submittingRec"
        @click="submitRecording"
      >
        <AppIcon name="check" :size="16" />
        {{ submittingRec ? '提取中…' : '提取声纹并保存' }}
      </button>
    </div>

    <!-- 从会议选取 -->
    <div v-else class="meeting-pick-area">
      <div class="field">
        选择会议
        <select v-model="selectedMeetingId" class="text-input" @change="loadMeetingSegments">
          <option value="" disabled>请选择已保存的会议…</option>
          <option v-for="m in meetings" :key="m.id" :value="m.id">
            {{ m.title }}（{{ m.segments }} 个片段）
          </option>
        </select>
      </div>

      <div v-if="loadingSegments" class="empty-state small">
        <p>正在加载片段…</p>
      </div>
      <div v-else-if="segments.length > 0" class="seg-select-list">
        <label
          v-for="seg in segments"
          :key="seg.id"
          class="seg-check-row"
          :class="{ checked: selectedSegIds.has(seg.id) }"
        >
          <input
            type="checkbox"
            :checked="selectedSegIds.has(seg.id)"
            @change="toggleSegment(seg.id)"
          />
          <span class="seg-text">{{ seg.text || '（此段暂无内容）' }}</span>
          <span class="seg-time">{{ fmtDuration(seg.start_ms) }} – {{ fmtDuration(seg.end_ms) }}</span>
          <span class="seg-duration">{{ seg.duration_s.toFixed(1) }}s</span>
          <span v-if="seg.duration_s < minAudioS" class="warn-badge">
            <AppIcon name="alert" :size="12" />
            过短
          </span>
        </label>
      </div>
      <div v-else-if="selectedMeetingId" class="empty-state small">
        <p>该会议还没有可信片段，无法导入声纹。</p>
      </div>

      <div v-if="selectedCount > 0" class="pick-summary">
        <span>已选 {{ selectedCount }} 个片段，共 {{ totalSelectedDuration.toFixed(1) }} 秒</span>
        <span v-if="selectedShortCount > 0" class="warn-badge">
          <AppIcon name="alert" :size="13" />
          其中 {{ selectedShortCount }} 个短于 {{ minAudioS }} 秒，声纹可能不准确
        </span>
      </div>
      <button
        v-if="selectedCount > 0"
        class="btn btn-primary"
        :disabled="submittingSeg"
        @click="submitMeetingSelection"
      >
        <AppIcon name="check" :size="16" />
        {{ submittingSeg ? '导入中…' : '导入声纹' }}
      </button>
    </div>

    <div v-if="addedVectors.length > 0" class="adder-results">
      <h3 class="adder-results-title">刚添加的声纹</h3>
      <div class="vector-result-list">
        <div v-for="v in addedVectors" :key="v.id" class="vector-result">
          <AppIcon name="check" :size="15" class="vector-ok" />
          <span>{{ v.source === 'recording' ? '现场录音' : '会议片段' }}</span>
          <span class="muted-text">{{ v.duration_s.toFixed(1) }} 秒</span>
          <span v-if="v.source === 'meeting' && v.segment_ids.length" class="muted-text">
            片段 #{{ v.segment_ids.join(', ') }}
          </span>
          <span v-if="v.short" class="warn-badge">
            <AppIcon name="alert" :size="12" />
            语音过短
          </span>
          <span v-if="v.text" class="muted-text vector-text">{{ v.text }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
