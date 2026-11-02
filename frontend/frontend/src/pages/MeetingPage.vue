<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { downloadUrl, hotwordsApi, meetingsApi, speakersApi } from '@/api/client'
import { RealtimeClient } from '@/api/ws'
import { startMicRecording } from '@/audio/recorder'
import AppIcon from '@/components/AppIcon.vue'
import { speakerColor } from '@/speaker/clustering'
import type {
  HotwordLibrary,
  Paragraph,
  Segment,
  ServerReady,
  ServerTranscript,
  Speaker,
} from '@/types'

const props = defineProps<{ id: string }>()
const route = useRoute()
const router = useRouter()

const WS_BASE = (import.meta.env.VITE_WS_URL as string | undefined) ?? ''

type Status = 'idle' | 'connecting' | 'recording' | 'stopping' | 'disconnected'
type ViewMode = 'segments' | 'paragraph'

const recording = ref(false)
const status = ref<Status>('idle')
const error = ref('')
const toast = ref('')
const vadThreshold = ref(10)
const recThreshold = ref(2)
const serverInfo = ref<ServerReady | null>(null)
const meetingTitle = ref('')
const bufferedMs = ref(0)
const segments = ref<Segment[]>([])
const viewMode = ref<ViewMode>('segments')
const mergeGapS = ref(2)
const transcriptMeta = ref({ paragraphs: 0, segments: 0 })
const transcriptEl = ref<HTMLElement | null>(null)
const saving = ref(false)
const speakers = ref<Speaker[]>([])
const hotwords = ref<HotwordLibrary[]>([])
const showParticipants = ref(false)
const selectedParticipantIds = ref<string[]>([])
const showHotwords = ref(false)
const selectedHotwordIds = ref<string[]>([])
const spkThreshold = ref(0.3)
const applying = ref(false)
const micLevel = ref(0)
const micSilent = ref(false)

let client: RealtimeClient | null = null
let recorder: Awaited<ReturnType<typeof startMicRecording>> | null = null
let clientSeq = 0
let stickToBottom = true
let stopRequested = false

const statusText = computed(() => {
  switch (status.value) {
    case 'connecting':
      return '连接中…'
    case 'recording':
      return '录音中'
    case 'stopping':
      return '收尾中…'
    case 'disconnected':
      return '连接已断开'
    default:
      return '未开始'
  }
})

const statusClass = computed(() => {
  if (recording.value || status.value === 'stopping') return 'ok'
  return status.value === 'disconnected' ? 'bad' : 'muted'
})

/** 段落视图由片段在本地分组生成：同说话人 + 相邻间隔小于 merge_gap。 */
const paragraphs = computed<Paragraph[]>(() => buildParagraphs(segments.value))
const displayed = computed<Array<Segment | Paragraph>>(() =>
  viewMode.value === 'paragraph' ? paragraphs.value : segments.value,
)
const participantNames = computed(() => {
  const byId = new Map(speakers.value.map((s) => [s.id, s.name]))
  return selectedParticipantIds.value
    .map((id) => byId.get(id))
    .filter((n): n is string => Boolean(n))
})
const hotwordNames = computed(() => {
  const byId = new Map(hotwords.value.map((h) => [h.id, h.name]))
  return selectedHotwordIds.value
    .map((id) => byId.get(id))
    .filter((n): n is string => Boolean(n))
})

function wsUrl(): string {
  const params = new URLSearchParams({
    vad_threshold: String(vadThreshold.value),
    rec_threshold: String(recThreshold.value),
    meeting_id: props.id,
    participants: selectedParticipantIds.value.join(','),
    hotwords: selectedHotwordIds.value.join(','),
    spk_threshold: String(spkThreshold.value),
  })
  if (WS_BASE) {
    const separator = WS_BASE.includes('?') ? '&' : '?'
    return `${WS_BASE}${separator}${params.toString()}`
  }
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${scheme}://${location.host}/ws/audio?${params.toString()}`
}

function showToast(message: string): void {
  toast.value = message
  window.setTimeout(() => {
    if (toast.value === message) toast.value = ''
  }, 2600)
}

function mergeGapMs(): number {
  return Math.round((mergeGapS.value || 2) * 1000)
}

/** 由可信段生成段落：同一说话人 + 间隔小于 merge_gap 才合并。 */
function buildParagraphs(segs: Segment[]): Paragraph[] {
  const ordered = [...segs].sort((a, b) => a.start_ms - b.start_ms)
  const gap = mergeGapMs()
  const paras: Paragraph[] = []
  for (const seg of ordered) {
    const last = paras[paras.length - 1]
    if (
      last &&
      seg.speaker_id === last.speaker_id &&
      seg.start_ms - last.end_ms < gap
    ) {
      last.end_ms = seg.end_ms
      last.segmentIds.push(seg.id)
      last.text += seg.text
    } else {
      paras.push({
        id: seg.id,
        speaker_id: seg.speaker_id,
        speaker_name: seg.speaker_name,
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        text: seg.text,
        segmentIds: [seg.id],
        color: seg.color,
      })
    }
  }
  return paras
}

/** 后端每次聚类结果变化都会推送完整 segment 快照，这里整体替换片段列表。 */
function applyTranscript(msg: ServerTranscript): void {
  const segs: Segment[] = msg.segments.map((s) => ({
    ...s,
    color: speakerColor(s.speaker_id),
  }))
  segments.value = segs
  transcriptMeta.value = {
    segments: msg.segments.length,
    paragraphs: buildParagraphs(segs).length,
  }
  void scrollIfNeeded()
}

function toggleView(): void {
  viewMode.value = viewMode.value === 'paragraph' ? 'segments' : 'paragraph'
  void scrollIfNeeded()
}

async function saveMeeting(quiet = false): Promise<boolean> {
  saving.value = true
  try {
    await meetingsApi.save(props.id)
    if (!quiet) showToast('已保存到磁盘')
    return true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    return false
  } finally {
    saving.value = false
  }
}

/** 读取会议已保存的参与者配置 + 声纹库列表（连接前初始化）。 */
async function loadMeetingConfig(): Promise<void> {
  try {
    const { meeting } = await meetingsApi.get(props.id)
    selectedParticipantIds.value = meeting.participant_ids ?? []
    selectedHotwordIds.value = meeting.hotword_ids ?? []
    spkThreshold.value = meeting.spk_threshold ?? 0.3
  } catch {
    // 新会议尚无配置，使用默认值（阈值 0.3、不选参与者/热词库）
    selectedParticipantIds.value = []
    selectedHotwordIds.value = []
    spkThreshold.value = 0.3
  }
  try {
    speakers.value = (await speakersApi.list()).speakers
    hotwords.value = (await hotwordsApi.list()).hotwords
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function toggleParticipant(id: string): void {
  const next = new Set(selectedParticipantIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedParticipantIds.value = [...next]
}

function toggleHotword(id: string): void {
  const next = new Set(selectedHotwordIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  selectedHotwordIds.value = [...next]
}

/** 应用参与者配置：保存到会议并重连（重连后后端按新配置重算标签）。 */
async function applyParticipants(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showParticipants.value = false
    await saveMeeting(true)
    connect()
    showToast('已应用参与者设置')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    applying.value = false
  }
}

/** 应用热词库配置：保存到会议并重连（重连后后端按新热词识别）。 */
async function applyHotwords(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showHotwords.value = false
    await saveMeeting(true)
    connect()
    showToast('已应用热词库设置')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    applying.value = false
  }
}

async function exportMeeting(format: 'md' | 'json'): Promise<void> {
  const ok = await saveMeeting(true)
  if (!ok) return
  downloadUrl(meetingsApi.exportUrl(props.id, format))
  showToast(format === 'md' ? '已导出 Markdown 文件' : '已导出 JSON 文件')
}

async function copyTranscript(): Promise<void> {
  const text = displayed.value
    .map((item) => `[${item.speaker_name || `说话人${item.speaker_id}`}] ${item.text}`)
    .join('\n')
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
  } catch {
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
  }
  showToast('已复制全文')
}

function clearTranscript(): void {
  segments.value = []
  transcriptMeta.value = { paragraphs: 0, segments: 0 }
}

async function startMic(): Promise<void> {
  if (!client?.connected || recording.value) return
  error.value = ''
  micLevel.value = 0
  micSilent.value = false
  let silentCount = 0
  try {
    recorder = await startMicRecording(
      (pcm) => client?.sendPcm(pcm),
      16000,
      (level) => {
        micLevel.value = level
        // 连续静音约 1.5 秒才提示（每块约 2.7ms，550 块 ≈ 1.5s）
        if (level < 0.0003) {
          silentCount++
          if (silentCount > 550) micSilent.value = true
        } else {
          silentCount = 0
          micSilent.value = false
        }
      },
    )
    recording.value = true
    status.value = 'recording'
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    status.value = serverInfo.value ? 'idle' : status.value
  }
}

function start(): void {
  if (recording.value || status.value === 'connecting' || status.value === 'stopping') return
  if (client?.connected) {
    void startMic()
    return
  }
  // 连接已断开（例如刚结束录音）：重新连接同一会议，恢复已有记录并自动继续录音
  error.value = ''
  connect(true)
}

function stop(): void {
  // 停麦后让服务端收尾：处理剩余音频、自动保存、推送最终 transcript 后由服务端关闭
  recording.value = false
  stopRequested = true
  recorder?.stop()
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  status.value = 'stopping'
  client?.sendStop()
}

function stopInternal(): void {
  recording.value = false
  recorder?.stop()
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  client?.close()
  client = null
}

function connect(autostart = false): void {
  error.value = ''
  clearTranscript()
  bufferedMs.value = 0
  serverInfo.value = null
  stickToBottom = true
  stopRequested = false

  const mySeq = ++clientSeq
  client = new RealtimeClient({
    url: wsUrl(),
    onReady: (msg) => {
      if (clientSeq !== mySeq) return
      serverInfo.value = msg
      meetingTitle.value = msg.title ?? '未命名会议'
      mergeGapS.value = msg.merge_gap_s || 2
      if (msg.participant_ids) selectedParticipantIds.value = msg.participant_ids
      if (msg.hotword_ids) selectedHotwordIds.value = msg.hotword_ids
      if (msg.spk_threshold !== undefined) spkThreshold.value = msg.spk_threshold
      status.value = 'idle'
      // 从首页「新建/继续」进入，或结束录音后再次点击「开始录音」时自动开始录音
      if (autostart || route.query.autostart === '1') void startMic()
    },
    onTranscript: (msg) => {
      if (clientSeq !== mySeq) return
      applyTranscript(msg)
    },
    onState: (ms) => {
      if (clientSeq !== mySeq) return
      bufferedMs.value = ms
    },
    onClosed: (reason) => {
      if (clientSeq !== mySeq) return
      client = null
      recording.value = false
      recorder?.stop()
      recorder = null
      if (reason === 'closed') {
        status.value = 'idle'
        if (stopRequested) showToast('已停止并自动保存到磁盘')
      } else {
        status.value = 'disconnected'
      }
      stopRequested = false
    },
  })
  client.connect()
  status.value = 'connecting'
}

function onTranscriptScroll(): void {
  const el = transcriptEl.value
  if (!el) return
  stickToBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 80
}

async function scrollIfNeeded(): Promise<void> {
  if (!stickToBottom) return
  await nextTick()
  const el = transcriptEl.value
  if (el) el.scrollTop = el.scrollHeight
}

function formatTime(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(s).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

onMounted(async () => {
  await loadMeetingConfig()
  connect()
})

onBeforeUnmount(() => stopInternal())
</script>

<template>
  <div class="meeting-page">
    <div class="meeting-head">
      <div class="meeting-head-left">
        <button class="btn btn-ghost btn-sm" @click="router.push('/')">
          <AppIcon name="arrow-left" :size="16" />
          返回首页
        </button>
        <div>
          <h1 class="meeting-title">{{ meetingTitle || '未命名会议' }}</h1>
          <p class="meeting-sub">
            <span class="status-chip" :class="statusClass">
              <span class="status-dot" aria-hidden="true"></span>{{ statusText }}
            </span>
            <span v-if="recording" class="mic-level" :title="`麦克风电平 ${(micLevel * 100).toFixed(1)}%`">
              <span class="mic-level-bar" :style="{ width: Math.min(100, micLevel * 300) + '%' }"></span>
            </span>
            <span v-if="serverInfo">缓冲 {{ (bufferedMs / 1000).toFixed(1) }}s</span>
            <span>片段 {{ transcriptMeta.segments }} · 段落 {{ transcriptMeta.paragraphs }}</span>
            <span v-if="participantNames.length > 0" class="participant-chip">
              <AppIcon name="users" :size="13" />
              参与者：{{ participantNames.join('、') }}
            </span>
            <span v-if="hotwordNames.length > 0" class="hotword-chip">
              <AppIcon name="flame" :size="13" />
              热词库：{{ hotwordNames.join('、') }}
            </span>
          </p>
        </div>
      </div>
      <div class="meeting-actions">
        <button class="btn btn-ghost" :disabled="saving" @click="saveMeeting()">
          <AppIcon name="save" :size="16" />
          {{ saving ? '保存中…' : '保存会议' }}
        </button>
        <button class="btn btn-ghost" @click="exportMeeting('md')">
          <AppIcon name="download" :size="16" />
          导出 MD
        </button>
        <button class="btn btn-ghost" @click="exportMeeting('json')">
          <AppIcon name="file-text" :size="16" />
          导出 JSON
        </button>
        <button class="btn btn-primary" :class="{ danger: recording }" @click="recording ? stop() : start()">
          <AppIcon :name="recording ? 'stop' : 'mic'" :size="16" />
          {{ recording ? '停止录音' : '开始录音' }}
        </button>
      </div>
    </div>

    <div class="meeting-controls">
      <label class="field" title="缓冲音频达到该时长时触发一轮识别处理">
        处理阈值（秒）
        <input
          v-model.number="vadThreshold"
          type="number"
          min="0.5"
          max="30"
          step="0.5"
          :disabled="recording"
        />
      </label>
      <label class="field" title="句子结束点距缓冲区末尾超过该时长才确认为可信段">
        确认阈值（秒）
        <input
          v-model.number="recThreshold"
          type="number"
          min="0.2"
          max="30"
          step="0.5"
          :disabled="recording"
        />
      </label>
      <button
        class="btn btn-ghost"
        :class="{ active: viewMode === 'paragraph' }"
        :disabled="segments.length === 0"
        @click="toggleView"
      >
        {{ viewMode === 'paragraph' ? '片段视图' : '段落视图' }}
      </button>
      <button class="btn btn-ghost" :disabled="displayed.length === 0" @click="copyTranscript">
        复制全文
      </button>
      <button
        class="btn btn-ghost"
        :title="recording ? '停止录音后才能修改参与者' : '选择声纹库中的说话人作为会议参与者'"
        @click="showParticipants = true"
      >
        <AppIcon name="users" :size="15" />
        参与者{{ selectedParticipantIds.length > 0 ? `（${selectedParticipantIds.length} 人 · ${spkThreshold}）` : '' }}
      </button>
      <button
        class="btn btn-ghost"
        :title="recording ? '停止录音后才能修改热词库' : '选择热词库，识别时优先输出热词'"
        @click="showHotwords = true"
      >
        <AppIcon name="flame" :size="15" />
        热词库{{ selectedHotwordIds.length > 0 ? `（${selectedHotwordIds.length} 个）` : '' }}
      </button>
      <button class="btn btn-ghost" :disabled="segments.length === 0" @click="clearTranscript">
        清空
      </button>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>

    <main ref="transcriptEl" class="transcript" @scroll="onTranscriptScroll">
      <div v-if="segments.length === 0" class="empty-state">
        <div class="empty-icon"><AppIcon name="mic" :size="26" /></div>
        <h3>{{ recording ? '正在聆听…' : '还没有内容' }}</h3>
        <p v-if="!recording">点击「开始录音」并允许麦克风权限；说完一句话并停顿 2 秒以上即可生成片段。</p>
        <p v-else>说完一句话并停顿 2 秒以上即可生成片段。</p>
        <p v-if="recording && micSilent" class="hint-text warn-text" style="margin-top:8px">
          未检测到麦克风输入，请检查浏览器是否已授权麦克风、系统麦克风是否开启且未静音。
        </p>
      </div>

      <template v-else-if="viewMode === 'segments'">
        <article v-for="seg in segments" :key="seg.id" class="segment">
          <div class="para-head">
            <span class="speaker-dot" :style="{ background: seg.color }"></span>
            <span class="speaker-name">{{ seg.speaker_name || `说话人${seg.speaker_id}` }}</span>
            <span class="para-time">{{ formatTime(seg.start_ms) }} – {{ formatTime(seg.end_ms) }}</span>
          </div>
          <p class="para-text">{{ seg.text || '（此段暂无内容）' }}</p>
        </article>
      </template>

      <template v-else>
        <article v-for="para in paragraphs" :key="para.id" class="paragraph">
          <div class="para-head">
            <span class="speaker-dot" :style="{ background: para.color }"></span>
            <span class="speaker-name">{{ para.speaker_name || `说话人${para.speaker_id}` }}</span>
            <span v-if="para.segmentIds.length > 1" class="seg-count">
              {{ para.segmentIds.length }} 段
            </span>
            <span class="para-time">{{ formatTime(para.start_ms) }} – {{ formatTime(para.end_ms) }}</span>
          </div>
          <p class="para-text">{{ para.text || '（此段暂无内容）' }}</p>
        </article>
      </template>
    </main>

    <div v-if="showParticipants" class="dialog-backdrop" @click.self="showParticipants = false">
      <div class="dialog dialog-wide">
        <h3>会议参与者</h3>
        <p class="dialog-sub">
          选中声纹库中的说话人作为参与者。识别时每个片段先与参与者的全部声纹做余弦匹配，
          相似度达到阈值即标注为该参与者姓名，其余片段进入聚类并标记为“未知说话人N”。
        </p>
        <div class="participant-list">
          <label
            v-for="spk in speakers"
            :key="spk.id"
            class="participant-row"
            :class="{ checked: selectedParticipantIds.includes(spk.id) }"
          >
            <input
              type="checkbox"
              :checked="selectedParticipantIds.includes(spk.id)"
              @change="toggleParticipant(spk.id)"
            />
            <span class="speaker-avatar">{{ spk.name.slice(0, 1) }}</span>
            <span class="participant-name">{{ spk.name }}</span>
            <span class="muted-text">{{ spk.vector_count }} 条声纹</span>
          </label>
          <div v-if="speakers.length === 0" class="empty-state small">
            <p>声纹库为空，请先在首页「新建说话人」中添加声纹样本。</p>
          </div>
        </div>
        <div class="threshold-row">
          <label class="field">
            匹配阈值
            <input
              v-model.number="spkThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
            />
          </label>
          <span class="muted-text">
            相似度 ≥ 阈值时直接标注为参与者（默认 0.3）；调低更容易命中，调高则更多进入未知说话人。
          </span>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          录音中不能修改参与者，请先停止录音。
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showParticipants = false">取消</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyParticipants"
          >
            {{ applying ? '应用中…' : '应用并重连' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showHotwords" class="dialog-backdrop" @click.self="showHotwords = false">
      <div class="dialog dialog-wide">
        <h3>会议热词库</h3>
        <p class="dialog-sub">
          选中热词库后，识别时引擎会优先输出库中的热词，适合包含大量专有名词、
          人名或产品名的会议。可同时选择多个热词库（同一会议最多 200 个热词）。
        </p>
        <div class="hotword-pick-list">
          <label
            v-for="lib in hotwords"
            :key="lib.id"
            class="hotword-pick-row"
            :class="{ checked: selectedHotwordIds.includes(lib.id) }"
          >
            <input
              type="checkbox"
              :checked="selectedHotwordIds.includes(lib.id)"
              @change="toggleHotword(lib.id)"
            />
            <span class="hotword-pick-icon"><AppIcon name="flame" :size="15" /></span>
            <span class="hotword-pick-info">
              <span class="hotword-pick-name">
                {{ lib.name }}
                <span class="muted-text">{{ lib.word_count }} 个热词</span>
              </span>
              <span v-if="lib.words.length > 0" class="hotword-tags compact">
                <span
                  v-for="w in lib.words.slice(0, 5)"
                  :key="w"
                  class="hotword-tag"
                >
                  {{ w }}
                </span>
                <span v-if="lib.words.length > 5" class="hotword-more">
                  +{{ lib.words.length - 5 }}
                </span>
              </span>
            </span>
          </label>
          <div v-if="hotwords.length === 0" class="empty-state small">
            <p>还没有热词库，请先在首页「新建热词库」中添加。</p>
          </div>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          录音中不能修改热词库，请先停止录音。
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showHotwords = false">取消</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyHotwords"
          >
            {{ applying ? '应用中…' : '应用并重连' }}
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
