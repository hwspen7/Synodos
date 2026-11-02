<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { downloadUrl, hotwordsApi, meetingsApi, speakersApi } from '@/api/client'
import AppIcon from '@/components/AppIcon.vue'
import type { HotwordLibrary, MeetingSummary, Speaker } from '@/types'

const router = useRouter()

const meetings = ref<MeetingSummary[]>([])
const speakers = ref<Speaker[]>([])
const hotwords = ref<HotwordLibrary[]>([])
const loading = ref(true)
const error = ref('')
const showCreate = ref(false)
const newTitle = ref('')
const creating = ref(false)

const totalDurationMs = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.duration_ms ?? 0), 0),
)
const totalSegments = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.segments ?? 0), 0),
)
const totalSpeakers = computed(() =>
  meetings.value.reduce((sum, m) => sum + (m.speakers ?? 0), 0),
)
const totalSpeakerVectors = computed(() =>
  speakers.value.reduce((sum, s) => sum + (s.vector_count ?? 0), 0),
)
const totalHotwordWords = computed(() =>
  hotwords.value.reduce((sum, h) => sum + (h.word_count ?? 0), 0),
)

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    meetings.value = (await meetingsApi.list()).meetings
    speakers.value = (await speakersApi.list()).speakers
    hotwords.value = (await hotwordsApi.list()).hotwords
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function createMeeting(): Promise<void> {
  const title = newTitle.value.trim()
  if (!title || creating.value) return
  creating.value = true
  error.value = ''
  try {
    const { meeting } = await meetingsApi.create(title)
    showCreate.value = false
    newTitle.value = ''
    await router.push({ name: 'meeting', params: { id: meeting.id }, query: { autostart: '1' } })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    creating.value = false
  }
}

function viewMeeting(meeting: MeetingSummary): void {
  void router.push(`/meeting/${meeting.id}`)
}

function resumeMeeting(meeting: MeetingSummary): void {
  void router.push(`/meeting/${meeting.id}?autostart=1`)
}

function exportMeeting(meeting: MeetingSummary, format: 'md' | 'json'): void {
  downloadUrl(meetingsApi.exportUrl(meeting.id, format))
}

async function removeMeeting(meeting: MeetingSummary): Promise<void> {
  if (!window.confirm(`确定删除会议「${meeting.title}」？此操作不可恢复。`)) return
  error.value = ''
  try {
    await meetingsApi.remove(meeting.id)
    await load()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function fmtDuration(ms: number): string {
  const total = Math.max(0, Math.floor((ms ?? 0) / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h} 小时 ${m} 分`
  if (m > 0) return `${m} 分 ${s} 秒`
  return `${s} 秒`
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

onMounted(load)
</script>

<template>
  <div class="home">
    <section class="hero">
      <div class="hero-inner">
        <p class="eyebrow">MEETING OF PIONEERS</p>
        <h1>
          让每一场会议
          <span class="hero-accent">都留下清晰印记</span>
        </h1>
        <p class="lead">
          实时语音识别 · 自动说话人分段 · 一键成文。
          打开麦克风，专注交谈，剩下的交给我们。
        </p>
        <div class="hero-actions">
          <button class="btn btn-primary btn-lg" @click="showCreate = true">
            <AppIcon name="plus" :size="18" />
            新建会议记录
          </button>
          <a class="btn btn-ghost btn-lg" href="#meetings">
            <AppIcon name="file-text" :size="18" />
            浏览已保存会议
          </a>
          <RouterLink class="btn btn-ghost btn-lg" to="/speakers/new">
            <AppIcon name="user-plus" :size="18" />
            新建说话人
          </RouterLink>
          <RouterLink class="btn btn-ghost btn-lg" to="/speakers">
            <AppIcon name="users" :size="18" />
            管理说话人
          </RouterLink>
        </div>
      </div>
    </section>

    <main class="page-shell">
      <div class="metric-grid">
        <div class="metric-tile">
          <div class="metric-icon tone-primary"><AppIcon name="file-text" :size="20" /></div>
          <div>
            <div class="metric-label">会议记录</div>
            <div class="metric-value">{{ meetings.length }}</div>
          </div>
        </div>
        <div class="metric-tile">
          <div class="metric-icon tone-secondary"><AppIcon name="clock" :size="20" /></div>
          <div>
            <div class="metric-label">累计时长</div>
            <div class="metric-value">{{ fmtDuration(totalDurationMs) }}</div>
          </div>
        </div>
        <div class="metric-tile">
          <div class="metric-icon tone-accent"><AppIcon name="mic" :size="20" /></div>
          <div>
            <div class="metric-label">语音片段</div>
            <div class="metric-value">{{ totalSegments }}</div>
          </div>
        </div>
      <div class="metric-tile">
        <div class="metric-icon tone-success"><AppIcon name="users" :size="20" /></div>
        <div>
          <div class="metric-label">累计说话人</div>
          <div class="metric-value">{{ totalSpeakers }}</div>
        </div>
      </div>
      </div>

      <section class="speaker-lib-section">
        <div class="speaker-lib-card">
          <div class="speaker-lib-icon"><AppIcon name="users" :size="24" /></div>
          <div class="speaker-lib-info">
            <h2>说话人声纹库</h2>
            <p class="section-sub">
              现场录音或从会议片段提取声纹，为每个说话人保存多条特征向量；
              开会时选中参与者即可自动标注姓名。
            </p>
          </div>
          <div class="speaker-lib-stats">
            <div>
              <div class="metric-label">说话人</div>
              <div class="metric-value">{{ speakers.length }}</div>
            </div>
            <div>
              <div class="metric-label">声纹样本</div>
              <div class="metric-value">{{ totalSpeakerVectors }}</div>
            </div>
          </div>
          <div class="speaker-lib-actions">
            <button class="btn btn-primary" @click="router.push('/speakers/new')">
              <AppIcon name="user-plus" :size="16" />
              新建说话人
            </button>
            <button class="btn btn-ghost" @click="router.push('/speakers')">
              <AppIcon name="list" :size="16" />
              管理说话人
            </button>
          </div>
        </div>
      </section>

      <section class="hotword-lib-section">
        <div class="hotword-lib-card">
          <div class="hotword-lib-icon"><AppIcon name="flame" :size="24" /></div>
          <div class="hotword-lib-info">
            <h2>热词库</h2>
            <p class="section-sub">
              为专业术语、人名、产品名等建立热词库；开会时选中对应热词库，
              识别引擎会优先输出这些热词，提升专有名词的识别准确率。
            </p>
          </div>
          <div class="hotword-lib-stats">
            <div>
              <div class="metric-label">热词库</div>
              <div class="metric-value">{{ hotwords.length }}</div>
            </div>
            <div>
              <div class="metric-label">热词</div>
              <div class="metric-value">{{ totalHotwordWords }}</div>
            </div>
          </div>
          <div class="hotword-lib-actions">
            <button class="btn btn-primary" @click="router.push('/hotwords/new')">
              <AppIcon name="flame" :size="16" />
              新建热词库
            </button>
            <button class="btn btn-ghost" @click="router.push('/hotwords')">
              <AppIcon name="list" :size="16" />
              管理热词库
            </button>
          </div>
        </div>
      </section>

      <p v-if="error" class="error-banner">{{ error }}</p>

      <section id="meetings" class="meetings-section">
        <div class="toolbar-row">
          <div>
            <h2>已保存的会议</h2>
            <p class="section-sub">保存到磁盘的记录，可继续记录、查看或导出。</p>
          </div>
          <button class="btn btn-primary" @click="showCreate = true">
            <AppIcon name="plus" :size="16" />
            新建会议
          </button>
        </div>

        <div v-if="loading" class="empty-state">
          <p>正在加载会议记录…</p>
        </div>
        <div v-else-if="meetings.length === 0" class="empty-state">
          <div class="empty-icon"><AppIcon name="mic" :size="26" /></div>
          <h3>还没有会议记录</h3>
          <p>点击「新建会议记录」，打开麦克风开始第一场会议。</p>
          <button class="btn btn-primary" @click="showCreate = true">
            <AppIcon name="plus" :size="16" />
            新建会议记录
          </button>
        </div>
        <div v-else class="meeting-list">
          <article v-for="meeting in meetings" :key="meeting.id" class="meeting-card">
            <div class="meeting-main" @click="viewMeeting(meeting)">
              <h3 class="meeting-title">{{ meeting.title }}</h3>
              <p class="meeting-meta">
                更新于 {{ fmtDate(meeting.updated_at) }}
                <span class="dot-sep">·</span>
                时长 {{ fmtDuration(meeting.duration_ms) }}
                <span class="dot-sep">·</span>
                {{ meeting.segments }} 个片段
                <span class="dot-sep">·</span>
                {{ meeting.speakers }} 位说话人
              </p>
            </div>
            <div class="meeting-actions">
              <button class="btn btn-ghost btn-sm" title="查看记录" @click="viewMeeting(meeting)">
                <AppIcon name="edit" :size="15" />
                查看
              </button>
              <button
                class="btn btn-primary btn-sm"
                title="从末尾继续记录"
                @click="resumeMeeting(meeting)"
              >
                <AppIcon name="play" :size="14" />
                继续记录
              </button>
              <div class="action-group">
                <button class="btn btn-ghost btn-sm" title="导出 Markdown" @click="exportMeeting(meeting, 'md')">
                  <AppIcon name="download" :size="15" />
                  MD
                </button>
                <button class="btn btn-ghost btn-sm" title="导出 JSON" @click="exportMeeting(meeting, 'json')">
                  <AppIcon name="file-text" :size="15" />
                  JSON
                </button>
                <button class="btn btn-danger btn-sm" title="删除" @click="removeMeeting(meeting)">
                  <AppIcon name="trash" :size="15" />
                </button>
              </div>
            </div>
          </article>
        </div>
      </section>
    </main>

    <div v-if="showCreate" class="dialog-backdrop" @click.self="showCreate = false">
      <div class="dialog">
        <h3>新建会议记录</h3>
        <p class="dialog-sub">给这场会议起个名字，稍后可以随时从末尾继续记录。</p>
        <input
          v-model="newTitle"
          class="text-input"
          placeholder="例如：产品周会 / 项目复盘"
          autofocus
          @keyup.enter="createMeeting"
        />
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showCreate = false">取消</button>
          <button
            class="btn btn-primary"
            :disabled="!newTitle.trim() || creating"
            @click="createMeeting"
          >
            {{ creating ? '创建中…' : '创建并开始记录' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
