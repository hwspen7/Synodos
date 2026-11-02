export interface ServerReady {
  type: 'ready'
  session_id: string
  sample_rate: number
  vad_threshold_s: number
  rec_threshold_s: number
  merge_gap_s: number
  meeting_id?: string
  resumed?: boolean
  title?: string
  participant_ids?: string[]
  participants?: Array<{ id: string; name: string }>
  hotword_ids?: string[]
  hotwords?: Array<{ id: string; name: string; word_count: number }>
  spk_threshold?: number
  min_speaker_audio_s?: number
}

export interface TranscriptSegment {
  id: number
  speaker_id: number
  /** 展示用说话人名称：参与者姓名 / 未知说话人N / 说话人N */
  speaker_name?: string
  start_ms: number
  end_ms: number
  text: string
}

/** 后端下发的完整 segment 快照：只下发可信段，段落由前端本地分组。 */
export interface ServerTranscript {
  type: 'transcript'
  version: number
  segments: TranscriptSegment[]
}

export interface ServerState {
  type: 'state'
  buffered_ms: number
}

export interface ServerClosed {
  type: 'closed'
}

export type ServerMessage = ServerReady | ServerTranscript | ServerState | ServerClosed

/** 前端展示用的可信段：附上说话人颜色。 */
export interface Segment extends TranscriptSegment {
  color: string
}

/**
 * 前端本地由可信段分组生成的段落：
 * 同一说话人（speaker_id）且相邻段间隔小于 merge_gap_s 才合并。
 */
export interface Paragraph {
  /** 段落 id：等于段落第一个可信段的 id，分组结构变化时随之变化 */
  id: number
  speaker_id: number
  speaker_name?: string
  start_ms: number
  end_ms: number
  text: string
  segmentIds: number[]
  color: string
}

/** 已保存到磁盘的会议记录摘要 */
export interface MeetingSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
  sample_rate: number
  merge_gap_s: number
  segments: number
  duration_ms: number
  speakers: number
}

/** 会议详情：meta + 参与者设置（用于初始化“会议参与者”选择） */
export interface MeetingDetail {
  id: string
  title: string
  created_at?: string
  updated_at?: string
  participant_ids: string[]
  hotword_ids: string[]
  spk_threshold: number
}

/** 说话人的一条声纹向量（元数据，不含向量本身） */
export interface SpeakerVector {
  id: string
  duration_s: number
  source: 'recording' | 'meeting'
  meeting_id?: string
  segment_ids: number[]
  text?: string
  created_at: string
  /** 时长低于 min_speaker_audio_s，声纹可能不准确 */
  short: boolean
}

export interface Speaker {
  id: string
  name: string
  vector_count: number
  created_at: string
  updated_at: string
  vectors: SpeakerVector[]
}

/** 热词库：一组热词，会议开始时按库选择后注入 ASR */
export interface HotwordLibrary {
  id: string
  name: string
  words: string[]
  word_count: number
  created_at: string
  updated_at: string
}

/** 会议中的一个可信段摘要（用于导入说话人声纹时选择） */
export interface MeetingSegment {
  id: number
  text: string
  start_ms: number
  end_ms: number
  duration_s: number
}

export interface HealthInfo {
  status: string
  sample_rate: number
  min_speaker_audio_s: number
  spk_match_threshold: number
  [key: string]: unknown
}
