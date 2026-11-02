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
  /* Audio processing and UI rendering. */
  speaker_name?: string
  start_ms: number
  end_ms: number
  text: string
}

/* Audio processing and UI rendering. */
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

/* Audio processing and UI rendering. */
export interface Segment extends TranscriptSegment {
  color: string
}

/**
  * Audio processing and UI rendering.
  * Audio processing and UI rendering.
 */
export interface Paragraph {
  /* Audio processing and UI rendering. */
  id: number
  speaker_id: number
  speaker_name?: string
  start_ms: number
  end_ms: number
  text: string
  segmentIds: number[]
  color: string
}

/* Audio processing and UI rendering. */
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

/* Audio processing and UI rendering. */
export interface MeetingDetail {
  id: string
  title: string
  created_at?: string
  updated_at?: string
  participant_ids: string[]
  hotword_ids: string[]
  spk_threshold: number
}

/* Audio processing and UI rendering. */
export interface SpeakerVector {
  id: string
  duration_s: number
  source: 'recording' | 'meeting'
  meeting_id?: string
  segment_ids: number[]
  text?: string
  created_at: string
  /* Audio processing and UI rendering. */
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

/* Audio processing and UI rendering. */
export interface HotwordLibrary {
  id: string
  name: string
  words: string[]
  word_count: number
  created_at: string
  updated_at: string
}

/* Audio processing and UI rendering. */
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
