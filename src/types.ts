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
  auto_match_saved?: boolean
  auto_match_active?: boolean
  auto_match_profile_count?: number
  participants?: Array<{ id: string; name: string }>
  hotword_ids?: string[]
  hotwords?: Array<{ id: string; name: string; word_count: number }>
  spk_threshold?: number
  min_speaker_audio_s?: number
  asr_engine?: string
  asr_model?: string
  independent_diarization?: 'off' | 'nemotron3'
  diarization_mode?: string
  asr_language?: string
  translation_target?: string
  wav_recording_enabled?: boolean
  participant_enrollments?: Array<{ id: string; name: string; voiceprint_count: number }>
  active_hotword_count?: number
  processing_window_s?: number
  confirmation_delay_s?: number
  min_commit_context_s?: number
}

export interface TranscriptSegment {
  id: number
  speaker_id: number
  /* Audio processing and UI rendering. */
  speaker_name?: string
  start_ms: number
  end_ms: number
  text: string
  language?: string
  language_confidence?: number | null
  translation?: string
  timestamp_quality?: string
  /** Optional acoustic voice intervals, separate from the ASR text bounds. */
  speaker_activity_intervals_ms?: Array<[number, number]>
  speaker_activity_fraction?: number | null
  speaker_activity_time_source?: 'nemotron3_official_turns'
  /** Optional display-only word-conserving speaker cues; original text is canonical. */
  caption_cues?: Array<{
    speaker_id: number
    speaker_name: string
    start_ms: number
    end_ms: number
    text: string
    word_count: number
  }>
  overlap_intervals_ms?: Array<{
    start_ms: number
    end_ms: number
    speaker_ids: number[]
  }>
  caption_projection_status?: string
  speaker_diarization_source?: string
  /** A registered voiceprint is a biometric candidate, not human confirmation. */
  matched_profile_id?: string | null
  matched_profile_name?: string | null
  speaker_match_similarity?: number | null
  /** UI-only pointer to the editable canonical ASR segment. */
  source_segment_id?: number
  revision?: number
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

export interface ServerAudioProgress {
  type: 'audio_progress'
  samples: number
  last_sequence?: number | null
  durable_samples?: number | null
}

export interface ServerAudioReceived {
  type: 'audio_received'
  samples: number
  last_sequence?: number | null
  durable_samples?: number | null
  matched: boolean
}

export interface ServerClosed {
  type: 'closed'
  saved?: boolean
}

export interface ServerError {
  type: 'error'
  message: string
}

export interface ServerPartial {
  type: 'partial'
  text: string
  language: string
  revision: number
}
export interface ServerFeatureWarning {
  type: 'feature_warning'
  feature: string
  message: string
}
export interface ServerAudioAnalysis {
  type: 'audio_analysis'
  result: { emotion?: string | null, events?: string[], overlap?: Array<{start_ms: number, end_ms: number}>, quality?: { rms: number } }
}
export type ServerMessage = ServerReady | ServerTranscript | ServerState | ServerClosed | ServerError | ServerPartial | ServerFeatureWarning | ServerAudioAnalysis | ServerAudioReceived | ServerAudioProgress

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
  auto_match_saved?: boolean
  hotword_ids: string[]
  spk_threshold: number
  ui_settings?: {
    language?: string
    translate_to?: string
    record_audio?: boolean
    vad_threshold?: number
    rec_threshold?: number
    participant_ids?: string[]
    auto_match_saved?: boolean
    hotword_ids?: string[]
    spk_threshold?: number
  }
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
  embedding_model?: string
  human_verified?: boolean
  verification_source?: string | null
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
  caption_cues?: TranscriptSegment['caption_cues']
  overlap_intervals_ms?: TranscriptSegment['overlap_intervals_ms']
  caption_projection_status?: string
  speaker_diarization_source?: string
  /** Same saved voice-match provenance as the live transcript. */
  matched_profile_id?: TranscriptSegment['matched_profile_id']
  matched_profile_name?: TranscriptSegment['matched_profile_name']
  speaker_match_similarity?: TranscriptSegment['speaker_match_similarity']
  text: string
  start_ms: number
  end_ms: number
  duration_s: number
  speaker_id?: number
  speaker_name?: string
  language?: string
  translation?: string
  revision?: number
  review_count?: number
  manually_corrected?: boolean
  speaker_profile_id?: string | null
  speaker_status?: string
  speaker_uncertain?: boolean
  speaker_evidence_status?: string
}

export interface HealthInfo {
  status: string
  sample_rate: number
  min_speaker_audio_s: number
  spk_match_threshold: number
  [key: string]: unknown
}
