<script setup lang="ts">
import IntelligencePanel from '@/components/IntelligencePanel.vue'
import ContextCorrectionPanel from '@/components/ContextCorrectionPanel.vue'
import OfflineSpeakerPanel from '@/components/OfflineSpeakerPanel.vue'
import SpeakerContributionPanel from '@/components/SpeakerContributionPanel.vue'
import Phase07Status from '@/components/Phase07Status.vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { downloadUrl, hotwordsApi, meetingsApi, speakersApi } from '@/api/client'
import { RealtimeClient } from '@/api/ws'
import { startMicRecording } from '@/audio/recorder'
import AppIcon from '@/components/AppIcon.vue'
import { speakerColor } from '@/speaker/clustering'
import { splitDisplaySentences } from '@/transcript/display'
import { expandContentFirstSegments } from '@/transcript/contentFirst'
import { buildTranscriptParagraphs } from '@/transcript/paragraphs'
import { projectChannelSpeakers } from '@/transcript/channelSpeakers'
import {
  supportedLanguages, restoredSpeechChoices,
} from '@/config/phase1Languages'
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
const language = ref('auto')
const translationTarget = ref('')
const recordAudio = ref(true)
const autoMatchSaved = ref(true)
const partialText = ref('')
const partialLanguage = ref('')
const audioAnalysis = ref('')
const recordingList = ref<Array<{ name: string, bytes: number, duration_s: number | null, modified_at: string }>>([])

type Status = 'idle' | 'connecting' | 'recording' | 'stopping' | 'disconnected'

const recording = ref(false)
const status = ref<Status>('idle')
const error = ref('')
const toast = ref('')
const vadThreshold = ref(4)
const recThreshold = ref(1.5)
const serverInfo = ref<ServerReady | null>(null)
const meetingTitle = ref('')
const bufferedMs = ref(0)
const segments = ref<Segment[]>([])
const mergeGapS = ref(2)
const transcriptMeta = ref({ paragraphs: 0, segments: 0 })
const transcriptEl = ref<HTMLElement | null>(null)
const saving = ref(false)
const speakers = ref<Speaker[]>([])
const hotwords = ref<HotwordLibrary[]>([])
const showParticipants = ref(false)
const showDiagnostics = ref(false)
const showAllRecordings = ref(false)
type MeetingTool = 'transcript' | 'review' | 'recordings' | 'settings'
type ReviewTool = 'overview' | 'correction' | 'speakers'
const activeMeetingTool = ref<MeetingTool>('transcript')
const activeReviewTool = ref<ReviewTool>('overview')
function selectMeetingTool(tool: MeetingTool): void {
  activeMeetingTool.value = tool
  if (tool === 'review') activeReviewTool.value = 'overview'
  if (tool === 'recordings') void loadRecordings()
}

const selectedParticipantIds = ref<string[]>([])
const showHotwords = ref(false)
const selectedHotwordIds = ref<string[]>([])
const spkThreshold = ref(0.3)
const applying = ref(false)
const micLevel = ref(0)
const micSilent = ref(false)
const capturedSeconds = ref(0)
const deliveredSeconds = ref(0)
const durableSeconds = ref<number | null>(null)
const wavRecordingEnabled = ref(false)
const settingsSaving = ref(false)
const translatingSaved = ref(false)
const translationWarning = ref('')
const appliedSettings = ref('')
const editingSpeakerId = ref<number | null>(null)
const identityReviewSegmentId = ref<number | null>(null)
const selectedSpeakerId = ref<number | null>(null)
const speakerDetailTab = ref<'insights' | 'statements' | 'identity'>('insights')
const highlightedSegmentId = ref<number | null>(null)
const showIndependentSpeakerAnalysis = ref(false)
const reviewSpeakers = computed(() => {
  const byId = new Map<number, { id: number; name: string; segments: number; firstSegmentId: number }>()
  const ordered = [...segments.value].sort((a, b) => a.start_ms - b.start_ms || a.id - b.id)
  for (const segment of ordered) {
    if (!segment.text.trim()) continue
    const name = segment.speaker_name || (segment.speaker_id > 0
      ? `Speaker ${segment.speaker_id}` : 'Unidentified Speaker')
    const previous = byId.get(segment.speaker_id)
    if (previous) previous.segments += 1
    else byId.set(segment.speaker_id, {
      id: segment.speaker_id,
      name,
      segments: 1,
      firstSegmentId: segment.id,
    })
  }
  return [...byId.values()]
})
function inspectSpeaker(segmentId: number): void {
  const segment = segments.value.find(item => item.id === segmentId)
  if (!segment) return
  identityReviewSegmentId.value = segmentId
  selectedSpeakerId.value = segment.speaker_id
  speakerDetailTab.value = 'insights'
  activeReviewTool.value = 'speakers'
}
function jumpToSpeakerStatement(segmentId: number): void {
  const paragraph = paragraphs.value.find(p => p.segmentIds.includes(segmentId))
  if (!paragraph) return
  highlightedSegmentId.value = segmentId
  const target = [...(transcriptEl.value?.querySelectorAll<HTMLElement>('[data-paragraph-id]') ?? [])]
    .find(element => element.dataset.paragraphId === String(paragraph.id))
  target?.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

const speakerChoice = ref('')
const speakerSaving = ref(false)
const selectedWithoutVoiceprints = computed(() => selectedParticipantIds.value
  .map((id) => speakers.value.find((s) => s.id === id))
  .filter((s): s is Speaker => Boolean(s && !s.vector_count)))
const settingsPending = computed(() => appliedSettings.value !== settingsSignature())

let client: RealtimeClient | null = null
let recorder: Awaited<ReturnType<typeof startMicRecording>> | null = null
let clientSeq = 0
let stickToBottom = true
let stopRequested = false
let realtimeError = false
let audioStopAcknowledged = false
let lastPartialRevision = 0
let capturedSamples = 0
let capturePaintSample = 0
let startingSegmentCount = 0
let audioDeliveryWatchdog: ReturnType<typeof setTimeout> | null = null

function clearAudioDeliveryWatchdog(): void {
  if (audioDeliveryWatchdog !== null) {
    window.clearTimeout(audioDeliveryWatchdog)
    audioDeliveryWatchdog = null
  }
}

const statusText = computed(() => {
  switch (status.value) {
    case 'connecting':
      return 'Connecting...'
    case 'recording':
      return 'Recording'
    case 'stopping':
      return 'Finishing...'
    case 'disconnected':
      return 'Disconnected'
    default:
      return 'Not started'
  }
})

const statusClass = computed(() => {
  if (recording.value || status.value === 'stopping') return 'ok'
  return status.value === 'disconnected' ? 'bad' : 'muted'
})

/** Use acoustic report channels for headings when the meeting has them. */
const channelSegments = computed(() => projectChannelSpeakers(segments.value))
const paragraphs = computed<Paragraph[]>(() =>
  buildTranscriptParagraphs(channelSegments.value ?? expandContentFirstSegments(
    segments.value, translationTarget.value,
  ), { mergeGapS: mergeGapS.value, translationTarget: translationTarget.value,
    channelHeadings: channelSegments.value !== null,
    singlePrimarySpeaker: channelSegments.value === null }),
)
const overlapWarnings = computed(() => segments.value.reduce(
  (sum, seg) => sum + (seg.overlap_intervals_ms?.length ?? 0), 0,
))
const untranslatedCount = computed(() => segments.value.filter((seg) =>
  translationTarget.value && seg.language !== translationTarget.value &&
  (seg.translation_target !== translationTarget.value || !seg.translation?.trim()),
).length)
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

function settingsPayload() {
  return {
    language: language.value, translate_to: translationTarget.value,
    record_audio: recordAudio.value, vad_threshold: vadThreshold.value,
    rec_threshold: recThreshold.value,
    participant_ids: [...selectedParticipantIds.value].sort(),
    auto_match_saved: autoMatchSaved.value,
    hotword_ids: [...selectedHotwordIds.value].sort(),
    spk_threshold: spkThreshold.value,
  }
}

function settingsSignature(): string {
  return JSON.stringify(settingsPayload())
}

async function applySpeechSettings(quiet = false): Promise<boolean> {
  if (recording.value || status.value === 'stopping' || status.value === 'connecting' || settingsSaving.value) return false
  settingsSaving.value = true
  error.value = ''
  try {
    await meetingsApi.settings(props.id, settingsPayload())
    appliedSettings.value = settingsSignature()
    if (!quiet) showToast('Speech settings saved for the next recording')
    return true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    return false
  } finally {
    settingsSaving.value = false
  }
}

function wsUrl(): string {
  const params = new URLSearchParams({
    vad_threshold: String(vadThreshold.value),
    rec_threshold: String(recThreshold.value),
    meeting_id: props.id,
    language: language.value,
    translate_to: translationTarget.value,
    record_audio: String(recordAudio.value),
    participants: selectedParticipantIds.value.join(','),
    auto_match_saved: String(autoMatchSaved.value),
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

/** Replace the entire segment list when the backend sends a snapshot. */
function applyTranscript(msg: ServerTranscript): void {
  const segs: Segment[] = msg.segments.map((s) => ({
    ...s,
    color: speakerColor(s.speaker_id),
  }))
  segments.value = segs
  transcriptMeta.value = {
    segments: msg.segments.length,
    paragraphs: paragraphs.value.length,
  }
  void scrollIfNeeded()
}

async function saveMeeting(quiet = false): Promise<boolean> {
  saving.value = true
  try {
    await meetingsApi.save(props.id)
    if (!quiet) showToast('Saved to disk')
    return true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    return false
  } finally {
    saving.value = false
  }
}

/** Load saved participant selections and speaker records before connecting. */
async function loadMeetingConfig(): Promise<void> {
  try {
    const { meeting } = await meetingsApi.get(props.id)
    selectedParticipantIds.value = meeting.ui_settings?.participant_ids ?? meeting.participant_ids ?? []
    autoMatchSaved.value = meeting.ui_settings?.auto_match_saved ?? meeting.auto_match_saved ?? true
    selectedHotwordIds.value = meeting.ui_settings?.hotword_ids ?? meeting.hotword_ids ?? []
    spkThreshold.value = meeting.ui_settings?.spk_threshold ?? meeting.spk_threshold ?? 0.3
    const restored = restoredSpeechChoices(
      meeting.ui_settings?.language ?? 'auto',
      meeting.ui_settings?.translate_to ?? '',
    )
    language.value = restored.language
    translationTarget.value = restored.translate_to
    if (restored.hadUnsupportedSavedChoice) {
      // Never rewrite historic saved content or claim support for old
      // language selections. Recording starts with explicitly supported
      // settings instead.
      showToast('Previous language selection is outside the Phase 1 ten-language list. Auto detection and no translation selected.')
    }
    recordAudio.value = meeting.ui_settings?.record_audio ?? true
    vadThreshold.value = meeting.ui_settings?.vad_threshold ?? 4
    recThreshold.value = meeting.ui_settings?.rec_threshold ?? 1.5
  } catch {
    // Use default threshold 0.3 and empty participant/hotword selections.
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
  appliedSettings.value = settingsSignature()
}

async function loadSavedTranscript(): Promise<void> {
  try {
    const { segments: saved } = await meetingsApi.segments(props.id)
    applyTranscript({
      type: 'transcript', version: 0,
      segments: saved.map((s) => ({
        id: s.id, text: s.text, start_ms: s.start_ms, end_ms: s.end_ms,
        speaker_id: s.speaker_id ?? -1, speaker_name: s.speaker_name ?? 'Unidentified Speaker',
        language: s.language ?? 'und', translation: s.translation ?? '',
        translation_target: s.translation_target,
        revision: s.revision ?? 0,
        caption_cues: s.caption_cues,
        overlap_intervals_ms: s.overlap_intervals_ms,
        caption_projection_status: s.caption_projection_status,
        speaker_diarization_source: s.speaker_diarization_source,
        speaker_acoustic_channel: s.speaker_acoustic_channel,
        speaker_channel_gate_reason: s.speaker_channel_gate_reason,
        speaker_acoustic_stream_start: s.speaker_acoustic_stream_start,
        speaker_continuity_id: s.speaker_continuity_id,
        speaker_uncertain: s.speaker_uncertain,
        speaker_final_resolution_status: s.speaker_final_resolution_status,
        matched_profile_id: s.matched_profile_id,
        matched_profile_name: s.matched_profile_name,
        speaker_match_similarity: s.speaker_match_similarity,
      })),
    })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

async function translateSavedTranscript(): Promise<void> {
  const target = translationTarget.value
  if (!target || translatingSaved.value || recording.value || status.value === 'stopping') return
  translatingSaved.value = true
  translationWarning.value = ''
  try {
    if (settingsPending.value && !(await applySpeechSettings(true))) return
    let after = -1
    let translated = 0
    while (target === translationTarget.value && !recording.value) {
      const batch = await meetingsApi.translate(props.id, target, after)
      translated += batch.translated
      if (!batch.has_more || batch.next_after <= after) break
      after = batch.next_after
    }
    showToast(`${translated} caption(s) translated`)
  } catch (err) {
    translationWarning.value = err instanceof Error ? err.message : String(err)
  } finally {
    if (!recording.value) await loadSavedTranscript()
    translatingSaved.value = false
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

/** Save participants and reconnect to update speaker labels. */
async function applyParticipants(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showParticipants.value = false
    if (!(await applySpeechSettings(true))) return
    showToast('Participant selection saved')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    applying.value = false
  }
}

/** Save hotword selections and reconnect with new recognition terms. */
async function applyHotwords(): Promise<void> {
  if (recording.value || applying.value) return
  applying.value = true
  error.value = ''
  try {
    showHotwords.value = false
    if (!(await applySpeechSettings(true))) return
    showToast('Hotword selection saved')
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
  showToast(format === 'md' ? 'Markdown exported' : 'JSON exported')
}

async function copyTranscript(): Promise<void> {
  const text = paragraphs.value
    .map((item) => `[${item.speaker_name || `Speaker ${item.speaker_id}`}] ${item.text}`)
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
  showToast('Transcript copied')
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
  capturedSamples = 0
  capturePaintSample = 0
  capturedSeconds.value = 0
  deliveredSeconds.value = 0
  durableSeconds.value = null
  let silentCount = 0
  try {
    recorder = await startMicRecording(
      (pcm) => {
        if (!client?.connected || !client.sendPcm(pcm)) {
          realtimeError = true
          error.value = 'Microphone audio could not be delivered. Recording has stopped.'
          if (recording.value) void stop()
          return
        }
        capturedSamples += pcm.length
        if (capturedSamples - capturePaintSample >= 8000) {
          capturePaintSample = capturedSamples
          capturedSeconds.value = capturedSamples / 16000
        }
      },
      16000,
      (level) => {
        micLevel.value = level
        // Warn after about 1.5 seconds of continuous microphone silence.
        if (level < 0.0003) {
          silentCount++
          if (silentCount > 550) micSilent.value = true
        } else {
          silentCount = 0
          micSilent.value = false
        }
      },
      () => {
        realtimeError = true
        error.value = 'Microphone device disconnected or permission was revoked.'
        if (recording.value) void stop()
      },
    )
    startingSegmentCount = segments.value.length
    recording.value = true
    status.value = 'recording'
    clearAudioDeliveryWatchdog()
    audioDeliveryWatchdog = window.setTimeout(() => {
      if (recording.value && capturedSamples >= 16000 && deliveredSeconds.value === 0) {
        realtimeError = true
        error.value = 'Microphone audio is not confirmed by the server. Check the connection before recording further.'
      }
    }, 7000)
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
  // Reconnect to resume the meeting after the previous session ends.
  error.value = ''
  connect(true)
}

async function stop(): Promise<void> {
  // Do not request final ASR until the browser worklet has drained.
  if (!recording.value) return
  recording.value = false
  stopRequested = true
  audioStopAcknowledged = false
  status.value = 'stopping'
  clearAudioDeliveryWatchdog()
  const activeRecorder = recorder
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  try {
    await activeRecorder?.stop()
    if (client?.connected) client.sendStop()
    else throw new Error('Audio connection closed before recording finalization')
  } catch (err) {
    realtimeError = true
    error.value = err instanceof Error ? err.message : String(err)
    client?.close()
    status.value = 'disconnected'
  }
}

function stopInternal(): void {
  clearAudioDeliveryWatchdog()
  recording.value = false
  void recorder?.stop().catch(() => {})
  recorder = null
  micLevel.value = 0
  micSilent.value = false
  client?.close()
  client = null
  clientSeq += 1
}

function connect(autostart = false): void {
  // Dispose the previous socket before attempting a new connection.
  const previous = client
  client = null
  void recorder?.stop().catch(() => {})
  recorder = null
  previous?.close()
  error.value = ''
  bufferedMs.value = 0
  partialText.value = ''
  lastPartialRevision = 0
  serverInfo.value = null
  capturedSamples = 0
  capturedSeconds.value = 0
  deliveredSeconds.value = 0
  durableSeconds.value = null
  wavRecordingEnabled.value = false
  clearAudioDeliveryWatchdog()
  stickToBottom = true
  stopRequested = false
  realtimeError = false
  audioStopAcknowledged = false

  const mySeq = ++clientSeq
  client = new RealtimeClient({
    url: wsUrl(),
    onReady: (msg) => {
      if (clientSeq !== mySeq) return
      serverInfo.value = msg
      appliedSettings.value = settingsSignature()
      wavRecordingEnabled.value = msg.wav_recording_enabled === true
      if (recordAudio.value && !wavRecordingEnabled.value) {
        error.value = 'WAV recording was requested but the server could not enable it.'
      }
      meetingTitle.value = msg.title ?? 'Untitled Meeting'
      mergeGapS.value = msg.merge_gap_s || 2
      if (msg.participant_ids) selectedParticipantIds.value = msg.participant_ids
      if (msg.hotword_ids) selectedHotwordIds.value = msg.hotword_ids
      if (msg.spk_threshold !== undefined) spkThreshold.value = msg.spk_threshold
      status.value = 'idle'
      // Start automatically on create, resume, or restart.
      if (autostart) void startMic()
    },
    onPartial: (text, detectedLanguage, revision) => {
      if (clientSeq !== mySeq || revision <= lastPartialRevision) return
      lastPartialRevision = revision
      partialText.value = text
      partialLanguage.value = detectedLanguage
      if (text) void scrollIfNeeded()
    },
    onAudioAnalysis: (result) => {
      if (clientSeq !== mySeq) return
      audioAnalysis.value = [result.emotion, ...(result.events || []),
        (result.overlap?.length || 0) > 0 ? 'Overlapping speech detected' : '']
        .filter(Boolean).join(', ')
    },
    onFeatureWarning: (message, feature) => {
      if (clientSeq !== mySeq) return
      if (feature === 'translation') translationWarning.value = message
      else error.value = message
    },
    onTranscript: (msg) => {
      if (clientSeq !== mySeq) return
      // Speaker/translation updates are snapshots, not draft-finalization
      // signals. Only an explicit empty partial should remove live text.
      applyTranscript(msg)
    },
    onState: (ms) => {
      if (clientSeq !== mySeq) return
      bufferedMs.value = ms
    },
    onAudioProgress: (samples, durableSamples) => {
      if (clientSeq !== mySeq) return
      deliveredSeconds.value = samples / 16000
      if (durableSamples !== null && durableSamples !== undefined) {
        durableSeconds.value = durableSamples / 16000
      }
      if (samples > 0) clearAudioDeliveryWatchdog()
    },
    onAudioReceived: (samples, matched, durableSamples) => {
      if (clientSeq !== mySeq) return
      audioStopAcknowledged = matched
      deliveredSeconds.value = samples / 16000
      if (durableSamples !== null && durableSamples !== undefined) {
        durableSeconds.value = durableSamples / 16000
        if (durableSamples !== samples) {
          realtimeError = true
          error.value = 'WAV durable sample count differs from the received audio.'
        }
      }
      if (!matched) {
        realtimeError = true
        error.value = 'Audio sample mismatch: the server did not receive every sent sample.'
      }
    },
    onError: (message) => {
      if (clientSeq !== mySeq) return
      realtimeError = true
      error.value = message
    },
    onClosed: (reason, saved) => {
      if (clientSeq !== mySeq) return
      client = null
      recording.value = false
      partialText.value = ''
      partialLanguage.value = ''
      clearAudioDeliveryWatchdog()
      void recorder?.stop().catch(() => {})
      recorder = null
      if (reason === 'closed') {
        status.value = 'idle'
        if (saved === false) {
          error.value = 'Final meeting save failed. Please check the server log.'
        } else if (stopRequested && !audioStopAcknowledged) {
          realtimeError = true
          error.value = 'Audio stop acknowledgement missing. Verify the recording before relying on it.'
        } else if (stopRequested && !realtimeError) {
          if (deliveredSeconds.value >= 1 && segments.value.length <= startingSegmentCount) {
            error.value = 'Audio reached the server, but no speech was recognized. Check the microphone level and the saved WAV.'
          } else {
            showToast('Stopped and automatically saved')
          }
        }
        if (stopRequested && recordAudio.value) void loadRecordings()
        if (stopRequested && saved === true && audioStopAcknowledged &&
            !realtimeError && untranslatedCount.value > 0) void translateSavedTranscript()
      } else {
        status.value = 'disconnected'
        if (!realtimeError) error.value = 'Connection lost before final confirmation.'
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

function formatTime(ms: number, precise = false): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(s).padStart(2, '0')
  const time = h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
  return precise ? time + '.' + String(Math.max(0, Math.floor(ms)) % 1000).padStart(3, '0') : time
}

function formatCaptionTime(start: number, end: number): string {
  const precise = end > start && Math.floor(start / 1000) === Math.floor(end / 1000)
  return formatTime(start, precise) + ' – ' + formatTime(end, precise)
}

async function loadRecordings(): Promise<void> {
  try {
    const response = await fetch(`/api/speech/meetings/${encodeURIComponent(props.id)}/recordings`)
    if (!response.ok) throw new Error('Unable to load recordings')
    const payload = await response.json() as { recordings: Array<{ name: string, bytes: number, duration_s: number | null, modified_at: string }> }
    recordingList.value = payload.recordings
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

function recordingUrl(name: string): string {
  return `/api/speech/meetings/${encodeURIComponent(props.id)}/recordings/${encodeURIComponent(name)}`
}

function openIdentityEvidence(id: number): void {
  if (recording.value || status.value === 'stopping') return
  const segment = segments.value.find(item => item.id === id)
  if (!segment) return
  identityReviewSegmentId.value = id
  selectedSpeakerId.value = segment.speaker_id
  selectMeetingTool('review')
  activeReviewTool.value = 'speakers'
  speakerDetailTab.value = 'identity'
}

function editSpeakerById(id: number): void {
  if (recording.value || status.value === 'stopping') return
  editingSpeakerId.value = id
  speakerChoice.value = ''
}

async function saveSpeakerLabel(id: number): Promise<void> {
  const segment = segments.value.find((item) => item.id === id)
  if (!segment || speakerSaving.value) return
  speakerSaving.value = true
  error.value = ''
  try {
    const response = await fetch(`/api/speech/meetings/${encodeURIComponent(props.id)}/segments/${id}/speaker`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ participant_id: speakerChoice.value || null, revision: segment.revision ?? 0 }),
    })
    if (!response.ok) {
      const detail = await response.json() as { detail?: string }
      throw new Error(detail.detail || 'Speaker assignment failed')
    }
    const result = await response.json() as Pick<Segment,
      'speaker_id' | 'speaker_name' | 'speaker_final_resolution_status' | 'revision'>
    segments.value = segments.value.map((item) => item.id === id
      ? { ...item, ...result, color: speakerColor(result.speaker_id) }
      : item)
    editingSpeakerId.value = null
    showToast('Speaker assignment saved')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    speakerSaving.value = false
  }
}

function editSegmentById(id: number): void {
  const segment = segments.value.find((item) => item.id === id)
  if (segment) void correctTranscript(segment)
}

async function correctTranscript(seg: Segment): Promise<void> {
  if (recording.value || status.value === 'stopping') return
  const text = window.prompt('Correct transcript segment', seg.text)
  if (text === null || !text.trim() || text === seg.text) return
  try {
    const response = await fetch(`/api/speech/meetings/${encodeURIComponent(props.id)}/segments/${seg.id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, revision: seg.revision ?? 0 }),
    })
    if (!response.ok) throw new Error(await response.text())
    const result = await response.json() as { text: string, revision: number }
    segments.value = segments.value.map((item) => item.id === seg.id
      ? { ...item, text: result.text, revision: result.revision, translation: '', translation_target: '' }
      : item)
    showToast('Correction saved')
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(async () => {
  // Consume the one-time start request before asynchronous setup. Browser
  // refresh must not start another microphone capture.
  const startOnce = route.query.autostart === '1'
  if (startOnce) {
    const { autostart: _consumed, ...remainingQuery } = route.query
    await router.replace({ query: remainingQuery, hash: route.hash })
  }
  await loadMeetingConfig()
  await loadSavedTranscript()
  await loadRecordings()
  if (startOnce) connect(true)
})

onBeforeUnmount(() => stopInternal())
</script>

<template>
  <div class="meeting-page">
    <header class="compact-meeting-header">
      <div class="compact-meeting-heading">
        <button class="meeting-back" aria-label="Back to home" title="Back to home" @click="router.push('/')">
          <AppIcon name="arrow-left" :size="18" />
        </button>
        <div class="compact-meeting-titles">
          <h1>{{ meetingTitle || 'Untitled Meeting' }}</h1>
          <div class="meeting-running-status">
            <span class="status-chip" :class="statusClass"><span class="status-dot" aria-hidden="true"></span>{{ statusText }}</span>
            <span v-if="recording" class="compact-mic-meter" :style="{ '--mic-fill': Math.min(100, micLevel * 300) + '%' }" aria-label="Microphone input level"></span>
            <span v-if="recording">Captured {{ capturedSeconds.toFixed(1) }}s · Received {{ deliveredSeconds.toFixed(1) }}s</span>
            <span v-else>{{ paragraphs.length }} turns</span>
          </div>
          <div class="meeting-header-context" aria-label="Meeting details">
            <span><strong>Language</strong> {{ language === 'auto' ? 'Auto' : language.toUpperCase() }}</span>
            <span class="identity-mode-label"><strong>Voice identities</strong> {{ selectedParticipantIds.length ? selectedParticipantIds.length + ' selected' : !autoMatchSaved ? 'Off' : serverInfo && (!serverInfo.auto_match_active || serverInfo.auto_match_profile_count === 0) ? 'No eligible voices' : 'Auto library match' }}</span>
            <span><strong>Translation</strong> {{ translationTarget || 'Off' }}</span>
            <span v-if="durableSeconds !== null"><strong>Audio</strong> {{ durableSeconds.toFixed(1) }}s saved</span>
          </div>
        </div>
      </div>
      <div class="compact-meeting-toolbar" aria-label="Meeting actions">
        <details class="workspace-export-menu">
        <summary><AppIcon name="download" :size="15" /> Export <span aria-hidden="true">⌄</span></summary>
        <div class="workspace-export-options">
          <button type="button" @click="exportMeeting('md')">Download transcript (.md)</button>
          <button type="button" @click="exportMeeting('json')">Download structured data (.json)</button>
          <button type="button" :disabled="paragraphs.length === 0" @click="copyTranscript">Copy transcript</button>
        </div>
      </details>
        <div class="toolbar-secondary-actions">
          <button class="btn btn-ghost toolbar-save" :disabled="saving" @click="saveMeeting()">
            <AppIcon name="save" :size="16" /> {{ saving ? 'Saving...' : 'Save' }}
          </button>
        </div>
        <button class="btn btn-primary compact-record-action" :class="{ danger: recording }" :disabled="status === 'connecting' || status === 'stopping'" @click="recording ? stop() : start()">
          <AppIcon :name="recording ? 'stop' : 'mic'" :size="17" />
          {{ recording ? 'Stop recording' : 'Start recording' }}
        </button>
      </div>
    </header>
    <p v-if="settingsPending && !recording" class="hint-text warn-text" role="status">Settings changed. Apply to save them, or start recording to use these values.</p>
    <p v-if="selectedWithoutVoiceprints.length" class="hint-text warn-text" role="status">
      {{ selectedWithoutVoiceprints.length }} selected participant(s) have no usable voiceprints. Enroll clean voice samples before relying on identification.
      <RouterLink to="/speakers">Manage voiceprints</RouterLink>
    </p>
    <div v-if="untranslatedCount > 0" class="hint-text" role="status">
      <span>{{ untranslatedCount }} caption(s) awaiting translation. Original text is shown.</span>
      <p v-if="translationWarning" class="warn-text">{{ translationWarning }}</p>
      <button v-if="!recording && status !== 'connecting' && status !== 'stopping'"
        type="button" class="btn btn-ghost btn-sm" :disabled="translatingSaved || settingsSaving"
        @click="translateSavedTranscript">
        {{ translatingSaved ? 'Translating...' : 'Translate saved transcript' }}
      </button>
    </div>
    <div class="meeting-workspace" :class="{ 'tool-panel-open': activeMeetingTool !== 'transcript' }">
      <nav class="meeting-tool-sidebar" aria-label="Meeting tools">
        <div class="meeting-tool-nav-title">MEETING</div>
        <button type="button" class="meeting-tool-link" :class="{ active: activeMeetingTool === 'transcript' }" :aria-current="activeMeetingTool === 'transcript' ? 'page' : undefined" @click="selectMeetingTool('transcript')">
          <AppIcon name="file-text" :size="18" /><span>Transcript</span>
        </button>
        <div class="meeting-tool-nav-title meeting-tool-section-title">TOOLS</div>
        <button type="button" class="meeting-tool-link" :class="{ active: activeMeetingTool === 'review' }" :aria-pressed="activeMeetingTool === 'review'" @click="selectMeetingTool('review')">
          <AppIcon name="edit" :size="18" /><span>Review</span>
        </button>
        <button type="button" class="meeting-tool-link" :class="{ active: activeMeetingTool === 'recordings' }" :aria-pressed="activeMeetingTool === 'recordings'" @click="selectMeetingTool('recordings')">
          <AppIcon name="list" :size="18" /><span>Recordings</span>
        </button>
        <button type="button" class="meeting-tool-link" :class="{ active: activeMeetingTool === 'settings' }" :aria-pressed="activeMeetingTool === 'settings'" @click="selectMeetingTool('settings')">
          <AppIcon name="settings" :size="18" /><span>Settings</span>
          <span v-if="settingsPending" class="meeting-tool-unsaved" title="Unsaved changes"></span>
        </button>
        <div class="meeting-tool-nav-bottom">
          <span class="meeting-tool-status-dot" :class="{ recording }"></span>
          <span>{{ recording ? 'Recording now' : 'Local session' }}</span>
        </div>
      </nav>
      <div class="meeting-primary">
    <main ref="transcriptEl" class="transcript" @scroll="onTranscriptScroll">
      <div v-if="segments.length === 0 && !partialText" class="empty-state">
        <div class="empty-icon"><AppIcon name="mic" :size="26" /></div>
        <h3>{{ recording ? 'Listening...' : 'No transcript yet' }}</h3>
        <p v-if="!recording">Start recording and speak for several seconds. Short recordings are finalized when stopped.</p>
        <p v-else>Speak for several seconds. Confirmed text may appear after about seven seconds, or when recording stops.</p>
        <p v-if="recording && micSilent" class="hint-text warn-text" style="margin-top:8px">
          No microphone input detected. Check browser permission and system microphone settings.
        </p>
      </div>

      <template v-else>
        <p v-if="overlapWarnings > 0" class="transcript-overlap-warning">
          Overlapping voices detected in {{ overlapWarnings }} acoustic intervals.
          The system may not have transcribed every simultaneous speaker.
          Untranscribed words are never fabricated.
        </p>
        <article v-for="para in paragraphs" :key="para.id" :data-paragraph-id="para.id"
          class="paragraph" :class="{ 'paragraph-unidentified': para.speaker_id <= 0,
            'paragraph-highlighted': highlightedSegmentId !== null && para.segmentIds.includes(highlightedSegmentId) }">
          <div class="para-head">
            <span class="speaker-dot" :style="{ background: para.color }"></span>
            <span class="speaker-name">{{ para.speaker_name || `Speaker ${para.speaker_id}` }}</span>
            <span v-if="para.overlapping" class="speaker-registered-label">{{ channelSegments ? 'Overlap' : 'Primary during overlap' }}</span>
            <span v-if="segments.some(seg => para.segmentIds.includes(seg.id) && seg.speaker_name === para.speaker_name && seg.speaker_diarization_source === 'registered_voiceprint')"
              class="speaker-registered-label" title="Matched to a saved voiceprint. Verify important identities before relying on them.">
              <AppIcon name="check" :size="12" /> Voice match
            </span>
            <!-- Caption chunk count stays internal; a badge next to Speaker 1
                 looked like a phantom Speaker 12 or Speaker 13. -->
            <span class="para-time">{{ formatCaptionTime(para.start_ms, para.end_ms) }}</span>
          </div>
          <div class="transcript-utterance-lines">
            <p v-for="(sentence, sentenceIndex) in splitDisplaySentences(para.text)"
              :key="sentenceIndex" class="para-text transcript-sentence">{{ sentence }}</p>
          </div>
          <p v-if="para.originalText" class="para-text hint-text">
            <strong>Original</strong> {{ para.originalText }}
          </p>
          <details class="transcript-edit-details" v-if="!recording">
            <summary>Edit transcript or speaker</summary>
            <div class="transcript-edit-actions">
            <button v-for="segmentId in para.segmentIds" :key="segmentId"
              class="btn btn-ghost btn-sm" :disabled="status === 'stopping'"
              @click="editSegmentById(segmentId)">Edit #{{ segmentId }}</button>
            <button v-for="segmentId in para.segmentIds" :key="`speaker-${segmentId}`"
              class="btn btn-ghost btn-sm" :disabled="status === 'stopping'"
              @click="editSpeakerById(segmentId)">Edit speaker for segment #{{ segmentId }}</button>
            <button v-for="segmentId in para.segmentIds" :key="`identity-${segmentId}`"
              class="btn btn-ghost btn-sm" :disabled="status === 'stopping'"
              @click="openIdentityEvidence(segmentId)">Evidence #{{ segmentId }}</button>
            <div v-if="editingSpeakerId !== null && para.segmentIds.includes(editingSpeakerId)" class="speaker-edit-controls">
              <select v-model="speakerChoice" aria-label="Correct speaker">
                <option value="">Unidentified speaker</option>
                <option v-for="spk in speakers" :key="spk.id" :value="spk.id">{{ spk.name }}</option>
              </select>
              <button class="btn btn-primary btn-sm" :disabled="speakerSaving" @click="saveSpeakerLabel(editingSpeakerId!)">Save speaker</button>
              <button class="btn btn-ghost btn-sm" @click="editingSpeakerId = null">Cancel</button>
            </div>
            </div>
          </details>
        </article>
      </template>
      <article v-if="partialText" class="paragraph transcript-live-preview"
        aria-live="polite" aria-label="Provisional live caption">
        <div class="para-head">
          <span class="speaker-dot transcript-preview-dot" aria-hidden="true"></span>
          <span class="speaker-name">Live caption · not confirmed</span>
          <span v-if="partialLanguage && partialLanguage !== 'und'"
            class="seg-count">{{ partialLanguage.toUpperCase() }}</span>
        </div>
        <div class="transcript-utterance-lines">
          <p v-for="(sentence, index) in splitDisplaySentences(partialText)"
            :key="index" class="para-text transcript-sentence">{{ sentence }}</p>
        </div>
      </article>
    </main>
      </div>

      <aside v-if="activeMeetingTool !== 'transcript'" class="meeting-detail-panel" :aria-label="activeMeetingTool === 'review' ? 'Review tools' : activeMeetingTool === 'settings' ? 'Meeting settings' : 'Recording library'">
        <header class="meeting-detail-header">
          <div>
            <span class="transcript-eyebrow">{{ activeMeetingTool === 'review' ? 'REVIEW WORKSPACE' : activeMeetingTool === 'settings' ? 'PREFERENCES' : 'AUDIO LIBRARY' }}</span>
            <h2>{{ activeMeetingTool === 'review' ? 'Review' : activeMeetingTool === 'settings' ? 'Meeting settings' : 'Recordings' }}</h2>
          </div>
          <button type="button" class="meeting-back" aria-label="Close tool panel" @click="selectMeetingTool('transcript')"><AppIcon name="x" :size="17" /></button>
        </header>
        <div class="meeting-detail-scroll">
          <template v-if="activeMeetingTool === 'review'">
            <nav class="meeting-review-tools" aria-label="Review sections">
              <button type="button" :class="{ active: activeReviewTool === 'overview' }"
                :aria-current="activeReviewTool === 'overview' ? 'page' : undefined"
                @click="activeReviewTool = 'overview'">Overview</button>
              <button type="button" :class="{ active: activeReviewTool === 'correction' }"
                :aria-current="activeReviewTool === 'correction' ? 'page' : undefined"
                @click="activeReviewTool = 'correction'">Corrections</button>
              <button type="button" :class="{ active: activeReviewTool === 'speakers' }"
                :aria-current="activeReviewTool === 'speakers' ? 'page' : undefined"
                @click="activeReviewTool = 'speakers'">Speakers</button>
            </nav>
            <div class="meeting-selected-tool">
              <IntelligencePanel v-if="activeReviewTool === 'overview'" compact
                :meeting-id="props.id" :segments="segments" />
              <div v-else-if="activeReviewTool === 'correction'" class="review-focus-panel">
                <div class="review-focus-heading">
                  <h3>Transcript corrections</h3>
                  <p>Review AI suggestions before applying them. To edit a specific turn, use its Edit control in the transcript.</p>
                </div>
                <ContextCorrectionPanel :meeting-id="props.id" :segments="segments"
                  :recording="recording || status === 'connecting' || status === 'stopping'"
                  @applied="loadSavedTranscript" />
              </div>
              <div v-else class="review-focus-panel">
                <div class="review-focus-heading">
                  <h3>Speakers</h3>
                  <p>Review fused speaker identities and voice matches. Transcript headings follow acoustic channels when available.</p>
                </div>
                <div v-if="!reviewSpeakers.length" class="review-tool-empty">
                  Speaker details appear after the meeting has a transcript.
                </div>
                <div v-else class="review-speaker-list" aria-label="Speakers in this meeting">
                  <button v-for="speaker in reviewSpeakers" :key="speaker.id" type="button"
                    :class="{ active: selectedSpeakerId === speaker.id }"
                    @click="inspectSpeaker(speaker.firstSegmentId)">
                    <span class="review-speaker-avatar"><AppIcon name="users" :size="16" /></span>
                    <span class="review-speaker-details">
                      <strong>{{ speaker.name }}</strong>
                      <small>{{ speaker.segments }} transcript {{ speaker.segments === 1 ? 'segment' : 'segments' }}</small>
                    </span>
                    <AppIcon name="chevron-right" :size="16" />
                  </button>
                </div>
                <SpeakerContributionPanel v-if="selectedSpeakerId !== null"
                  :key="`${props.id}-${selectedSpeakerId}`"
                  :meeting-id="props.id"
                  :speaker-id="selectedSpeakerId"
                  :speaker-name="reviewSpeakers.find(speaker => speaker.id === selectedSpeakerId)?.name ?? 'Speaker'"
                  :segments="segments"
                  :recording="recording || status === 'connecting' || status === 'stopping'"
                  :selected-segment-id="identityReviewSegmentId" :tab="speakerDetailTab"
                  @update:tab="speakerDetailTab = $event"
                  @reviewed="loadSavedTranscript" @jump-to="jumpToSpeakerStatement" />
                <p v-else-if="reviewSpeakers.length" class="review-tool-hint">
                  Choose a speaker above to see their independent analysis, original statements and identity evidence.
                </p>
                <div class="review-advanced-section">
                  <button type="button" class="review-advanced-toggle"
                    :aria-expanded="showIndependentSpeakerAnalysis"
                    @click="showIndependentSpeakerAnalysis = !showIndependentSpeakerAnalysis">
                    <span>Independent speaker analysis</span>
                    <AppIcon :name="showIndependentSpeakerAnalysis ? 'chevron-up' : 'chevron-down'" :size="16" />
                  </button>
                  <OfflineSpeakerPanel v-if="showIndependentSpeakerAnalysis" :meeting-id="props.id"
                    :recording="recording || status === 'connecting' || status === 'stopping'" />
                </div>
              </div>
            </div>
          </template>
          <template v-else-if="activeMeetingTool === 'settings'">
            <section class="meeting-config-panel" aria-label="Meeting configuration">
                  <div class="config-panel-heading">
                    <div>
                      <span class="transcript-eyebrow">PREFERENCES</span>
                      <h2>Meeting settings</h2>
                      <p>Configure how speech is captured and presented. Changes take effect when applied.</p>
                    </div>
                    <button type="button" class="meeting-back" aria-label="Close settings" @click="selectMeetingTool('transcript')"><AppIcon name="x" :size="16" /></button>
                  </div>
                  <div class="meeting-config-grid">
                    <section class="config-group" aria-label="Speech settings">
                      <div class="config-group-title"><span class="config-step">01</span><div><h3>Language & output</h3><p>Choose how conversations are recognized.</p></div></div>
                      <label class="field">Spoken language
                        <select v-model="language" :disabled="recording || status === 'connecting' || status === 'stopping'">
                          <option v-for="entry in supportedLanguages" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
                        </select>
                      </label>
                      <label class="field">Display translation
                        <select v-model="translationTarget" :disabled="recording || status === 'connecting' || status === 'stopping'">
                          <option value="">No translation</option>
                          <option v-for="entry in supportedLanguages.filter((item) => item[0] !== 'auto')" :key="entry[0]" :value="entry[0]">{{ entry[1] }}</option>
                        </select>
                      </label>
                      <label class="config-checkbox"><input v-model="recordAudio" type="checkbox" :disabled="recording" /><span><strong>Save audio recording</strong><small>Keep a WAV copy for playback and review.</small></span></label>
                    </section>
                    <section class="config-group" aria-label="Recognition timing">
                      <div class="config-group-title"><span class="config-step">02</span><div><h3>Recognition timing</h3><p>Advanced controls for transcription latency.</p></div></div>
                      <label class="field" title="Trigger recognition when this audio duration is buffered.">Processing window (seconds)
                        <input v-model.number="vadThreshold" type="number" min="0.5" max="30" step="0.5" :disabled="recording" />
                      </label>
                      <label class="field" title="Confirm a sentence after this trailing audio delay.">Confirmation delay (seconds)
                        <input v-model.number="recThreshold" type="number" min="0.2" max="30" step="0.5" :disabled="recording" />
                      </label>
                      <p class="config-help">Default values work for most meetings. Smaller windows can increase processing load.</p>
                    </section>
                    <section class="config-group" aria-label="Recognition resources">
                      <div class="config-group-title"><span class="config-step">03</span><div><h3>People & vocabulary</h3><p>Improve attribution and specialized words.</p></div></div>
                      <label class="config-checkbox">
            <input v-model="autoMatchSaved" type="checkbox" :disabled="recording || status === 'connecting' || status === 'stopping'" />
            <span><strong>Recognize saved voices automatically</strong><small>When no participants are selected, compare speech against eligible voices in your local library. Uncertain matches stay anonymous.</small></span>
          </label>
          <button type="button" class="config-resource" @click="showParticipants = true">
                        <span class="config-resource-icon"><AppIcon name="users" :size="18" /></span>
                        <span><strong>Participants</strong><small>{{ selectedParticipantIds.length ? selectedParticipantIds.length + ' selected · Match registered voices' : autoMatchSaved ? 'Auto-match eligible voices from your saved library' : 'Automatic identity matching disabled' }}</small></span><span class="resource-chevron">›</span>
                      </button>
                      <button type="button" class="config-resource" @click="showHotwords = true">
                        <span class="config-resource-icon"><AppIcon name="flame" :size="18" /></span>
                        <span><strong>Hotwords</strong><small>{{ selectedHotwordIds.length }} libraries · Recognize domain terms</small></span><span class="resource-chevron">›</span>
                      </button>
                    </section>
                  </div>
                  <div class="config-panel-footer">
                    <div class="config-footer-meta"><span v-if="settingsPending">Unsaved speech preferences</span><span v-else>Speech preferences are up to date</span></div>
                    <button class="btn btn-primary" :disabled="recording || status === 'connecting' || status === 'stopping' || settingsSaving" @click="applySpeechSettings()">{{ settingsSaving ? 'Applying...' : 'Apply speech settings' }}</button>
                  </div>
                  <div class="config-utilities">
                    <button type="button" class="config-disclosure" :aria-expanded="showDiagnostics" @click="showDiagnostics = !showDiagnostics"><AppIcon name="settings" :size="15" /> Technical diagnostics <span>{{ showDiagnostics ? '−' : '+' }}</span></button>
                    <div v-if="showDiagnostics" class="diagnostics-content">
                      <Phase07Status :segments="segments" :buffered-ms="bufferedMs" :recording="recording" :asr-engine="serverInfo?.asr_engine ?? 'unknown'" :asr-model="serverInfo?.asr_model" :diarization-mode="serverInfo?.diarization_mode" />
                      <p v-if="serverInfo" class="hint-text">ASR {{ serverInfo.asr_model }} · Window {{ serverInfo.processing_window_s }}s · Confirmation {{ serverInfo.confirmation_delay_s }}s · Hotwords {{ serverInfo.active_hotword_count ?? 0 }}</p>
                      <p v-if="serverInfo?.resumed && serverInfo.speaker_label_offset" class="hint-text">
                        Resumed recordings reuse earlier speaker numbers when the voice evidence agrees.
                      </p>
                      <p v-if="audioAnalysis" class="hint-text">Audio analysis: {{ audioAnalysis }}</p>
                    </div>
                  </div>
                </section>
          </template>
          <template v-else-if="activeMeetingTool === 'recordings'">
            <section class="meeting-recordings-panel" aria-label="Saved recordings">
                  <div class="config-panel-heading">
                    <div><span class="transcript-eyebrow">AUDIO LIBRARY</span><h2>Saved recordings</h2><p>Listen to a recording or download the original WAV file.</p></div>
                    <button type="button" class="meeting-back" aria-label="Close recordings" @click="selectMeetingTool('transcript')"><AppIcon name="x" :size="16" /></button>
                  </div>
                  <div class="recordings-toolbar">
                    <span>{{ recordingList.length }} recording{{ recordingList.length === 1 ? '' : 's' }}</span>
                    <button type="button" class="btn btn-ghost btn-sm" @click="loadRecordings()">Refresh list</button>
                  </div>
                  <div v-if="recordingList.length === 0" class="recording-empty">
                    <AppIcon name="mic" :size="22" />
                    <h3>No saved recordings</h3>
                    <p>Enable Save audio recording in Settings before starting a new session.</p>
                  </div>
                  <div v-else class="recording-card-grid">
                    <article v-for="(item, index) in (showAllRecordings ? recordingList : recordingList.slice(0, 4))" :key="item.name" class="recording-media-card">
                      <div class="recording-media-top"><span class="recording-media-icon"><AppIcon name="mic" :size="17" /></span><div><h3>Recording {{ index + 1 }}</h3><p>{{ item.duration_s === null ? 'Duration unavailable' : formatTime(item.duration_s * 1000) }} · {{ (item.bytes / 1048576).toFixed(1) }} MB</p></div></div>
                      <audio controls preload="metadata" :src="recordingUrl(item.name)" :aria-label="`Play recording ${index + 1}`"></audio>
                      <a :href="recordingUrl(item.name)" :download="item.name" class="recording-download"><AppIcon name="download" :size="15" /> Download WAV</a>
                    </article>
                  </div>
                  <button v-if="recordingList.length > 4" type="button" class="recordings-show-all" @click="showAllRecordings = !showAllRecordings">{{ showAllRecordings ? 'Show fewer recordings' : `Show all ${recordingList.length} recordings` }}</button>
                </section>
          </template>
        </div>
      </aside>
    </div>

    <div v-if="showParticipants" class="dialog-backdrop" @click.self="showParticipants = false">
      <div class="dialog dialog-wide">
        <h3>Meeting Participants</h3>
        <p class="dialog-sub">
          Select participants to limit matching to the people you expect. If none are
          selected, the system checks eligible saved CAM++ voiceprints in your
          local library with a stricter similarity and ambiguity threshold.
          Uncertain speech remains anonymous. Automatic matching can be
          disabled by setting MEETING_AUTO_SPEAKER_MATCH=0 on the server.
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
            <span class="muted-text" :class="{ 'warn-text': spk.vector_count === 0 }">{{ spk.vector_count }} voiceprints {{ spk.vector_count === 0 ? '(enrollment required)' : '' }}</span>
          </label>
          <div v-if="speakers.length === 0" class="empty-state small">
            <p>No enrolled voiceprints. Add a speaker first.</p>
          </div>
        </div>
        <div class="threshold-row">
          <label class="field">
            Match Threshold
            <input
              v-model.number="spkThreshold"
              type="number"
              min="0"
              max="1"
              step="0.05"
            />
          </label>
          <span class="muted-text">
            The default threshold is 0.3. Lower values match more readily.
          </span>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          Stop recording before changing participants.
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showParticipants = false">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyParticipants"
          >
            {{ applying ? 'Saving...' : 'Save participants' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="showHotwords" class="dialog-backdrop" @click.self="showHotwords = false">
      <div class="dialog dialog-wide">
        <h3>Meeting Hotwords</h3>
        <p class="dialog-sub">
          Selected hotwords improve recognition of specialized terms,
          names, and products, up to 200 hotwords per meeting.
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
                <span class="muted-text">{{ lib.word_count }} hotwords</span>
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
            <p>No hotword libraries yet. Create one on the home page.</p>
          </div>
        </div>
        <p v-if="recording" class="hint-text warn-text">
          Stop recording before changing hotwords.
        </p>
        <div class="dialog-actions">
          <button class="btn btn-ghost" @click="showHotwords = false">Cancel</button>
          <button
            class="btn btn-primary"
            :disabled="recording || applying"
            @click="applyHotwords"
          >
            {{ applying ? 'Saving...' : 'Save hotwords' }}
          </button>
        </div>
      </div>
    </div>

    <Transition name="meeting-alert">
      <aside v-if="error" class="meeting-floating-alert" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="meeting-floating-alert-icon"><AppIcon name="alert" :size="18" /></div>
        <div class="meeting-floating-alert-body">
          <strong>Recording needs attention</strong>
          <p>{{ error }}</p>
          <div class="meeting-floating-alert-actions">
            <button type="button" @click="selectMeetingTool('settings')">Settings</button>
            <button type="button" @click="selectMeetingTool('recordings')">Recordings</button>
          </div>
        </div>
        <button type="button" class="meeting-floating-alert-dismiss" aria-label="Dismiss recording notification" @click="error = ''"><AppIcon name="x" :size="16" /></button>
      </aside>
    </Transition>

    <Transition name="toast">
      <div v-if="toast" class="toast">
        <AppIcon name="check" :size="16" />
        {{ toast }}
      </div>
    </Transition>
  </div>
</template>


<style scoped>
.recorded-audio-item {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  padding: 0.75rem;
  border: 1px solid var(--border, #d1d5db);
  border-radius: 10px;
  min-width: 280px;
}
.recorded-audio-item > span {
  min-width: 180px;
  font-size: 0.8rem;
}
.recorded-audio-item audio {
  min-width: 190px;
  max-width: 340px;
  flex: 1 1 200px;
}
.speaker-edit-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.5rem;
}
.speaker-edit-controls select {
  min-height: 2.1rem;
  max-width: 240px;
}
</style>

<style src="../assets/meeting-refresh.css"></style>
<style src="../assets/meeting-tool-layout.css"></style>
