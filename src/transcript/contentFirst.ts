/** Safe view-only splitting of confirmed ASR into word-preserving speaker cues.
 *
 * No synthetic item can be edited directly or persisted as another caption.
 * Source segment IDs and complete source text remain authoritative.
 */
import type { Segment } from '@/types'
import { speakerColor } from '@/speaker/clustering'
import { isExactContentProjection } from './contentProof'

/** Reject unstable acoustic micro-cues before they become visible speaker turns.
 *
 * An exact text projection alone does not prove a correct speaker assignment.
 * When the diarizer alternates identities within a word or loses identity on
 * sub-second spans, keep the original confirmed caption and its stable label.
 */
export function hasStableSpeakerCues(
  cues: readonly { speaker_id: number; start_ms: number; end_ms: number; text: string }[],
): boolean {
  if (cues.length < 2) return false
  let previousEnd = -1
  let previousSpeaker = -1
  let runDuration = 0
  for (const cue of cues) {
    const duration = cue.end_ms - cue.start_ms
    if (
      !Number.isInteger(cue.speaker_id) || cue.speaker_id <= 0 ||
      !Number.isFinite(duration) || duration < 350 ||
      (previousEnd >= 0 && cue.start_ms < previousEnd)
    ) return false
    if (cue.speaker_id !== previousSpeaker) {
      // A speaker change cannot be justified by a short orphaned syllable.
      if (previousSpeaker > 0 && runDuration < 700) return false
      previousSpeaker = cue.speaker_id
      runDuration = 0
    }
    runDuration += duration
    previousEnd = cue.end_ms
  }
  return runDuration >= 700
}

export function expandContentFirstSegments(
  canonical: Segment[], useTranslation: boolean,
): Segment[] {
  const out: Segment[] = []
  for (const segment of canonical) {
    const cues = segment.caption_cues
    if (
      !Array.isArray(cues) || cues.length < 2 || cues.length > 32 ||
      segment.caption_projection_status !== 'split_verified' ||
      segment.speaker_diarization_source === 'manual_override' ||
      (useTranslation && !!segment.translation?.trim()) ||
      !hasStableSpeakerCues(cues) ||
      !isExactContentProjection(
        segment.text, segment.start_ms, segment.end_ms, cues,
      )
    ) {
      out.push(segment)
      continue
    }
    // Distinct negative view-only IDs prevent cross-record collisions,
    // while source_segment_id keeps edits mapped to the actual backend ID.
    // If the canonical ID is too large for a safe synthetic integer, fall
    // back to the canonical text.
    if (!Number.isSafeInteger(segment.id) ||
        segment.id <= 0 || segment.id > 1_000_000_000) {
      out.push(segment)
      continue
    }
    cues.forEach((cue, index) => out.push({
      ...segment,
      id: -(segment.id * 64 + index + 1),
      source_segment_id: segment.id,
      text: cue.text,
      start_ms: cue.start_ms,
      end_ms: cue.end_ms,
      speaker_id: cue.speaker_id,
      speaker_name: cue.speaker_id > 0
        ? cue.speaker_name : 'Unidentified Speaker',
      color: cue.speaker_id > 0 ? speakerColorForCue(cue.speaker_id)
        : segment.color,
      // Split text does NOT imply translated text can be split identically.
      translation: '',
    }))
  }
  return out
}

function speakerColorForCue(speakerId: number): string {
  // Use the existing deterministic voice color mapping, without altering
  // identity in canonical segments.
  return speakerColor(speakerId)
}
