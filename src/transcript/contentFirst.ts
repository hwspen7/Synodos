/** Safe view-only splitting of confirmed ASR into word-preserving speaker cues.
 *
 * No synthetic item can be edited directly or persisted as another caption.
 * Source segment IDs and complete source text remain authoritative.
 */
import type { Segment } from '@/types'

function plain(text: string): string {
  return text.normalize('NFKC').trim()
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
      cues.some((cue) =>
        !Number.isInteger(cue.speaker_id) ||
        !Number.isInteger(cue.start_ms) ||
        !Number.isInteger(cue.end_ms) ||
        cue.end_ms <= cue.start_ms ||
        cue.start_ms < segment.start_ms - 150 ||
        cue.end_ms > segment.end_ms + 150 ||
        !cue.text?.trim()
      ) ||
      cues.some((cue, index) =>
        index > 0 && cue.start_ms < cues[index - 1]!.end_ms
      ) ||
      plain(cues.map((cue) => cue.text).join('')) !== plain(segment.text)
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
      text: cue.text.trim(),
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

import { speakerColor } from '@/speaker/clustering'
