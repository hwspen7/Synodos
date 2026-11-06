/** Presentation-only grouping of consecutive caption segments by speaker.
 *
 * Never change the source ASR segments, timestamps, speaker labels, or text.
 * A paragraph is a view grouping, not a second speaker turn or ASR commit.
 */
import type { Segment, Paragraph } from '../types'

const CJK = /[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]/u

function joinCaptionText(left: string, right: string): string {
  const a = left.trimEnd()
  const b = right.trimStart()
  if (!a) return b
  if (!b) return a
  if (/[[({\uff08\u300a\u300c\u300e\u201c\u2018]$/u.test(a) ||
      /[-'\u2019]$/u.test(a) || /^[-'\u2019]/u.test(b) ||
      /^[,!.?;:%)\]}\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u300b\u300d\u300f\u201d\u2019]/u.test(b)) {
    return a + b
  }
  if (CJK.test(a.slice(-1)) || CJK.test(b.charAt(0))) return a + b
  return a + ' ' + b
}

export function buildTranscriptParagraphs(
  segments: readonly Segment[], opts: {
    mergeGapS?: number
    translationTarget?: string
    maxParagraphMs?: number
    maxParagraphChars?: number
    singlePrimarySpeaker?: boolean
  } = {},
): Paragraph[] {
  // The previous 750 ms hard cap fragmented continuous conversations
  // into a separate card for each small pause. Keep the source turns intact
  // and group consecutive utterances for reading within a bounded paragraph.
  const gapMs = Math.min(2500, Math.max(0, Math.round((opts.mergeGapS ?? 2) * 1000)))
  const maxDuration = opts.maxParagraphMs ?? 28000
  const maxChars = opts.maxParagraphChars ?? 520
  const ordered = [...segments].sort((a, b) => a.start_ms - b.start_ms || a.id - b.id)
  const paragraphs: Paragraph[] = []
  for (const seg of ordered) {
    if (!seg.text.trim()) continue
    const translated = opts.translationTarget && seg.translation_target === opts.translationTarget
      ? seg.translation?.trim() : undefined
    const text = translated || seg.text
    const last = paragraphs[paragraphs.length - 1]
    const canonicalId = seg.source_segment_id ?? seg.id
    const provisional = seg.speaker_diarization_source === 'nemotron3_channel_provisional' ||
      seg.speaker_final_resolution_status === 'existing_speaker_provisional' ||
      seg.speaker_final_resolution_status === 'new_speaker_provisional'
    if (opts.singlePrimarySpeaker && last && seg.start_ms < last.end_ms) {
      // Overlap is one display interval. Keep both original texts and
      // editing targets while showing only one primary speaker heading.
      last.overlapping = true
      if (seg.speaker_id > 0 &&
          (last.speaker_id <= 0 || (last.provisional && !provisional))) {
        last.speaker_id = seg.speaker_id
        last.speaker_name = seg.speaker_name
        last.color = seg.color
        last.provisional = provisional
      }
      last.end_ms = Math.max(last.end_ms, seg.end_ms)
      if (!last.segmentIds.includes(canonicalId)) last.segmentIds.push(canonicalId)
      if (last.originalText || translated) {
        last.originalText = joinCaptionText(last.originalText || last.text, seg.text)
      }
      last.text = joinCaptionText(last.text, text)
      continue
    }
    // A genuine speaker switch, overlapping speech or an unknown identity
    // remains separate. Sentence punctuation alone is not a speaker switch:
    // multiple sentences from the same voice form one readable paragraph.
    const canMerge = Boolean(
      last && !last.overlapping && seg.speaker_id > 0 &&
      seg.speaker_id === last.speaker_id &&
      Boolean(last.originalText) === Boolean(translated) &&
      (seg.speaker_name || '') === (last.speaker_name || '') &&
      !(seg.overlap_intervals_ms?.length) &&
      seg.start_ms >= last.end_ms &&
      seg.start_ms - last.end_ms <= gapMs &&
      // An explicitly repeated sentence is a separate utterance, not a
      // duplicate to be visually glued to its predecessor.
      last.text.trim() !== text.trim() &&
      seg.end_ms - last.start_ms <= maxDuration &&
      last.text.length + text.length + 1 <= maxChars
    )
    if (canMerge && last) {
      last.provisional ||= provisional
      last.end_ms = Math.max(last.end_ms, seg.end_ms)
      if (!last.segmentIds.includes(canonicalId)) last.segmentIds.push(canonicalId)
      last.text = joinCaptionText(last.text, text)
      if (last.originalText) last.originalText = joinCaptionText(last.originalText, seg.text)
    } else {
      paragraphs.push({
        id: seg.id, speaker_id: seg.speaker_id,
        speaker_name: seg.speaker_name,
        start_ms: seg.start_ms, end_ms: seg.end_ms,
        text, originalText: translated ? seg.text : undefined,
        segmentIds: [canonicalId], color: seg.color,
        provisional,
      })
    }
  }
  return paragraphs
}
