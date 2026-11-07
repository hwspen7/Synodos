import type { Segment } from '../types'
import { speakerColor } from '../speaker/clustering.ts'

/** Keep the report's speaker assignment. Raw channels are acoustic evidence. */
export function projectReportedSpeakers(canonical: Segment[]): Segment[] | null {
  if (!canonical.some(segment =>
    Number.isInteger(segment.speaker_acoustic_channel) && (segment.speaker_acoustic_channel ?? 0) > 0,
  )) return null

  return canonical.map(segment => {
    const alias = segment.speaker_continuity_id
    if (alias === undefined || !Number.isInteger(alias) || alias <= 0 ||
        alias === segment.speaker_id || segment.speaker_id <= 0 || segment.speaker_uncertain ||
        segment.speaker_final_resolution_status !== 'independent_acoustic_assigned') {
      return segment
    }
    return { ...segment, speaker_id: alias, speaker_name: `Speaker ${alias}`,
      color: speakerColor(alias) }
  })
}
