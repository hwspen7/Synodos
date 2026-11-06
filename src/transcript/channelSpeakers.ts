import type { Segment } from '../types'
import { speakerColor } from '../speaker/clustering.ts'

function channelOf(segment: Segment): number | undefined {
  const channel = segment.speaker_acoustic_channel
  return typeof channel === 'number' && Number.isInteger(channel) && channel > 0
    ? channel : undefined
}

/** Display diarizer channels without changing the canonical identity evidence. */
export function projectChannelSpeakers(canonical: Segment[]): Segment[] | null {
  const streams = new Map<number, Set<number>>()
  for (const segment of canonical) {
    const channel = channelOf(segment)
    if (channel === undefined) continue
    const stream = segment.speaker_acoustic_stream_start ?? 0
    const channels = streams.get(stream) ?? new Set<number>()
    channels.add(channel)
    streams.set(stream, channels)
  }
  if (!streams.size) return null

  const labels = new Map<string, number>()
  const orderedStreams = [...streams].sort(([left], [right]) => left - right)
  let largest = 0
  for (const [stream, channels] of orderedStreams) {
    const offset = largest
    for (const channel of channels) {
      const label = offset + channel
      labels.set(`${stream}:${channel}`, label)
      largest = Math.max(largest, label)
    }
  }
  return canonical.map((segment) => {
    const namedIdentity = segment.speaker_diarization_source === 'registered_voiceprint' ||
      Boolean(segment.matched_profile_id && segment.matched_profile_name) ||
      segment.speaker_final_resolution_status === 'manual_override'
    if (namedIdentity && segment.speaker_id > 0 && segment.speaker_name) return segment
    const channel = channelOf(segment)
    const stream = segment.speaker_acoustic_stream_start ?? 0
    const label = labels.get(`${stream}:${channel}`)
    if (label === undefined) {
      return { ...segment, speaker_id: 0, speaker_name: 'Speaker pending',
        speaker_uncertain: false, color: '#8892a0' }
    }
    return { ...segment, speaker_id: label, speaker_name: `Speaker ${label}`,
      color: speakerColor(label) }
  })
}
