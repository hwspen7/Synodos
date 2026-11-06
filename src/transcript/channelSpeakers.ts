import type { Segment } from '../types'
import { speakerColor } from '../speaker/clustering.ts'

function channelOf(segment: Segment): number | undefined {
  const channel = segment.speaker_acoustic_channel
  return typeof channel === 'number' && Number.isInteger(channel) && channel > 0
    ? channel : undefined
}

function confirmedSpeaker(segment: Segment): number | undefined {
  return segment.speaker_id > 0 && !segment.speaker_uncertain &&
    segment.speaker_final_resolution_status === 'independent_acoustic_assigned'
    ? segment.speaker_continuity_id ?? segment.speaker_id : undefined
}

/** Use raw channels within a recording and corroborated identities across recordings. */
export function projectChannelSpeakers(canonical: Segment[]): Segment[] | null {
  const streams = new Map<number, Map<number, Set<number>>>()
  for (const segment of canonical) {
    const channel = channelOf(segment)
    if (channel === undefined) continue
    const stream = segment.speaker_acoustic_stream_start ?? 0
    const channels = streams.get(stream) ?? new Map<number, Set<number>>()
    const confirmed = channels.get(channel) ?? new Set<number>()
    const identity = confirmedSpeaker(segment)
    if (identity !== undefined) confirmed.add(identity)
    channels.set(channel, confirmed)
    streams.set(stream, channels)
  }
  if (!streams.size) return null

  const labels = new Map<string, number>()
  const priorIdentities = new Map<number, Set<number>>()
  let largest = 0
  for (const [stream, channels] of [...streams].sort(([left], [right]) => left - right)) {
    const offset = largest
    const currentIdentities: Array<[number, number]> = []
    for (const [channel, confirmed] of channels) {
      const identity = confirmed.size === 1 ? [...confirmed][0] : undefined
      const earlier = identity === undefined ? undefined : priorIdentities.get(identity)
      if (priorIdentities.size && (confirmed.size !== 1 || (earlier && earlier.size > 1))) continue
      const label = earlier?.size === 1 ? [...earlier][0]! : offset + channel
      labels.set(`${stream}:${channel}`, label)
      largest = Math.max(largest, label)
      if (identity !== undefined) currentIdentities.push([identity, label])
    }
    for (const [identity, label] of currentIdentities) {
      const knownLabels = priorIdentities.get(identity) ?? new Set<number>()
      knownLabels.add(label)
      priorIdentities.set(identity, knownLabels)
    }
  }
  return canonical.map((segment) => {
    const namedIdentity = segment.speaker_diarization_source === 'registered_voiceprint' ||
      Boolean(segment.matched_profile_id && segment.matched_profile_name) ||
      segment.speaker_final_resolution_status === 'manual_override'
    if (namedIdentity && segment.speaker_id > 0 && segment.speaker_name) return segment
    const channel = channelOf(segment)
    const stream = segment.speaker_acoustic_stream_start ?? 0
    const identity = segment.speaker_continuity_id ?? segment.speaker_id
    const known = priorIdentities.get(identity)
    const label = labels.get(`${stream}:${channel}`) ??
      (known?.size === 1 ? [...known][0] : undefined)
    // Keep the server's existing assignment when no unique channel label exists.
    if (label === undefined) return segment
    return { ...segment, speaker_id: label, speaker_name: `Speaker ${label}`,
      color: speakerColor(label) }
  })
}
