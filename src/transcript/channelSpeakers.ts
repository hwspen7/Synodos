import type { Segment } from '../types'
import { speakerColor } from '../speaker/clustering.ts'

function channelOf(segment: Segment): number | undefined {
  const channel = segment.speaker_acoustic_channel
  return typeof channel === 'number' && Number.isInteger(channel) && channel > 0
    ? channel : undefined
}

function hasChannelVoiceConflict(segment: Segment): boolean {
  return segment.speaker_channel_gate_reason === 'existing_channel_voice_conflict' ||
    segment.speaker_channel_gate_reason === 'existing_channel_matches_other_voice'
}

function confirmedSpeaker(segment: Segment): number | undefined {
  return segment.speaker_id > 0 && !segment.speaker_uncertain &&
    segment.speaker_final_resolution_status === 'independent_acoustic_assigned'
    ? segment.speaker_continuity_id ?? segment.speaker_id : undefined
}

function hasNamedIdentity(segment: Segment): boolean {
  return segment.speaker_id > 0 && Boolean(segment.speaker_name) && (
    segment.speaker_diarization_source === 'registered_voiceprint' ||
    segment.speaker_final_resolution_status === 'manual_override' ||
    Boolean(segment.matched_profile_id && segment.matched_profile_name)
  )
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
    const identity = hasChannelVoiceConflict(segment) ? undefined : confirmedSpeaker(segment)
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
  const reservedLabels = new Set(canonical.filter(hasNamedIdentity).map(segment => segment.speaker_id))
  const fallbackLabels = new Map<string, number>()
  const projected = new Map<number, Segment>()
  const ordered = [...canonical].sort((a, b) => a.start_ms - b.start_ms || a.id - b.id)
  for (const segment of ordered) {
    if (hasNamedIdentity(segment)) {
      projected.set(segment.id, segment)
      continue
    }
    const channel = channelOf(segment)
    const stream = segment.speaker_acoustic_stream_start ?? 0
    const channelLabel = labels.get(`${stream}:${channel}`)
    const conflict = hasChannelVoiceConflict(segment)
    const confirmed = confirmedSpeaker(segment)
    const identity = conflict ? confirmed : confirmed ?? segment.speaker_id
    const known = confirmed === undefined ? undefined : priorIdentities.get(confirmed)
    const matchedLabel = known?.size === 1 ? [...known][0] : undefined
    // A conflicting row can use a separately corroborated voice, but
    // must not inherit the heading of the channel that rejected it.
    let label = conflict
      ? matchedLabel === channelLabel ? undefined : matchedLabel
      : channelLabel ?? matchedLabel
    const allocated = label === undefined
    if (label === undefined) {
      const unconfirmedChannel = !conflict && channel !== undefined &&
        streams.get(stream)?.get(channel)?.size === 0
      const key = unconfirmedChannel ? `channel:${stream}:${channel}`
        : identity !== undefined && identity > 0
          ? `${conflict ? 'conflict' : 'voice'}:${stream}:${identity}`
          : `caption:${segment.id}`
      label = fallbackLabels.get(key)
      if (label === undefined) {
        do { largest += 1 } while (reservedLabels.has(largest))
        label = largest
        fallbackLabels.set(key, label)
      }
    }
    projected.set(segment.id, { ...segment, speaker_id: label,
      speaker_name: `Speaker ${label}`, color: speakerColor(label),
      speaker_uncertain: allocated || segment.speaker_uncertain })
  }
  return canonical.map(segment => projected.get(segment.id)!)
}
