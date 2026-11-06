import test from 'node:test'
import assert from 'node:assert/strict'
import { projectReportedSpeakers } from '../../../src/transcript/reportedSpeakers.ts'
import { buildTranscriptParagraphs } from '../../../src/transcript/paragraphs.ts'

function segment(id, start, end, fusedId, channel, extra = {}) {
  return {
    id, start_ms: start, end_ms: end, speaker_id: fusedId,
    speaker_name: `Speaker ${fusedId}`, speaker_acoustic_channel: channel,
    text: `Turn ${id}`, color: '#abcdef', ...extra,
  }
}

test('report speaker assignments win over conflicting acoustic channels', () => {
  const rows = [
    segment(1, 2000, 4000, 1, 1),
    segment(6, 31000, 32000, 4, null, {
      speaker_final_resolution_status: 'new_speaker_provisional',
    }),
    segment(14, 55000, 57000, 3, 5, { speaker_uncertain: true }),
    segment(25, 87144, 87510, 2, 1, { speaker_uncertain: true }),
    segment(26, 87490, 87778, 2, 2, { speaker_uncertain: true }),
    segment(27, 126000, 128000, 2, 3, {
      language: 'zh', translation: 'English caption', translation_target: 'en',
    }),
  ]
  const before = JSON.stringify(rows)
  const projected = projectReportedSpeakers(rows)
  assert.deepEqual(projected.map(row => row.speaker_name),
    ['Speaker 1', 'Speaker 4', 'Speaker 3', 'Speaker 2', 'Speaker 2', 'Speaker 2'])
  const paragraphs = buildTranscriptParagraphs(projected, {
    channelHeadings: true, singlePrimarySpeaker: false, translationTarget: 'en',
  })
  assert.deepEqual(paragraphs.filter(row => row.overlapping).map(row => row.speaker_id), [2, 2])
  assert.deepEqual(paragraphs.flatMap(row => row.segmentIds), [1, 6, 14, 25, 26, 27])
  assert.equal(paragraphs.at(-1).text, 'English caption')
  assert.equal(paragraphs.at(-1).originalText, 'Turn 27')
  assert.equal(JSON.stringify(rows), before)
})

test('a new recording stream preserves reported speaker IDs instead of allocating offsets', () => {
  const old = segment(1, 0, 2000, 1, 1)
  const resumed = segment(2, 4000, 6000, 2, 3, { speaker_acoustic_stream_start: 64000 })
  const later = segment(3, 7000, 8000, 3, 1, { speaker_acoustic_stream_start: 64000 })
  assert.deepEqual(projectReportedSpeakers([old, resumed]).map(row => row.speaker_id), [1, 2])
  assert.deepEqual(projectReportedSpeakers([old, resumed, later]).map(row => row.speaker_id), [1, 2, 3])
})

test('voice-only meetings retain their existing view and manual identities survive', () => {
  const manual = segment(2, 3000, 4500, 8, 1, {
    speaker_name: 'Alex', speaker_final_resolution_status: 'manual_override',
  })
  assert.equal(projectReportedSpeakers([segment(1, 0, 2000, 1, null)]), null)
  assert.equal(projectReportedSpeakers([segment(1, 0, 2000, 1, 1), manual])[1], manual)
})

test('confirmed voices keep their headings when channels swap after a pause', () => {
  const confirmed = { speaker_final_resolution_status: 'independent_acoustic_assigned' }
  const first = segment(1, 0, 2000, 1, 1, confirmed)
  const second = segment(2, 2100, 4000, 2, 2, confirmed)
  const pending = segment(3, 6000, 6100, 3, 1, { speaker_acoustic_stream_start: 96000 })
  const resumedSecond = segment(4, 6200, 7300, 2, 1, {
    ...confirmed, speaker_acoustic_stream_start: 96000,
  })
  const resumedFirst = segment(5, 7400, 8800, 1, 2, {
    ...confirmed, speaker_acoustic_stream_start: 96000,
  })
  assert.equal(projectReportedSpeakers([first, second, pending])[2], pending)
  assert.deepEqual(projectReportedSpeakers([first, second, pending, resumedSecond, resumedFirst])
    .map(row => row.speaker_id), [1, 2, 3, 2, 1])
})

test('saved legacy numbers follow their verified continuity aliases', () => {
  const confirmed = { speaker_final_resolution_status: 'independent_acoustic_assigned' }
  const rows = [
    segment(1, 0, 2000, 1, 1, confirmed),
    segment(2, 2100, 4000, 2, 2, confirmed),
    segment(3, 6000, 7600, 3, 1, {
      ...confirmed, speaker_acoustic_stream_start: 96000, speaker_continuity_id: 2,
    }),
    segment(4, 7700, 9000, 4, 2, {
      ...confirmed, speaker_acoustic_stream_start: 96000, speaker_continuity_id: 1,
    }),
  ]
  assert.deepEqual(projectReportedSpeakers(rows).map(row => row.speaker_id), [1, 2, 2, 1])
  assert.deepEqual(rows.map(row => row.speaker_id), [1, 2, 3, 4])
})

test('a provisional speaker already assigned by the report keeps its ID', () => {
  const assigned = segment(1, 76000, 77000, 5, 3, {
    speaker_continuity_id: 3, speaker_uncertain: true,
    speaker_final_resolution_status: 'new_speaker_provisional',
  })
  assert.equal(projectReportedSpeakers([assigned])[0], assigned)
})
