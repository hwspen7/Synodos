import test from 'node:test'
import assert from 'node:assert/strict'
import { buildTranscriptParagraphs } from '../../../src/transcript/paragraphs.ts'

function segment(id, start_ms, end_ms, text, speaker_id = 1, more = {}) {
  return { id, start_ms, end_ms, text, speaker_id,
    speaker_name: speaker_id > 0 ? 'Speaker ' + speaker_id : 'Unidentified Speaker',
    color: '#68389A', ...more }
}

test('continuous long speech is one readable speaker paragraph, not three artificial turns', () => {
  const parts = buildTranscriptParagraphs([
    segment(1, 50000, 57000, 'I have seen others who really did not have the talent'),
    segment(2, 57050, 63000, 'and they are among the most successful people today'),
    segment(3, 63200, 70000, 'because they never quit and never give up.'),
  ])
  assert.equal(parts.length, 1)
  assert.equal(parts[0].start_ms, 50000)
  assert.equal(parts[0].end_ms, 70000)
  assert.deepEqual(parts[0].segmentIds, [1, 2, 3])
  assert.equal(parts[0].text,
    'I have seen others who really did not have the talent and they are among the most successful people today because they never quit and never give up.')
})

test('three explicitly repeated sentences remain three actual turns', () => {
  const utterance = 'Hey, how are you, man?'
  const parts = buildTranscriptParagraphs([
    segment(1, 19000, 20000, utterance),
    segment(2, 22000, 23000, utterance),
    segment(3, 26000, 27000, utterance),
  ])
  assert.equal(parts.length, 3)
  assert.deepEqual(parts.map(p => p.text), [utterance, utterance, utterance])
})

test('different speakers cannot be merged even without a pause', () => {
  const parts = buildTranscriptParagraphs([
    segment(1, 1000, 2000, 'Can you'),
    segment(2, 2000, 2600, 'hear me?', 2),
    segment(3, 2600, 2900, 'Yes.', 1),
  ])
  assert.equal(parts.length, 3)
  assert.deepEqual(parts.map(p => p.speaker_id), [1, 2, 1])
})

test('unknown speaker and overlapping turns are never collapsed into one speaker', () => {
  const parts = buildTranscriptParagraphs([
    segment(1, 0, 900, 'Unknown', -1),
    segment(2, 950, 1300, 'speaker', -1),
    segment(3, 1500, 2300, 'Speaker one'),
    segment(4, 2320, 2800, 'overlap', 1, {
      overlap_intervals_ms: [{ start_ms: 2310, end_ms: 2800, speaker_ids: [1, 2] }],
    }),
  ])
  assert.equal(parts.length, 4)
})

test('hard paragraph bound and pause preserve meeting readability without changing source', () => {
  const original = [
    segment(1, 0, 15000, 'This is a continuous first thought'),
    segment(2, 15010, 29000, 'that exceeds the maximum paragraph span'),
    segment(3, 32000, 33000, 'New thought'),
  ]
  const before = JSON.stringify(original)
  const parts = buildTranscriptParagraphs(original)
  assert.equal(parts.length, 3)
  assert.equal(JSON.stringify(original), before)
  assert.equal(parts[0].text, original[0].text)
})

test('translation display and content-first canonical ids remain intact', () => {
  const parts = buildTranscriptParagraphs([
    segment(-65, 0, 1200, 'Hello', 1, {
      source_segment_id: 12, translation: '',
    }),
    segment(-66, 1250, 1900, 'world', 1, {
      source_segment_id: 12, translation: '',
    }),
  ])
  assert.equal(parts.length, 1)
  assert.deepEqual(parts[0].segmentIds, [12])
  const translated = buildTranscriptParagraphs([
    segment(5, 0, 1000, 'Bonjour', 1, { translation: 'Hello', translation_target: 'en' }),
  ], { translationTarget: 'en' })
  assert.equal(translated[0].text, 'Hello')
})

test('an overlapping provisional speaker yields the heading to a supported speaker', () => {
  const parts = buildTranscriptParagraphs([
    segment(1, 0, 1000, 'First', 2, {
      speaker_final_resolution_status: 'new_speaker_provisional',
    }),
    segment(2, 400, 1300, 'Second', 1),
  ], { singlePrimarySpeaker: true })
  assert.equal(parts.length, 1)
  assert.equal(parts[0].speaker_id, 1)
  assert.equal(parts[0].provisional, false)
  assert.deepEqual(parts[0].segmentIds, [1, 2])
})

test('a numbered uncertain caption retains its provisional badge when merged', () => {
  const parts = buildTranscriptParagraphs([
    segment(1, 0, 1000, 'First'),
    segment(2, 1100, 1400, 'Second', 1, {
      speaker_final_resolution_status: 'existing_speaker_provisional',
    }),
  ])
  assert.equal(parts.length, 1)
  assert.equal(parts[0].provisional, true)
})

test('display keeps the original and ignores translations for a different target', () => {
  const rows = [segment(7, 0, 1000, 'Bonjour', 1, {
    translation: 'Hello', translation_target: 'en',
  })]
  assert.equal(buildTranscriptParagraphs(rows, { translationTarget: 'en' })[0].originalText, 'Bonjour')
  const wrongTarget = buildTranscriptParagraphs(rows, { translationTarget: 'zh' })[0]
  assert.equal(wrongTarget.text, 'Bonjour')
  assert.equal(wrongTarget.originalText, undefined)
  assert.equal(buildTranscriptParagraphs(rows)[0].text, 'Bonjour')
})
