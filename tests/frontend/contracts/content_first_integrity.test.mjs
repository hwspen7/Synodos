import test from 'node:test'
import assert from 'node:assert/strict'
import { isExactContentProjection } from '../../../src/transcript/contentProof.ts'

test('one canonical transcript can display two voices without deleting text', () => {
  const source = 'Hello, how are you?'
  const cues = [
    { start_ms: 0, end_ms: 430, text: 'Hello,' },
    { start_ms: 450, end_ms: 950, text: ' how are you?' },
  ]
  assert.equal(isExactContentProjection(source, 0, 1000, cues), true)
  assert.equal(cues.map((cue) => cue.text).join(''), source)
})

test('silently dropped or duplicated words fail content projection', () => {
  const source = 'Hello, how are you?'
  const retained = [
    { start_ms: 0, end_ms: 430, text: 'Hello,' },
    { start_ms: 450, end_ms: 950, text: ' how are you?' },
  ]
  for (const broken of [
    [retained[0], { ...retained[1], text: ' are you?' }],
    [retained[0], { ...retained[1], text: ' how how are you?' }],
    [retained[0], { ...retained[1], text: ' how are you!' }],
    [{ ...retained[0], end_ms: 600 }, retained[1]],
    [{ ...retained[0], start_ms: -500 }, retained[1]],
  ]) {
    assert.equal(isExactContentProjection(source, 0, 1000, broken), false)
  }
})

test('CJK transcript and punctuation survive literal projection unchanged', () => {
  const source = '\u4f60\u597d\uff0c\u4eca\u5929\u600e\u4e48\u6837\uff1f'
  const cues = [
    { start_ms: 100, end_ms: 450, text: '\u4f60\u597d\uff0c' },
    { start_ms: 550, end_ms: 1200, text: '\u4eca\u5929\u600e\u4e48\u6837\uff1f' },
  ]
  assert.equal(isExactContentProjection(source, 100, 1250, cues), true)
  assert.equal(isExactContentProjection(source, 100, 1250, [
    cues[0], { ...cues[1], text: '\u4eca\u5929\u600e\u4e48\u6837' },
  ]), false)
})

test('unsafe timing or zero split fails rather than mutating source', () => {
  assert.equal(isExactContentProjection('a b', 0, 1000, []), false)
  assert.equal(isExactContentProjection('a b', 0, 1000, [
    { start_ms: 0, end_ms: 300, text: 'a' },
    { start_ms: 1000, end_ms: 1200, text: ' b' },
  ]), false)
})
