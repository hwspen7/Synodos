import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import ts from 'typescript'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const source = readFileSync(path.join(root, 'src/transcript/paragraphs.ts'), 'utf8')
const js = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText
const { buildTranscriptParagraphs } = await import(
  'data:text/javascript;charset=utf-8,' + encodeURIComponent(js)
)
const turn = (id, speaker_id, start_ms, end_ms, text) => ({
  id, speaker_id, speaker_name: speaker_id > 0
    ? `Speaker ${speaker_id}` : 'Unidentified Speaker',
  start_ms, end_ms, text,
})

test('continuous same-speaker sentences form one readable paragraph', () => {
  const result = buildTranscriptParagraphs([
    turn(1, 5, 60000, 61400, 'Physics was my first love.'),
    turn(2, 5, 62100, 65000, 'I still love physics.'),
    turn(3, 5, 65900, 68000, 'I applied to Princeton.'),
  ])
  assert.equal(result.length, 1)
  assert.deepEqual(result[0].segmentIds, [1, 2, 3])
  assert.equal(result[0].text, 'Physics was my first love. I still love physics. I applied to Princeton.')
})

test('an unidentified interval is never invented as the neighboring speaker', () => {
  const result = buildTranscriptParagraphs([
    turn(1, 5, 60000, 63000, 'I applied to Princeton'),
    turn(2, -1, 63000, 64000, 'one'),
    turn(3, 5, 64000, 67500, 'because of Einstein.'),
  ])
  assert.equal(result.length, 3)
  assert.equal(result[1].speaker_id, -1)
  assert.equal(result[1].text, 'one')
})

test('preserves joined hyphenated tokens without changing stored segments', () => {
  const input = [
    turn(1, 5, 0, 1500, 'pre'),
    turn(2, 5, 1600, 3000, '-teen, I loved physics.'),
  ]
  const result = buildTranscriptParagraphs(input)
  assert.equal(result.length, 1)
  assert.equal(result[0].text, 'pre-teen, I loved physics.')
  assert.equal(input[0].text, 'pre')
  assert.equal(input[1].text, '-teen, I loved physics.')
})

test('does not glue speakers across overlap, long pauses or paragraph limits', () => {
  const switched = buildTranscriptParagraphs([
    turn(1, 1, 0, 4000, 'One voice.'),
    turn(2, 2, 4100, 7500, 'Another voice.'),
  ])
  assert.equal(switched.length, 2)
  const paused = buildTranscriptParagraphs([
    turn(1, 1, 0, 1000, 'First.'),
    turn(2, 1, 3800, 4800, 'Later.'),
  ])
  assert.equal(paused.length, 2)
  const bounded = buildTranscriptParagraphs([
    turn(1, 1, 0, 17000, 'First thought.'),
    turn(2, 1, 17500, 34000, 'Second thought.'),
  ])
  assert.equal(bounded.length, 2)
})

test('the meeting shows one primary speaker for overlapping turns without dropping text or edit IDs', () => {
  const input = [
    turn(26, 4, 1000, 2400, 'Physics was my first love.'),
    turn(27, 5, 2100, 3000, 'Awesome.'),
    turn(28, 4, 3000, 4600, 'Today, I still love physics.'),
  ]
  const result = buildTranscriptParagraphs(input, { singlePrimarySpeaker: true })
  assert.deepEqual(result.map(item => [item.speaker_id, item.start_ms, item.end_ms]), [
    [4, 1000, 3000], [4, 3000, 4600],
  ])
  assert.deepEqual(result[0].segmentIds, [26, 27])
  assert.equal(result[0].text, 'Physics was my first love. Awesome.')
  assert.equal(result[0].overlapping, true)
  assert.equal(input[1].speaker_id, 5)
})
