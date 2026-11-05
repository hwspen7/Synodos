import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import ts from 'typescript'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const source = readFileSync(path.join(root, 'src/transcript/contentFirst.ts'), 'utf8')
const start = source.indexOf('export function hasStableSpeakerCues(')
const end = source.indexOf('export function expandContentFirstSegments(', start)
assert.ok(start >= 0 && end > start, 'Speaker cue guard must be exported')
const snippet = source.slice(start, end)
const js = ts.transpileModule(snippet, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText
const { hasStableSpeakerCues } = await import(
  'data:text/javascript;charset=utf-8,' + encodeURIComponent(js)
)

const cue = (speaker_id, start_ms, end_ms, text = 'speech') =>
  ({ speaker_id, start_ms, end_ms, text })

test('keeps a stable two-speaker transition when both turns have support', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 1200), cue(1, 1200, 2200),
    cue(2, 2200, 3200), cue(2, 3200, 4100),
  ]), true)
})

test('rejects single-character micro-cues and speaker oscillations', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 8000, 9200), cue(2, 9200, 9350, 'a'),
    cue(1, 9350, 10300),
  ]), false)
})

test('does not turn unknown speaker fragments into new visible turns', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 1200), cue(-1, 1200, 1600), cue(1, 1600, 2600),
  ]), false)
})

test('rejects a short unsupported speaker run', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 500), cue(2, 500, 1600),
  ]), false)
})

test('rejects overlapping projected cue timestamps', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 1000), cue(1, 900, 1800),
  ]), false)
})
