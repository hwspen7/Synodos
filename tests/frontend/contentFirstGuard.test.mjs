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

test('does not publish a brief phantom speaker between stable words', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 2100, 'since I was a teenager'),
    cue(2, 2100, 2800, 'or'),
    cue(1, 2800, 5100, 'pre-teen I loved physics'),
  ]), false)
})

test('requires at least 1.5 seconds of evidence on both sides of a change', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 2000, 'I applied to'),
    cue(2, 2000, 3500, 'Princeton'),
  ]), true)
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 2200, 'I applied'),
    cue(2, 2200, 3100, 'to Princeton'),
  ]), false)
})

test('declines fast alternating identity projections', () => {
  assert.equal(hasStableSpeakerCues([
    cue(1, 0, 1600), cue(2, 1600, 3200),
    cue(1, 3200, 4800), cue(2, 4800, 6400),
  ]), false)
})

test('registered voice identities are never replaced by anonymous word cues', () => {
  assert.match(source, /speaker_diarization_source === 'registered_voiceprint'/)
  assert.match(source, /Boolean\(segment\.matched_profile_id && segment\.matched_profile_name\)/)
})
