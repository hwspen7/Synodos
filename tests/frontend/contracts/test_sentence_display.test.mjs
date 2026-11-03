import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import ts from 'typescript'

// Exercise the actual production Vue sentence formatter, not a duplicate
// implementation that might conceal a UI regression.
const vue = readFileSync(join(process.cwd(), 'src/pages/MeetingPage.vue'), 'utf8')
const start = vue.indexOf('function splitDisplaySentences(')
const end = vue.indexOf('\nfunction endsFinishedThought(', start)
assert.ok(start > 0 && end > start, 'The production formatter must be present')
const source = vue.slice(start, end) + '\nexports.splitDisplaySentences = splitDisplaySentences'
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText
const module = { exports: {} }
new Function('exports', compiled)(module.exports)
const split = module.exports.splitDisplaySentences

test('user report displays six separate sentences rather than one block', () => {
  const input = "Really, what a coincidence. We laughed a lot. Absolutely. Hey, how are you? Hey, what's man? How are you doing?"
  assert.deepEqual(split(input), [
    'Really, what a coincidence.',
    'We laughed a lot.',
    'Absolutely.',
    'Hey, how are you?',
    "Hey, what's man?",
    'How are you doing?',
  ])
  assert.equal(split(input).join(' '), input)
})

test('abbreviations and decimal points do not cause spurious sentence cuts', () => {
  assert.deepEqual(split('Dr. Smith said 3.14 is fine. Hello!'), [
    'Dr. Smith said 3.14 is fine.', 'Hello!',
  ])
  assert.deepEqual(split('U.S. Navy. How are you?'), ['U.S. Navy.', 'How are you?'])
})

test('CJK punctuation creates visual boundaries even without spaces', () => {
  const input = '\u4f60\u597d\u3002\u6700\u8fd1\u600e\u4e48\u6837\uff1f\u5f88\u597d\uff01'
  assert.deepEqual(split(input), [
    '\u4f60\u597d\u3002', '\u6700\u8fd1\u600e\u4e48\u6837\uff1f',
    '\u5f88\u597d\uff01',
  ])
})

test('legacy transcripts without terminal punctuation preserve text', () => {
  assert.deepEqual(split("I don't like reading I like play games"), [
    "I don't like reading I like play games",
  ])
})
