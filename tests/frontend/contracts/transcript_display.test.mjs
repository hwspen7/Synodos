import test from 'node:test'
import assert from 'node:assert/strict'

import { splitDisplaySentences } from '../../../src/transcript/display.ts'

test('multiple completed English thoughts render on distinct caption lines', () => {
  const text = "Really, what a coincidence. We laughed a lot. Absolutely. Hey, how are you? Hey, what's man? How are you doing?"
  assert.deepEqual(splitDisplaySentences(text), [
    'Really, what a coincidence.',
    'We laughed a lot.',
    'Absolutely.',
    'Hey, how are you?',
    "Hey, what's man?",
    'How are you doing?',
  ])
})

test('long unpunctuated speech remains one semantic display block', () => {
  const text = Array.from({ length: 35 },
    (_, index) => 'spoken' + index).join(' ')
  assert.deepEqual(splitDisplaySentences(text), [text])
  assert.deepEqual(splitDisplaySentences('This is a short caption'), [
    'This is a short caption',
  ])
})

test('long meeting sentence is not falsely split before successful people', () => {
  const text = "I've seen others who really didn't have that talent or that ability and they're among the most successful people today in the world because they never quit and they never gave up."
  assert.ok(text.length > 100)
  assert.deepEqual(splitDisplaySentences(text), [text])
})

test('CJK without punctuation stays intact while real punctuation splits sentences', () => {
  const original = '这是没有标点符号的长句'.repeat(12)
  assert.deepEqual(splitDisplaySentences(original), [original])
  assert.deepEqual(splitDisplaySentences('您好。你好吗？我很好！'), [
    '您好。', '你好吗？', '我很好！',
  ])
})

test('sentence breaks preserve titles, decimals, and source punctuation', () => {
  const text = 'Dr. Adams reported 3.14 seconds. The results looked good!'
  const lines = splitDisplaySentences(text)
  assert.deepEqual(lines, [
    'Dr. Adams reported 3.14 seconds.',
    'The results looked good!',
  ])
})

test('line breaks and empty input do not create phantom turns', () => {
  assert.deepEqual(splitDisplaySentences(' \n \n '), [])
  assert.deepEqual(splitDisplaySentences('First note\nSecond note'), [
    'First note', 'Second note',
  ])
})

test('very long unbroken Latin token is preserved instead of split mid-word', () => {
  const token = 'token'.repeat(50)
  assert.deepEqual(splitDisplaySentences(token), [token])
})
