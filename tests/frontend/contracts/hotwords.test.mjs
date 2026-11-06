import test from 'node:test'
import assert from 'node:assert/strict'
import { parseHotwords } from '../../../src/config/hotwords.ts'

test('vocabulary entry preserves spaces inside terms and explicit separators', () => {
  assert.deepEqual(parseHotwords('Spring  Boot,Java\nSpring Boot;Visual Studio Code'),
    ['Spring Boot', 'Java', 'Visual Studio Code'])
})
