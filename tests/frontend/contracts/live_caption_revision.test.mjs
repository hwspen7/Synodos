import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { RealtimeClient } from '../../../src/api/ws.ts'

class FakeWebSocket {
  static OPEN = 1
  static CONNECTING = 0
  static current = null

  constructor(url) {
    this.url = url
    this.readyState = FakeWebSocket.OPEN
    FakeWebSocket.current = this
  }

  close() {
    this.readyState = 3
  }

  send() {}

  emit(message) {
    this.onmessage?.({ data: JSON.stringify(message) })
  }
}

test('live caption WebSocket preserves typed partial revision and transcript order', () => {
  const original = globalThis.WebSocket
  globalThis.WebSocket = FakeWebSocket
  try {
    const events = []
    const client = new RealtimeClient({
      url: 'ws://localhost/for-test',
      onPartial: (text, language, revision) =>
        events.push({ type: 'partial', text, language, revision }),
      onTranscript: msg =>
        events.push({ type: 'transcript', version: msg.version }),
    })
    client.connect()
    const ws = FakeWebSocket.current
    ws.emit({ type: 'partial', text: 'A few words', language: 'en',
              revision: 1 })
    ws.emit({ type: 'transcript', version: 1, segments: [] })
    ws.emit({ type: 'partial', text: '', language: 'und', revision: 2 })
    assert.deepEqual(events, [
      { type: 'partial', text: 'A few words', language: 'en', revision: 1 },
      { type: 'transcript', version: 1 },
      { type: 'partial', text: '', language: 'und', revision: 2 },
    ])
    client.close()
  } finally {
    globalThis.WebSocket = original
  }
})

test('live provisional text stays inside transcript and snapshots cannot clear it', () => {
  const source = readFileSync(
    new URL('../../../src/pages/MeetingPage.vue', import.meta.url), 'utf8')
  const handler = source.match(/onTranscript: \(msg\) => \{([\s\S]*?)\n    \},\n    onState:/)
  assert.ok(handler, 'Transcript callback must remain explicit')
  assert.doesNotMatch(handler[1], /partialText\.value\s*=\s*''/)
  assert.match(source, /revision <= lastPartialRevision/)
  assert.match(source, /lastPartialRevision = revision/)
  assert.match(source, /Live caption.*not confirmed/)
  const mainStart = source.indexOf('<main ref="transcriptEl"')
  const draft = source.indexOf('class="paragraph transcript-live-preview"')
  const mainEnd = source.indexOf('</main>', mainStart)
  assert.ok(mainStart !== -1 && mainStart < draft && draft < mainEnd)
})
