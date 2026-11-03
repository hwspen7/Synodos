import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

const sourcePath = path.join(process.cwd(), 'src', 'api', 'ws.ts')
const source = fs.readFileSync(sourcePath, 'utf8')
const transpiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function createHarness() {
  const sockets = []
  class FakeSocket {
    static CONNECTING = 0
    static OPEN = 1
    static CLOSED = 3

    constructor(url) {
      this.url = url
      this.readyState = FakeSocket.OPEN
      this.sent = []
      sockets.push(this)
    }
    send(bytes) { this.sent.push(bytes) }
    close() {
      this.readyState = FakeSocket.CLOSED
      this.onclose?.()
    }
    message(message) { this.onmessage?.({ data: JSON.stringify(message) }) }
  }
  const module = { exports: {} }
  new Function('module', 'exports', 'WebSocket', transpiled)(module, module.exports, FakeSocket)
  return { RealtimeClient: module.exports.RealtimeClient, sockets }
}

test('server error and final persistence failure are observable', () => {
  const { RealtimeClient, sockets } = createHarness()
  const errors = []
  const closed = []
  const client = new RealtimeClient({
    url: 'ws://localhost/ws/audio',
    onError: message => errors.push(message),
    onClosed: (reason, saved) => closed.push([reason, saved]),
  })
  client.connect()
  sockets[0].message({ type: 'error', message: 'Disk full' })
  sockets[0].message({ type: 'closed', saved: false })
  sockets[0].close()
  assert.deepEqual(errors, ['Disk full'])
  assert.deepEqual(closed, [['closed', false]])
})

test('reconnect closes old sockets and ignores stale messages', () => {
  const { RealtimeClient, sockets } = createHarness()
  const messages = []
  const client = new RealtimeClient({
    url: 'ws://localhost/ws/audio',
    onError: message => messages.push(message),
  })
  client.connect()
  const old = sockets[0]
  client.connect()
  assert.equal(old.readyState, 3)
  old.message({ type: 'error', message: 'stale' })
  assert.deepEqual(messages, [])
})

test('sendPcm respects typed array byte offsets', () => {
  const { RealtimeClient, sockets } = createHarness()
  const client = new RealtimeClient({ url: 'ws://localhost/ws/audio' })
  client.connect()
  const samples = new Int16Array([1, 2, 3, 4])
  client.sendPcm(samples.subarray(1, 3))
  const frame = sockets[0].sent[0]
  assert.equal(frame.byteLength, 20)
  const view = new DataView(frame)
  assert.equal(String.fromCharCode(...new Uint8Array(frame, 0, 4)), 'MOPA')
  assert.equal(view.getUint32(4, false), 0)
  assert.equal(view.getBigUint64(8, false), 0n)
  assert.deepEqual([...new Int16Array(frame.slice(16))], [2, 3])
})


test('stop sends the exact PCM sample count and surfaces audio receipt', () => {
  const { RealtimeClient, sockets } = createHarness()
  const receipts = []
  const client = new RealtimeClient({
    url: 'ws://localhost/ws/audio',
    onAudioReceived: (samples, matched) => receipts.push([samples, matched]),
  })
  client.connect()
  client.sendPcm(new Int16Array([1, 2, 3, 4]))
  client.sendPcm(new Int16Array([5, 6]))
  client.sendStop()
  assert.equal(JSON.parse(sockets[0].sent[2]).expected_samples, 6)
  sockets[0].message({ type: 'audio_received', samples: 6, matched: true })
  assert.deepEqual(receipts, [[6, true]])
})

test('a broken or congested stream cannot request a successful stop', () => {
  const { RealtimeClient, sockets } = createHarness()
  const errors = []
  const client = new RealtimeClient({
    url: 'ws://localhost/ws/audio',
    onError: message => errors.push(message),
  })
  client.connect()
  sockets[0].bufferedAmount = 1024 * 1024
  client.sendPcm(new Int16Array([1]))
  assert.equal(sockets[0].readyState, 3)
  client.sendStop()
  assert.equal(sockets[0].sent.length, 0)
  assert.ok(errors.some(text => text.includes('congested')))
})


test('versioned frames use contiguous offsets and monotonic sequence IDs', () => {
  const { RealtimeClient, sockets } = createHarness()
  const client = new RealtimeClient({ url: 'ws://localhost/ws/audio' })
  client.connect()
  assert.equal(client.sendPcm(new Int16Array([1, 2])), true)
  assert.equal(client.sendPcm(new Int16Array([3])), true)
  assert.equal(new DataView(sockets[0].sent[1]).getUint32(4, false), 1)
  assert.equal(new DataView(sockets[0].sent[1]).getBigUint64(8, false), 2n)
})

test('invalid audio acknowledgement is treated as a terminal transport failure', () => {
  const { RealtimeClient, sockets } = createHarness()
  const errors = []
  const client = new RealtimeClient({
    url: 'ws://localhost/ws/audio', onError: err => errors.push(err),
  })
  client.connect()
  client.sendPcm(new Int16Array([1, 2]))
  sockets[0].message({ type: 'audio_progress', samples: 999 })
  assert.equal(sockets[0].readyState, 3)
  assert.ok(errors.some(e => e.includes('inconsistent')))
})
