import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import ts from 'typescript'

const filename = join(process.cwd(), 'src', 'audio', 'recorder.ts')
const code = ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function harness({ deliverPcm = true } = {}) {
  const state = {
    sourceConnected: false,
    workletConnected: false,
    outputConnected: false,
    muted: false,
    tracksStopped: false,
    contextClosed: false,
    outputCount: 0,
  }
  let worklet

  const track = {
    stop() { state.tracksStopped = true },
    addEventListener() {},
  }
  const stream = {
    getTracks() { return [track] },
    getAudioTracks() { return [track] },
  }
  const navigator = { mediaDevices: { getUserMedia: async () => stream } }
  class FakeCtx {
    constructor() {
      this.sampleRate = 48000
      this.destination = { id: 'destination' }
      this.state = 'suspended'
      this.audioWorklet = { addModule: async () => {} }
    }
    createMediaStreamSource() {
      return {
        connect(node) {
          assert.equal(node, worklet)
          state.sourceConnected = true
        },
        disconnect() {},
      }
    }
    createGain() {
      const sink = {
        gain: { value: 1 },
        connect(destination) {
          assert.equal(destination.id, 'destination')
          state.outputConnected = true
          state.muted = sink.gain.value === 0
        },
        disconnect() {},
      }
      return sink
    }
    async resume() {
      this.state = 'running'
      if (deliverPcm && state.sourceConnected && state.workletConnected &&
        state.outputConnected && state.muted && state.outputCount === 1) {
        queueMicrotask(() => worklet.port.onmessage?.({ data: new Int16Array([16384, -16384]) }))
      }
    }
    async close() {
      this.state = 'closed'
      state.contextClosed = true
    }
  }
  class FakeWorkletNode {
    constructor(_ctx, _name, options) {
      state.outputCount = options.numberOfOutputs
      this.port = {
        onmessage: null,
        postMessage(message) {
          if (message.type === 'flush') {
            queueMicrotask(() => worklet.port.onmessage?.({ data: { type: 'flushed' } }))
          }
        },
      }
      worklet = this
    }
    connect(sink) {
      assert.equal(sink.gain.value, 0)
      state.workletConnected = true
    }
    disconnect() {}
  }
  const window = {
    AudioContext: FakeCtx,
    setTimeout: deliverPcm ? setTimeout : (fn) => setTimeout(fn, 0),
    clearTimeout,
  }
  const URL = { createObjectURL: () => 'blob:test', revokeObjectURL() {} }
  const module = { exports: {} }
  const require = (name) => {
    assert.equal(name, './pcm-processor')
    return { PCM_PROCESSOR_NAME: 'mop-pcm-processor', pcmProcessorCode: 'registerProcessor()' }
  }
  new Function(
    'module', 'exports', 'require', 'window', 'navigator', 'AudioWorkletNode', 'Blob', 'URL',
    code,
  )(module, module.exports, require, window, navigator, FakeWorkletNode, Blob, URL)
  return { startMicRecording: module.exports.startMicRecording, state }
}

test('microphone PCM is delivered only from a connected muted audio graph', async () => {
  const { startMicRecording, state } = harness()
  const received = []
  const recorder = await startMicRecording((pcm) => received.push([...pcm]), 16000)
  assert.deepEqual(received, [[16384, -16384]])
  assert.equal(state.outputCount, 1)
  assert.equal(state.sourceConnected, true)
  assert.equal(state.workletConnected, true)
  assert.equal(state.outputConnected, true)
  assert.equal(state.muted, true)
  await recorder.stop()
  assert.equal(state.tracksStopped, true)
  assert.equal(state.contextClosed, true)
})

test('missing PCM fails startup and releases the microphone', async () => {
  const { startMicRecording, state } = harness({ deliverPcm: false })
  await assert.rejects(
    startMicRecording(() => {}),
    /No microphone PCM reached the audio processor/,
  )
  assert.equal(state.tracksStopped, true)
  assert.equal(state.contextClosed, true)
})
