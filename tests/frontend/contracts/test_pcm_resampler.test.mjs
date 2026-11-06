// Node tests for fractional-phase resampling in the browser AudioWorklet.
import test from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { runInNewContext } from "node:vm"

test("48 kHz to 16 kHz keeps correct sample count across 128-sample worklet blocks", () => {
  const path = join(process.cwd(), "src", "audio", "pcm-processor.ts")
  const file = readFileSync(path, "utf8")
  const prefix = "export const pcmProcessorCode = `"
  const start = file.indexOf(prefix)
  const end = file.lastIndexOf("`")
  assert.ok(start >= 0 && end > start)
  const program = file.slice(start + prefix.length, end)
    .replace("${PCM_PROCESSOR_NAME}", "mop-pcm-processor")
  const chunks = []
  const acknowledgements = []
  let Worklet = null
  class FakeWorkletProcessor {
    constructor() {
      this.port = { postMessage(message) {
        if (message && message.BYTES_PER_ELEMENT === 2) chunks.push(message)
        if (message && message.type === 'flushed') acknowledgements.push(message.type)
      }}
    }
  }
  runInNewContext(program, {
    AudioWorkletProcessor: FakeWorkletProcessor,
    Float32Array, Int16Array, Math,
    registerProcessor(_name, Klass) { Worklet = Klass },
  })
  assert.ok(Worklet)
  const instance = new Worklet({ processorOptions: { sampleRate: 48000, targetRate: 16000 } })
  for (let i = 0; i < 375; i++) {
    const input = Float32Array.from({ length: 128 }, (_, j) => {
      const phase = 2 * Math.PI * (128 * i + j) / 48000
      return 0.2 * Math.sin(1000 * phase) + 0.25 * Math.sin(12000 * phase)
    })
    instance.process([[input]])
  }
  const outputSamples = chunks.reduce((sum, frame) => sum + frame.length, 0)
  assert.equal(outputSamples, 14400)
  assert.equal(chunks.length, 9)
  assert.ok(chunks.every((frame) => frame.length === 1600))
  const middle = chunks[2]
  const amplitudeAt = frequency => {
    let cosine = 0
    let sine = 0
    for (let i = 0; i < middle.length; i++) {
      const phase = 2 * Math.PI * frequency * i / 16000
      cosine += middle[i] * Math.cos(phase)
      sine += middle[i] * Math.sin(phase)
    }
    return 2 * Math.hypot(cosine, sine) / middle.length / 32768
  }
  assert.ok(amplitudeAt(1000) > 0.19)
  assert.ok(amplitudeAt(4000) < 0.002)
  // One extra worklet quantum has insufficient audio for a full packet.
  instance.process([[new Float32Array(128)]])
  assert.equal(chunks.length, 10)
  instance.port.onmessage({ data: { type: 'flush' } })
  assert.deepEqual(acknowledgements, ['flushed'])
  assert.equal(chunks.at(-1).length, 43)
  assert.equal(chunks.reduce((sum, frame) => sum + frame.length, 0), 16043)
  instance.process([[new Float32Array(128)]])
  assert.equal(chunks.reduce((sum, frame) => sum + frame.length, 0), 16043)
})
