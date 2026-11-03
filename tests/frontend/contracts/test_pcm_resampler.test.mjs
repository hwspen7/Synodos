"""Node tests for fractional-phase resampling in the browser AudioWorklet."""
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
  let Worklet = null
  class FakeWorkletProcessor {
    constructor() {
      this.port = { postMessage(message) {
        if (message && message.BYTES_PER_ELEMENT === 2) chunks.push(message)
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
    instance.process([[new Float32Array(128)]])
  }
  const outputSamples = chunks.reduce((sum, frame) => sum + frame.length, 0)
  assert.equal(outputSamples, 16000)
})
