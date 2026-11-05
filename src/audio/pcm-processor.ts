/**
  * Audio processing and UI rendering.
 *
  * Audio processing and UI rendering.
  * Audio processing and UI rendering.
 */
export const PCM_PROCESSOR_NAME = 'mop-pcm-processor'

export const pcmProcessorCode = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor(processorOptions) {
    super()
    // Process audio and meeting metadata.
    // Process audio and meeting metadata.
    const opts =
      (processorOptions && processorOptions.processorOptions) ||
      (typeof globalThis !== 'undefined' && globalThis.options && globalThis.options.processorOptions) ||
      {}
    const sampleRate = opts.sampleRate || 48000
    const targetRate = opts.targetRate || 16000
    this.ratio = sampleRate / targetRate
    this.pending = new Float32Array(0)
    this.pos = 0
    this._dbgCount = 0
    this.flushed = false
    // A 128-sample AudioWorklet quantum lasts only ~2.7ms at 48kHz.
    // Batch 16kHz PCM into 100ms transport frames to avoid hundreds of
    // WebSocket and diarization queue operations per second.
    this.packetSamples = Math.max(1, Math.round(targetRate / 10))
    this.pendingPcm = []
    this.port.onmessage = event => {
      if (event.data?.type === 'flush') {
        // Emit any final output sample that can be formed with a held
        // terminal input sample before acknowledging the stop boundary.
        const tail = []
        while (this.pos < this.pending.length) {
          const i0 = Math.floor(this.pos)
          const i1 = Math.min(i0 + 1, this.pending.length - 1)
          const frac = this.pos - i0
          tail.push(this.pending[i0] * (1 - frac) + this.pending[i1] * frac)
          this.pos += this.ratio
        }
        this._emitPcm(tail)
        // The stop acknowledgment must follow the final partial PCM frame.
        this._postPcm(this.pendingPcm)
        this.pendingPcm = []
        this.pending = new Float32Array(0)
        this.flushed = true
        this.port.postMessage({ type: 'flushed' })
      }
    }
  }

  _postPcm(out) {
    if (!out.length) return
    const pcm = new Int16Array(out.length)
    for (let i = 0; i < out.length; i++) {
      const sample = Math.max(-1, Math.min(1, out[i]))
      pcm[i] = sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff)
    }
    this.port.postMessage(pcm, [pcm.buffer])
  }

  _emitPcm(out) {
    for (const sample of out) {
      this.pendingPcm.push(sample)
      if (this.pendingPcm.length === this.packetSamples) {
        this._postPcm(this.pendingPcm)
        this.pendingPcm = []
      }
    }
  }

  process(inputs) {
    if (this.flushed) return true
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel || channel.length === 0) return true

    // Process audio and meeting metadata.
    this._dbgCount++
    if (this._dbgCount % 370 === 1) {
      let mx = 0
      for (let i = 0; i < channel.length; i++) {
        const a = Math.abs(channel[i])
        if (a > mx) mx = a
      }
      this.port.postMessage({ type: 'dbg', max: mx })
    }

    // Process audio and meeting metadata.
    const pending = new Float32Array(this.pending.length + channel.length)
    pending.set(this.pending, 0)
    pending.set(channel, this.pending.length)
    this.pending = pending

    // Process audio and meeting metadata.
    const ratio = this.ratio
    const out = []
    while (this.pos + 1 < this.pending.length) {
      const i0 = Math.floor(this.pos)
      const frac = this.pos - i0
      const s = this.pending[i0] * (1 - frac) + this.pending[i0 + 1] * frac
      out.push(s)
      this.pos += ratio
    }

    // Process audio and meeting metadata.
    // Retain the fractional resampling phase across audio worklet blocks.
    const keepFrom = Math.min(this.pending.length, Math.floor(this.pos))
    this.pending = this.pending.slice(keepFrom)
    this.pos -= keepFrom

    this._emitPcm(out)
    return true
  }
}

registerProcessor('${PCM_PROCESSOR_NAME}', PCMProcessor)
`
