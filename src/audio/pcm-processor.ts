export const PCM_PROCESSOR_NAME = 'mop-pcm-processor'

export const pcmProcessorCode = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor(processorOptions) {
    super()
    const opts =
      (processorOptions && processorOptions.processorOptions) ||
      (typeof globalThis !== 'undefined' && globalThis.options && globalThis.options.processorOptions) ||
      {}
    const sampleRate = opts.sampleRate || 48000
    const targetRate = opts.targetRate || 16000
    this.ratio = sampleRate / targetRate
    // Filter before decimation. Linear interpolation alone aliases energy
    // above the destination Nyquist frequency into the speech band.
    this.radius = this.ratio > 1 ? Math.ceil(12 * this.ratio) : 0
    this.kernel = new Float32Array(2 * this.radius + 1)
    if (this.radius) {
      const cutoff = 0.45 / this.ratio
      let sum = 0
      for (let i = -this.radius; i <= this.radius; i++) {
        const sinc = i === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * i) / (Math.PI * i)
        const window = 0.54 + 0.46 * Math.cos(Math.PI * i / this.radius)
        sum += this.kernel[i + this.radius] = sinc * window
      }
      for (let i = 0; i < this.kernel.length; i++) this.kernel[i] /= sum
    } else {
      this.kernel[0] = 1
    }
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
        this._emitPcm(this._resample(true))
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

  _filtered(index) {
    let value = 0
    for (let tap = -this.radius; tap <= this.radius; tap++) {
      const source = Math.max(0, Math.min(this.pending.length - 1, index + tap))
      value += this.pending[source] * this.kernel[tap + this.radius]
    }
    return value
  }

  _resample(final) {
    const out = []
    const limit = final ? this.pending.length : this.pending.length - this.radius - 1
    while (this.pos < limit) {
      const index = Math.floor(this.pos)
      const fraction = this.pos - index
      const first = this._filtered(index)
      out.push(fraction ? first * (1 - fraction) + this._filtered(index + 1) * fraction : first)
      this.pos += this.ratio
    }
    const keepFrom = Math.max(0, Math.min(this.pending.length, Math.floor(this.pos) - this.radius))
    this.pending = this.pending.slice(keepFrom)
    this.pos -= keepFrom
    return out
  }

  process(inputs) {
    if (this.flushed) return true
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel || channel.length === 0) return true

    this._dbgCount++
    if (this._dbgCount % 370 === 1) {
      let mx = 0
      for (let i = 0; i < channel.length; i++) {
        const a = Math.abs(channel[i])
        if (a > mx) mx = a
      }
      this.port.postMessage({ type: 'dbg', max: mx })
    }

    const pending = new Float32Array(this.pending.length + channel.length)
    pending.set(this.pending, 0)
    pending.set(channel, this.pending.length)
    this.pending = pending

    this._emitPcm(this._resample(false))
    return true
  }
}

registerProcessor('${PCM_PROCESSOR_NAME}', PCMProcessor)
`
