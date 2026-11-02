/**
 * AudioWorklet 处理器源码（以 Blob 方式注入）。
 *
 * 负责把浏览器输入的任意采样率音频降采样为 16kHz 单声道，
 * 再量化为 Int16 PCM 发回主线程（由主线程通过 WebSocket 上传）。
 */
export const PCM_PROCESSOR_NAME = 'mop-pcm-processor'

export const pcmProcessorCode = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor(processorOptions) {
    super()
    // processorOptions 按标准作为构造参数传入；个别实现把它挂在全局 options 上，
    // 这里两种方式都兼容（用 typeof/globalThis 避免未定义时报错）
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
  }

  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel || channel.length === 0) return true

    // 调试：约每秒输出一次输入信号最大值，确认麦克风是否真正有声音输入
    this._dbgCount++
    if (this._dbgCount % 370 === 1) {
      let mx = 0
      for (let i = 0; i < channel.length; i++) {
        const a = Math.abs(channel[i])
        if (a > mx) mx = a
      }
      this.port.postMessage({ type: 'dbg', max: mx })
    }

    // 追加到待处理缓冲
    const pending = new Float32Array(this.pending.length + channel.length)
    pending.set(this.pending, 0)
    pending.set(channel, this.pending.length)
    this.pending = pending

    // 线性插值降采样
    const ratio = this.ratio
    const out = []
    while (this.pos + 1 < this.pending.length) {
      const i0 = Math.floor(this.pos)
      const frac = this.pos - i0
      const s = this.pending[i0] * (1 - frac) + this.pending[i0 + 1] * frac
      out.push(s)
      this.pos += ratio
    }

    // 保留尾部未消费的样本
    const keepFrom = Math.floor(this.pos)
    this.pending = this.pending.slice(keepFrom)
    this.pos -= keepFrom

    if (out.length > 0) {
      const pcm = new Int16Array(out.length)
      for (let i = 0; i < out.length; i++) {
        const s = Math.max(-1, Math.min(1, out[i]))
        pcm[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff)
      }
      this.port.postMessage(pcm, [pcm.buffer])
    }
    return true
  }
}

registerProcessor('${PCM_PROCESSOR_NAME}', PCMProcessor)
`
