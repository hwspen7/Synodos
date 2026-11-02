import { PCM_PROCESSOR_NAME, pcmProcessorCode } from './pcm-processor'

export interface MicRecorder {
  stop: () => void
}

/**
 * 打开麦克风，用 AudioWorklet 实时降采样为 16kHz 单声道 Int16 PCM，
 * 每个处理块通过 onPcm 回调交给调用方（通常是 WebSocket）。
 * onLevel 可选，回调返回当前音频 RMS 电平（0~1），用于麦克风状态指示。
 */
export async function startMicRecording(
  onPcm: (chunk: Int16Array) => void,
  targetRate = 16000,
  onLevel?: (level: number) => void,
): Promise<MicRecorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  })

  let ctx: AudioContext | null = null
  try {
    const AudioCtx: typeof AudioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    ctx = new AudioCtx()
    const source = ctx.createMediaStreamSource(stream)

    const blob = new Blob([pcmProcessorCode], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    await ctx.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)

    const node = new AudioWorkletNode(ctx, PCM_PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { sampleRate: ctx.sampleRate, targetRate },
    })
    node.port.onmessage = (event: MessageEvent) => {
      const data = event.data
      // 调试消息：AudioWorklet 输入信号最大值
      if (data && typeof data === 'object' && data.type === 'dbg') {
        console.debug(`[mic-input] max=${(data.max as number).toFixed(5)}`)
        return
      }
      const pcm = data as Int16Array
      if (!pcm || pcm.length === 0) return
      // 先计算电平，再发送，避免部分浏览器 ws.send 分离 ArrayBuffer 导致数据失效
      if (onLevel) {
        let sum = 0
        let maxAbs = 0
        for (let i = 0; i < pcm.length; i++) {
          const v = (pcm[i] ?? 0) / 32768
          sum += v * v
          const a = Math.abs(v)
          if (a > maxAbs) maxAbs = a
        }
        const level = Math.sqrt(sum / pcm.length)
        onLevel(level)
        // 调试：每秒输出一次采样最大值，帮助确认麦克风是否真正采集到声音
        if (Math.random() < 0.02) {
          console.debug(`[mic] rms=${level.toFixed(5)} max=${maxAbs.toFixed(5)}`)
        }
      }
      onPcm(pcm)
    }

    source.connect(node)
    await ctx.resume()

    let stopped = false
    return {
      stop() {
        if (stopped) return
        stopped = true
        node.port.onmessage = null
        node.disconnect()
        source.disconnect()
        stream.getTracks().forEach((track) => track.stop())
        void ctx?.close()
      },
    }
  } catch (err) {
    // 初始化失败时释放麦克风，避免权限一直被占用
    stream.getTracks().forEach((track) => track.stop())
    void ctx?.close()
    throw err
  }
}
