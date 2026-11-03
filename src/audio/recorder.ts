import { PCM_PROCESSOR_NAME, pcmProcessorCode } from './pcm-processor'

export interface MicRecorder {
  stop: () => Promise<void>
}

/**
  * Audio processing and UI rendering.
  * Audio processing and UI rendering.
  * Audio processing and UI rendering.
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
    let resolveFlushed: (() => void) | null = null
    node.port.onmessage = (event: MessageEvent) => {
      const data = event.data
      if (data && typeof data === 'object' && data.type === 'flushed') {
        resolveFlushed?.()
        return
      }
      // Process audio and meeting metadata.
      if (data && typeof data === 'object' && data.type === 'dbg') {
        console.debug(`[mic-input] max=${(data.max as number).toFixed(5)}`)
        return
      }
      const pcm = data as Int16Array
      if (!pcm || pcm.length === 0) return
      // Process audio and meeting metadata.
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
        // Process audio and meeting metadata.
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
      async stop() {
        if (stopped) return
        stopped = true
        // Flush the worklet before closing the browser audio graph.
        try {
          await Promise.race([
            new Promise<void>((resolve) => {
              resolveFlushed = resolve
              node.port.postMessage({ type: 'flush' })
            }),
            new Promise<void>((_, reject) => {
              window.setTimeout(() => reject(new Error('Microphone drain was not acknowledged')), 3000)
            }),
          ])
        } finally {
          node.port.onmessage = null
          node.disconnect()
          source.disconnect()
          stream.getTracks().forEach((track) => track.stop())
          await ctx?.close()
        }
      },
    }
  } catch (err) {
    // Process audio and meeting metadata.
    stream.getTracks().forEach((track) => track.stop())
    void ctx?.close()
    throw err
  }
}
