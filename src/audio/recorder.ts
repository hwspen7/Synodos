import { PCM_PROCESSOR_NAME, pcmProcessorCode } from './pcm-processor'

export interface MicRecorder {
  stop: () => Promise<void>
}

/**
 * Capture microphone PCM through a connected, muted Web Audio graph.
 * A zero-output worklet can be pruned by the browser and never process audio.
 * Do not report recording as started before the first PCM packet arrives.
 */
export async function startMicRecording(
  onPcm: (chunk: Int16Array) => void,
  targetRate = 16000,
  onLevel?: (level: number) => void,
  onDeviceEnded?: () => void,
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
  let source: MediaStreamAudioSourceNode | null = null
  let node: AudioWorkletNode | null = null
  let mutedOutput: GainNode | null = null
  let readyTimeout: ReturnType<typeof setTimeout> | null = null
  let firstPcmResolve: (() => void) | null = null
  let stopped = false
  let stopPromise: Promise<void> | null = null

  const firstPcm = new Promise<void>((resolve) => {
    firstPcmResolve = resolve
  })

  async function release(): Promise<void> {
    if (readyTimeout !== null) {
      window.clearTimeout(readyTimeout)
      readyTimeout = null
    }
    if (node) {
      node.port.onmessage = null
      node.disconnect()
    }
    source?.disconnect()
    mutedOutput?.disconnect()
    stream.getTracks().forEach((track) => track.stop())
    if (ctx && ctx.state !== 'closed') await ctx.close()
  }

  try {
    const AudioCtx: typeof AudioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    if (!AudioCtx) throw new Error('This browser does not support microphone audio processing')
    ctx = new AudioCtx()
    source = ctx.createMediaStreamSource(stream)

    const blob = new Blob([pcmProcessorCode], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    try {
      await ctx.audioWorklet.addModule(url)
    } finally {
      URL.revokeObjectURL(url)
    }

    node = new AudioWorkletNode(ctx, PCM_PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { sampleRate: ctx.sampleRate, targetRate },
    })

    let resolveFlushed: (() => void) | null = null
    node.port.onmessage = (event: MessageEvent) => {
      const data = event.data
      if (data && typeof data === 'object' && data.type === 'flushed') {
        resolveFlushed?.()
        return
      }
      if (!(data instanceof Int16Array) || data.length === 0) return
      firstPcmResolve?.()
      firstPcmResolve = null

      if (onLevel) {
        let sum = 0
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] ?? 0) / 32768
          sum += v * v
        }
        onLevel(Math.sqrt(sum / data.length))
      }
      onPcm(data)
    }

    // Keep the audio-processing graph connected to the destination. The
    // worklet intentionally never writes to its output and the gain is zero,
    // so the microphone audio cannot be played back through the speakers.
    mutedOutput = ctx.createGain()
    mutedOutput.gain.value = 0
    source.connect(node)
    node.connect(mutedOutput)
    mutedOutput.connect(ctx.destination)

    for (const track of stream.getAudioTracks()) {
      track.addEventListener('ended', () => {
        if (!stopped) onDeviceEnded?.()
      }, { once: true })
    }

    await ctx.resume()
    await Promise.race([
      firstPcm,
      new Promise<void>((_, reject) => {
        readyTimeout = window.setTimeout(
          () => reject(new Error('No microphone PCM reached the audio processor. Check the input device and browser audio permissions.')),
          3000,
        )
      }),
    ])
    if (readyTimeout !== null) {
      window.clearTimeout(readyTimeout)
      readyTimeout = null
    }

    return {
      stop() {
        if (stopPromise) return stopPromise
        stopped = true
        stopPromise = (async () => {
          try {
            await Promise.race([
              new Promise<void>((resolve) => {
                resolveFlushed = resolve
                node!.port.postMessage({ type: 'flush' })
              }),
              new Promise<void>((_, reject) => {
                window.setTimeout(
                  () => reject(new Error('Microphone drain was not acknowledged')),
                  3000,
                )
              }),
            ])
          } finally {
            await release()
          }
        })()
        return stopPromise
      },
    }
  } catch (err) {
    stopped = true
    await release()
    throw err
  }
}
