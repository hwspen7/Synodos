import type { ServerMessage, ServerReady, ServerTranscript } from '../types'

export interface RealtimeClientOptions {
  url: string
  onReady?: (msg: ServerReady) => void
  onTranscript?: (msg: ServerTranscript) => void
  onPartial?: (text: string, language: string, revision: number) => void
  onFeatureWarning?: (message: string) => void
  onAudioAnalysis?: (result: { emotion?: string | null, events?: string[], overlap?: Array<{start_ms: number, end_ms: number}>, quality?: { rms: number } }) => void
  onState?: (bufferedMs: number) => void
  onAudioReceived?: (samples: number, matched: boolean, durableSamples?: number | null) => void
  onAudioProgress?: (samples: number, durableSamples?: number | null) => void
  onClosed?: (reason: 'closed' | 'disconnected', saved?: boolean) => void
  onError?: (message: string) => void
}

/** Own one browser WebSocket and suppress callbacks from superseded sockets. */
export class RealtimeClient {
  private ws: WebSocket | null = null
  private settled = false
  private sentSamples = 0
  private frameSequence = 0
  private acknowledgedSamples = 0
  private audioCongested = false
  private readonly options: RealtimeClientOptions

  constructor(options: RealtimeClientOptions) {
    this.options = options
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  connect(): void {
    this.close()
    const ws = new WebSocket(this.options.url)
    this.ws = ws
    this.settled = false
    this.sentSamples = 0
    this.frameSequence = 0
    this.acknowledgedSamples = 0
    this.audioCongested = false
    ws.binaryType = 'arraybuffer'

    ws.onmessage = (event: MessageEvent) => {
      if (this.ws !== ws || typeof event.data !== 'string') return
      let msg: ServerMessage
      try {
        msg = JSON.parse(event.data) as ServerMessage
      } catch {
        this.options.onError?.('Received invalid server message')
        return
      }

      switch (msg.type) {
        case 'ready':
          this.options.onReady?.(msg)
          break
        case 'partial':
          this.options.onPartial?.(msg.text, msg.language, msg.revision)
          break
        case 'feature_warning':
          this.options.onFeatureWarning?.(msg.message)
          break
        case 'transcript':
          this.options.onTranscript?.(msg)
          break
        case 'state':
          this.options.onState?.(msg.buffered_ms)
          break
        case 'audio_progress':
          if (msg.samples > this.sentSamples || msg.samples < this.acknowledgedSamples) {
            this.options.onError?.('Server audio acknowledgement is inconsistent.')
            ws.close(1011, 'Invalid audio acknowledgement')
            break
          }
          this.acknowledgedSamples = msg.samples
          this.options.onAudioProgress?.(msg.samples, msg.durable_samples)
          break
        case 'audio_received':
          this.options.onAudioReceived?.(msg.samples, msg.matched, msg.durable_samples)
          break
        case 'error':
          this.options.onError?.(msg.message || 'Server processing error')
          break
        case 'closed':
          this.settled = true
          this.options.onClosed?.('closed', msg.saved)
          ws.close()
          break
      }
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.ws = null
      if (!this.settled) this.options.onClosed?.('disconnected')
    }
    ws.onerror = () => {
      if (this.ws !== ws) return
      this.options.onError?.('WebSocket network error')
      ws.close()
    }
  }

  sendPcm(chunk: Int16Array): boolean {
    const ws = this.ws
    if (ws?.readyState !== WebSocket.OPEN || this.audioCongested) return false
    // Fail closed rather than silently pretending a chunk has reached the server.
    if (ws.bufferedAmount + chunk.byteLength + 16 > 1024 * 1024) {
      this.audioCongested = true
      this.options.onError?.('Audio transport congested. Recording stopped to prevent silent data loss.')
      ws.close(1011, 'Audio transport backpressure')
      return false
    }
    const packet = new ArrayBuffer(chunk.byteLength + 16)
    const view = new DataView(packet)
    view.setUint8(0, 0x4d)
    view.setUint8(1, 0x4f)
    view.setUint8(2, 0x50)
    view.setUint8(3, 0x41)
    view.setUint32(4, this.frameSequence, false)
    view.setBigUint64(8, BigInt(this.sentSamples), false)
    new Uint8Array(packet, 16).set(
      new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength),
    )
    try {
      ws.send(packet)
    } catch {
      this.audioCongested = true
      this.options.onError?.('Audio transport failed while sending microphone samples.')
      ws.close(1011, 'Audio send failure')
      return false
    }
    this.frameSequence += 1
    this.sentSamples += chunk.length
    return true
  }

  sendStop(): void {
    const ws = this.ws
    if (ws?.readyState === WebSocket.OPEN && !this.audioCongested) {
      ws.send(JSON.stringify({ type: 'stop', expected_samples: this.sentSamples }))
    }
  }

  close(): void {
    const ws = this.ws
    this.ws = null
    this.settled = true
    if (!ws) return
    ws.onopen = null
    ws.onmessage = null
    ws.onclose = null
    ws.onerror = null
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close()
    }
  }
}
