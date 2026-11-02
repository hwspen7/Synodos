import type { ServerMessage, ServerReady, ServerTranscript } from '../types'

export interface RealtimeClientOptions {
  url: string
  onReady?: (msg: ServerReady) => void
  onTranscript?: (msg: ServerTranscript) => void
  onState?: (bufferedMs: number) => void
  onClosed?: (reason: 'closed' | 'disconnected') => void
}

/* Audio processing and UI rendering. */
export class RealtimeClient {
  private ws: WebSocket | null = null
  private settled = false
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
    ws.binaryType = 'arraybuffer'
    this.settled = false
    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== 'string') return
      let msg: ServerMessage
      try {
        msg = JSON.parse(event.data) as ServerMessage
      } catch {
        return
      }
      switch (msg.type) {
        case 'ready':
          this.options.onReady?.(msg)
          break
        case 'transcript':
          this.options.onTranscript?.(msg)
          break
        case 'state':
          this.options.onState?.(msg.buffered_ms)
          break
        case 'closed':
          this.settled = true
          this.options.onClosed?.('closed')
          break
      }
    }
    ws.onclose = () => {
      // Process audio and meeting metadata.
      if (!this.settled) this.options.onClosed?.('disconnected')
    }
    ws.onerror = () => ws.close()
    this.ws = ws
  }

  sendPcm(chunk: Int16Array): void {
    const ws = this.ws
    if (ws && ws.readyState === WebSocket.OPEN) {
      // Process audio and meeting metadata.
      ws.send(chunk.buffer as ArrayBuffer)
    }
  }

  /* Audio processing and UI rendering. */
  sendStop(): void {
    const ws = this.ws
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
    }
  }

  close(): void {
    const ws = this.ws
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onclose = null
      ws.onerror = null
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
    }
    this.ws = null
  }
}
