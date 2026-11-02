import type { ServerMessage, ServerReady, ServerTranscript } from '../types'

export interface RealtimeClientOptions {
  url: string
  onReady?: (msg: ServerReady) => void
  onTranscript?: (msg: ServerTranscript) => void
  onState?: (bufferedMs: number) => void
  onClosed?: (reason: 'closed' | 'disconnected', saved?: boolean) => void
  onError?: (message: string) => void
}

/** Own one browser WebSocket and suppress callbacks from superseded sockets. */
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
    this.ws = ws
    this.settled = false
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
        case 'transcript':
          this.options.onTranscript?.(msg)
          break
        case 'state':
          this.options.onState?.(msg.buffered_ms)
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

  sendPcm(chunk: Int16Array): void {
    const ws = this.ws
    if (ws?.readyState === WebSocket.OPEN) {
      // Send only the bytes in the provided view, not its entire backing buffer.
      const copy = new Uint8Array(chunk.byteLength)
      copy.set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength))
      ws.send(copy.buffer)
    }
  }

  sendStop(): void {
    const ws = this.ws
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
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
