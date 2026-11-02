import type { ServerMessage, ServerReady, ServerTranscript } from '../types'

export interface RealtimeClientOptions {
  url: string
  onReady?: (msg: ServerReady) => void
  onTranscript?: (msg: ServerTranscript) => void
  onState?: (bufferedMs: number) => void
  onClosed?: (reason: 'closed' | 'disconnected') => void
}

/** 实时音频上传 + transcript 快照接收的 WebSocket 客户端。 */
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
      // 服务端收尾后会先发 closed JSON 再断开，避免重复回调
      if (!this.settled) this.options.onClosed?.('disconnected')
    }
    ws.onerror = () => ws.close()
    this.ws = ws
  }

  sendPcm(chunk: Int16Array): void {
    const ws = this.ws
    if (ws && ws.readyState === WebSocket.OPEN) {
      // Int16Array.buffer 的类型是 ArrayBufferLike，WebSocket 需要 ArrayBuffer
      ws.send(chunk.buffer as ArrayBuffer)
    }
  }

  /** 请求服务端收尾：把剩余音频处理完、推送最终 transcript 后由服务端关闭。 */
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
