import type {
  HealthInfo,
  HotwordLibrary,
  MeetingDetail,
  MeetingSegment,
  MeetingSummary,
  Speaker,
  SpeakerVector,
} from '../types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  // Process audio and meeting metadata.
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* keep status text */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const meetingsApi = {
  list: () => request<{ meetings: MeetingSummary[] }>('/meetings'),
  get: (id: string) => request<{ meeting: MeetingDetail }>(`/meetings/${id}`),
  segments: (id: string) =>
    request<{ segments: MeetingSegment[] }>(`/meetings/${id}/segments`),
  create: (title: string) =>
    request<{ meeting: MeetingSummary }>('/meetings', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),
  save: (id: string) =>
    request<{ ok: boolean; meeting: MeetingSummary }>(`/meetings/${id}/save`, {
      method: 'POST',
    }),
  remove: (id: string) =>
    request<{ ok: boolean }>(`/meetings/${id}`, { method: 'DELETE' }),
  exportUrl: (id: string, format: 'md' | 'json') =>
    `${BASE}/meetings/${id}/export?format=${format}`,
}

export const speakersApi = {
  list: () => request<{ speakers: Speaker[] }>('/speakers'),
  create: (name: string) =>
    request<{ speaker: Speaker }>('/speakers', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  rename: (id: string, name: string) =>
    request<{ speaker: Speaker }>(`/speakers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    }),
  remove: (id: string) =>
    request<{ ok: boolean }>(`/speakers/${id}`, { method: 'DELETE' }),
  /* Audio processing and UI rendering. */
  addRecord: (id: string, blob: Blob) =>
    request<{ vectors: SpeakerVector[]; min_audio_s: number }>(
      `/speakers/${id}/vectors/record`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
      },
    ),
  /* Audio processing and UI rendering. */
  addFromMeeting: (id: string, meetingId: string, segmentIds: number[]) =>
    request<{ vectors: SpeakerVector[]; missing: number[]; min_audio_s: number }>(
      `/speakers/${id}/vectors/meeting`,
      {
        method: 'POST',
        body: JSON.stringify({ meeting_id: meetingId, segment_ids: segmentIds }),
      },
    ),
  deleteVector: (id: string, vectorId: string) =>
    request<{ ok: boolean }>(`/speakers/${id}/vectors/${vectorId}`, {
      method: 'DELETE',
    }),
}

export const hotwordsApi = {
  list: () => request<{ hotwords: HotwordLibrary[] }>('/hotwords'),
  create: (name: string, words: string[]) =>
    request<{ hotword: HotwordLibrary }>('/hotwords', {
      method: 'POST',
      body: JSON.stringify({ name, words }),
    }),
  update: (id: string, payload: { name?: string; words?: string[] }) =>
    request<{ hotword: HotwordLibrary }>(`/hotwords/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  remove: (id: string) =>
    request<{ ok: boolean }>(`/hotwords/${id}`, { method: 'DELETE' }),
}

export const systemApi = {
  health: () => request<HealthInfo>('/health'),
}

/* Audio processing and UI rendering. */
export function downloadUrl(url: string): void {
  const a = document.createElement('a')
  a.href = url
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
}
