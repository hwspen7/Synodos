<script setup lang="ts">
import { computed } from 'vue'
import type { Segment } from '@/types'

const props = defineProps<{
  segments: Segment[]
  bufferedMs: number
  recording: boolean
  asrEngine: string
}>()

const backlog = computed(() => Math.max(0, props.bufferedMs) / 1000)
const backlogStatus = computed(() => backlog.value >= 20 ? 'high' : backlog.value >= 10 ? 'watch' : 'ok')
const languages = computed(() => [...new Set(props.segments.map(s => s.language).filter(
  (value): value is string => Boolean(value && value !== 'und'),
))].sort())
const unknownSpeakers = computed(() => props.segments.filter(s => s.speaker_id <= 0).length)
const changes = computed(() => {
  let last = 0
  let switches = 0
  for (const segment of props.segments) {
    if (segment.speaker_id > 0) {
      if (last > 0 && segment.speaker_id !== last) switches += 1
      last = segment.speaker_id
    }
  }
  return switches
})
</script>

<template>
  <section class="phase07-status" aria-label="Recording quality diagnostics">
    <span class="phase07-tag">PHASE 07</span>
    <span>ASR {{ asrEngine || 'unknown' }}</span>
    <span :class="['phase07-backlog', `is-${backlogStatus}`]" role="status">
      Audio backlog {{ backlog.toFixed(1) }}s
    </span>
    <span>Languages {{ languages.length ? languages.join(', ') : 'pending' }}</span>
    <span>Speaker changes {{ changes }}</span>
    <span v-if="unknownSpeakers">Unidentified {{ unknownSpeakers }}</span>
    <span v-if="recording && backlogStatus === 'high'" class="phase07-caution">
      Processing is behind the microphone stream
    </span>
  </section>
</template>

<style scoped>
.phase07-status { display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; padding:.55rem .85rem;
  margin:.5rem 0 1rem; border:1px solid var(--border, #ccd1d9); border-radius:12px;
  background:var(--surface, transparent); font-size:.78rem; opacity:.94; }
.phase07-tag { font-weight:700; letter-spacing:.08em; }
.phase07-backlog.is-watch { font-weight:600; }
.phase07-backlog.is-high,.phase07-caution { font-weight:700; color:var(--danger, #b45309); }
</style>
