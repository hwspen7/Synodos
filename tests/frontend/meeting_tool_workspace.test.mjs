import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const source = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const css = readFileSync(path.join(root, 'src/assets/meeting-tool-layout.css'), 'utf8')
const start = source.indexOf('function selectMeetingTool(')
const end = source.indexOf('const selectedParticipantIds', start)
const toolHandler = source.slice(start, end)

test('meeting tools share exactly one active destination', () => {
  assert.match(toolHandler, /activeMeetingTool\.value = tool/)
  assert.match(toolHandler, /showReviewPanel\.value = tool === 'review'/)
  assert.match(toolHandler, /showRecordings\.value = tool === 'recordings'/)
  assert.match(toolHandler, /showAdvancedControls\.value = tool === 'settings'/)
  assert.match(source, /activeMeetingTool = ref<MeetingTool>\('transcript'\)/)
})

test('meeting tool controls live inside the local sidebar', () => {
  const nav = source.slice(source.indexOf('class="meeting-tool-sidebar"'), source.indexOf('</nav>', source.indexOf('class="meeting-tool-sidebar"')))
  for (const tool of ['transcript', 'review', 'recordings', 'settings']) {
    assert.ok(nav.includes("selectMeetingTool('" + tool + "')"), tool + ' navigation missing')
  }
  assert.match(source, /class="meeting-detail-panel"/)
  assert.doesNotMatch(source, /showReviewPanel = !showReviewPanel/)
  assert.doesNotMatch(source, /showRecordings = !showRecordings/)
  assert.doesNotMatch(source, /showAdvancedControls = !showAdvancedControls/)
})

test('review functions render as independent selectable modules', () => {
  for (const module of ['ai', 'correction', 'speakers', 'identity']) {
    assert.ok(source.includes("activeReviewTool === '" + module + "'"))
  }
  assert.match(source, /v-if="activeReviewTool === 'ai'"/)
  assert.match(source, /v-else-if="activeReviewTool === 'correction'"/)
  assert.match(source, /v-else-if="activeReviewTool === 'speakers'"/)
  assert.match(source, /v-else\s*\n\s*:meeting-id=/)
  assert.match(source, /activeReviewTool\.value = 'identity'/)
})

test('existing transcription and recording workflows are retained', () => {
  for (const name of [
    'buildTranscriptParagraphs','expandContentFirstSegments',
    'saveMeeting()','recording ? stop() : start()','loadRecordings()',
    'applySpeechSettings()','copyTranscript','editSegmentById',
    'openIdentityEvidence','Phase2IdentityReviewPanel',
  ]) assert.ok(source.includes(name), 'Missing workflow ' + name)
})

test('responsive tool CSS has no unbalanced blocks', () => {
  assert.equal((css.match(/\{/g) || []).length, (css.match(/\}/g) || []).length)
  for (const selector of ['meeting-tool-sidebar', 'meeting-detail-panel', 'meeting-review-tools']) {
    assert.ok(css.includes('.' + selector))
  }
  assert.match(css, /@media\(max-width:720px\)/)
})
