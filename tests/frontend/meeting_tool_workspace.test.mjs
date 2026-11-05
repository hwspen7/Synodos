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
  assert.match(source, /activeMeetingTool = ref<MeetingTool>\('transcript'\)/)
  assert.doesNotMatch(source, /const showReviewPanel = ref/)
  assert.doesNotMatch(source, /const showRecordings = ref/)
  assert.doesNotMatch(source, /const showAdvancedControls = ref/)
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

test('review has three tabs and opens overview by default', () => {
  assert.match(source, /type ReviewTool = 'overview' \| 'correction' \| 'speakers'/)
  assert.match(source, /activeReviewTool = ref<ReviewTool>\('overview'\)/)
  assert.match(source, /if \(tool === 'review'\) activeReviewTool\.value = 'overview'/)
  const from = source.indexOf('<nav class="meeting-review-tools"')
  const nav = source.slice(from, source.indexOf('</nav>', from))
  assert.equal((nav.match(/<button\b/g) ?? []).length, 3)
  for (const label of ['Overview', 'Corrections', 'Speakers']) {
    assert.ok(nav.includes('>' + label + '</button>'), 'Review tab missing: ' + label)
  }
  assert.doesNotMatch(nav, /Identity evidence|AI workspace|Speaker review|Choose a review tool/)
  assert.match(source, /<IntelligencePanel v-if="activeReviewTool === 'overview'" compact/)
  assert.match(source, /v-else-if="activeReviewTool === 'correction'"/)
  assert.match(source, /openIdentityEvidence[\s\S]*?activeReviewTool\.value = 'speakers'/)
  assert.match(source, /<SpeakerContributionPanel v-if="selectedSpeakerId !== null"/)
  assert.match(source, /<OfflineSpeakerPanel v-if="showIndependentSpeakerAnalysis"/)
})

test('existing transcription and recording workflows are retained', () => {
  for (const name of [
    'buildTranscriptParagraphs','expandContentFirstSegments',
    'saveMeeting()','recording ? stop() : start()','loadRecordings()',
    'applySpeechSettings()','copyTranscript','editSegmentById',
    'openIdentityEvidence','SpeakerContributionPanel',
  ]) assert.ok(source.includes(name), 'Missing workflow ' + name)
})

test('responsive tool CSS has no unbalanced blocks', () => {
  assert.equal((css.match(/\{/g) || []).length, (css.match(/\}/g) || []).length)
  for (const selector of ['meeting-tool-sidebar', 'meeting-detail-panel', 'meeting-review-tools']) {
    assert.ok(css.includes('.' + selector))
  }
  assert.match(css, /@media\(max-width:720px\)/)
})
