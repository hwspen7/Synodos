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
  assert.match(source, /activeMeetingTool = ref<MeetingTool>\(/)
  assert.match(source, /: 'review',/)
  assert.doesNotMatch(source, /const showReviewPanel = ref/)
  assert.doesNotMatch(source, /const showRecordings = ref/)
  assert.doesNotMatch(source, /const showAdvancedControls = ref/)
})

test('meeting-local tools appear once in the compact header, not duplicated in navigation', () => {
  const start = source.indexOf('<Teleport to="#synodos-global-context">')
  const end = source.indexOf('</Teleport>', source.indexOf('<Teleport to="#synodos-global-actions">'))
  const header = source.slice(start, end + '</Teleport>'.length)
  assert.ok(start >= 0 && end > start, 'Global meeting toolbar teleport must exist')
  for (const tool of ['review', 'recordings', 'settings']) {
    assert.ok(header.includes("'" + tool + "'"), 'Missing tool in header: ' + tool)
  }
  assert.match(header, /class="synodos-quick-tool synodos-recordings-tool"/)
  assert.ok(header.indexOf('synodos-recordings-tool') < header.indexOf('synodos-meeting-more'))
  assert.doesNotMatch(header.slice(header.indexOf('synodos-meeting-more')), /\> Recordings\s*<\/button>/)
  assert.match(header, /class="synodos-meeting-more"/)
  assert.match(source, /class="meeting-detail-panel"/)
  assert.doesNotMatch(source, /class="meeting-tool-sidebar"/)
  assert.doesNotMatch(source, /showReviewPanel = !showReviewPanel/)
  assert.doesNotMatch(source, /showRecordings = !showRecordings/)
  assert.doesNotMatch(source, /showAdvancedControls = !showAdvancedControls/)
})

test('Assistant provides six directly accessible tools with overview as the default', () => {
  assert.match(source, /type ReviewTool = 'overview' \| 'knowledge' \| 'translation' \| 'media' \| 'correction' \| 'speakers'/)
  assert.match(source, /activeReviewTool = ref<ReviewTool>\('overview'\)/)
  assert.match(source, /if \(tool === 'review'\) activeReviewTool\.value = 'overview'/)
  const from = source.indexOf('<nav class="meeting-review-tools"')
  const nav = source.slice(from, source.indexOf('</nav>', from))
  assert.equal((nav.match(/<button\b/g) ?? []).length, 6)
  for (const label of ['Summary', 'Knowledge', 'Translation', 'Files', 'Corrections', 'Speakers']) {
    assert.ok(nav.includes('>' + label + '</button>'), 'Assistant section missing: ' + label)
  }
  assert.doesNotMatch(nav, /AI workspace|Choose a review tool|More AI tools/)
  assert.match(source, /<IntelligencePanel v-if="activeReviewTool !== 'correction' && activeReviewTool !== 'speakers'"/)
  assert.match(source, /:section="intelligenceSection"/)
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
  for (const selector of ['meeting-detail-panel', 'meeting-review-tools']) {
    assert.ok(css.includes('.' + selector))
  }
  assert.match(css, /@media\(max-width:720px\)/)
})
