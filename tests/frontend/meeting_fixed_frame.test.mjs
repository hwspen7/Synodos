import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const page = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const css = readFileSync(path.join(root, 'src/assets/meeting-tool-layout.css'), 'utf8')

test('a single meeting command bar keeps every critical action reachable', () => {
  const start = page.indexOf('<Teleport to="#synodos-global-context">')
  const end = page.indexOf('</Teleport>', page.indexOf('<Teleport to="#synodos-global-actions">'))
  const header = page.slice(start, end + '</Teleport>'.length)
  assert.ok(start >= 0 && end > start, 'Global meeting toolbar teleport must exist')
  for (const value of [
    'meetingTitle', 'statusText', 'synodos-global-actions',
    "activeReviewTool = 'speakers'", 'synodos-language-control',
    "saveMeeting()", "recording ? stop() : start()",
    "exportMeeting('md')", "exportMeeting('json')", 'copyTranscript',
    "activeMeetingTool === 'recordings' ? 'transcript' : 'recordings'", "selectMeetingTool('settings')",
  ]) assert.ok(header.includes(value), 'Missing header content: ' + value)
  assert.match(page, /class="synodos-meeting-center"/)
  assert.doesNotMatch(page, /<header class="compact-meeting-header/)
  assert.match(header, /synodos-recordings-tool/)
  assert.match(page, /class="synodos-transcript-heading"/)
  assert.doesNotMatch(page, /class="meeting-workspace-summary"/)
  assert.doesNotMatch(page, /class="synodos-participant-stage"|class="meeting-tool-sidebar"/)
})

test('desktop meeting fills viewport and only content panes scroll', () => {
  for (const selector of [
    '.workspace-shell.meeting-route', '.meeting-route .workspace-main-column',
    '.meeting-route .workspace-page-content', '.meeting-route .meeting-workspace',
    '.meeting-route .meeting-primary .transcript',
    '.meeting-route .meeting-detail-scroll',
  ]) assert.ok(css.includes(selector), 'Missing viewport constraint: ' + selector)
  assert.ok(css.includes('height:100dvh'))
  assert.ok(css.includes('overflow-y:auto'))
  assert.ok(css.includes('overscroll-behavior:contain'))
})

test('existing transcript and tool functionality are retained', () => {
  for (const name of ['buildTranscriptParagraphs','selectMeetingTool','loadRecordings',
    'saveMeeting()','copyTranscript','applySpeechSettings()','IntelligencePanel',
    'SpeakerContributionPanel']) assert.ok(page.includes(name))
})

test('meeting template and layout style are structurally balanced', () => {
  for (const tag of ['div','header','button','section','nav','aside','main','article','template']) {
    const opens = [...page.matchAll(new RegExp('<'+tag+'(?=[\\s>])','g'))].length
    const closes = [...page.matchAll(new RegExp('</'+tag+'>','g'))].length
    assert.equal(opens, closes, tag + ' count mismatch')
  }
  assert.equal((css.match(/\{/g) || []).length,(css.match(/\}/g) || []).length)
})
