import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const page = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const css = readFileSync(path.join(root, 'src/assets/meeting-tool-layout.css'), 'utf8')
const theme = readFileSync(path.join(root, 'src/assets/synodos-theme.css'), 'utf8')

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
  assert.doesNotMatch(page, /class="synodos-transcript-heading"/)
  assert.match(page, /<main ref="transcriptEl" class="transcript"[\s\S]*?aria-label="Meeting transcript"/)
  assert.match(page, /class="compact-meeting-titles"/)
  assert.match(page, /class="meeting-running-status" role="status"/)
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

test('meeting identity stays inline with an unobtrusive status', () => {
  assert.ok(theme.includes('display:flex;flex-direction:row;align-items:center;gap:12px;'))
  assert.ok(theme.includes('padding:0;border:0;border-radius:0;background:transparent;'))
  assert.ok(!theme.includes('.synodos-transcript-heading {'))
})

test('inline meeting rename persists, and controls use descriptive aligned icons', () => {
  const client = readFileSync(path.join(root, 'src/api/client.ts'), 'utf8')
  const icon = readFileSync(path.join(root, 'src/components/AppIcon.vue'), 'utf8')
  assert.match(page, /class="meeting-title-trigger"/)
  assert.match(page, /@click="editMeetingTitle"/)
  assert.match(page, /@keydown\.enter\.stop\.prevent="commitMeetingRename"/)
  assert.match(page, /@keydown\.esc\.stop\.prevent="cancelMeetingRename"/)
  assert.match(page, /@blur="commitMeetingRename"/)
  assert.match(page, /meetingsApi\.rename\(props\.id, title\)/)
  assert.match(client, /rename: async \(id: string, title: string\)/)
  assert.match(client, /method: 'POST', body/)
  assert.match(client, /method: 'PATCH', body/)
  assert.match(page, /const renameError = ref\(''\)/)
  assert.match(page, /class="meeting-title-error" role="alert"/)
  assert.doesNotMatch(page.slice(page.indexOf('async function commitMeetingRename'), page.indexOf('const bufferedMs')), /error\.value|showToast/)
  assert.match(theme, /\.meeting-title-editor\[aria-invalid="true"\]/)
  assert.match(page, /AppIcon name="sparkles"/)
  assert.match(page, /AppIcon name="audio-lines"/)
  assert.match(icon, /name === 'sparkles'/)
  assert.match(icon, /name === 'audio-lines'/)
  assert.match(theme, /meeting-title-editor/)
  assert.match(theme, /min-width:177px/)
  assert.match(theme, /\.synodos-brand-mark \{\s*position:relative;top:2px;/)
})

test('Assistant toolbar label and one shared vertical centerline are explicit', () => {
  assert.match(page, /class="synodos-quick-tool synodos-assistant-tool"/)
  assert.match(page, /aria-label="Assistant" title="Assistant"/)
  assert.match(page, /AppIcon name="sparkles" :size="16"\s*\/><span>Assistant<\/span>/)
  assert.doesNotMatch(page, /<span>AI<\/span>/)
  assert.match(theme, /height:52px;min-height:52px;box-sizing:border-box;align-items:center/)
  assert.match(theme, /height:34px;min-height:34px;align-items:center;gap:6px/)
  assert.match(theme, /\.synodos-global-actions \.synodos-meeting-more \{\s*display:flex;align-items:center;height:34px;/)
  assert.match(theme, /\.meeting-title-trigger \{[\s\S]*?height:34px;/)
})
