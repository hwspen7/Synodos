import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const read = (name) => readFileSync(path.join(root, name), 'utf8')

test('Synodos identity stays compact and warm', () => {
  const app = read('src/App.vue')
  const logo = read('src/components/SynodosMark.vue')
  const favicon = read('public/favicon.svg')
  const html = read('index.html')
  assert.match(app, /SynodosMark/)
  assert.match(app, /synodos-topbar/)
  assert.match(logo, /fill="#EB7100"/)
  assert.match(favicon, /#EB7100/)
  assert.match(html, /<title>Synodos/)
  assert.doesNotMatch(app, /synodos-tagline|synodos-profile-bubble|synodos-topbar-link/)
  assert.doesNotMatch(logo, /8850c9|4c1d83|68389a/i)
})

test('global sidebar lists unique destinations with no meeting-local duplicates', () => {
  const app = read('src/App.vue')
  for (const route of ['to="/"', 'to="/#meetings"', 'to="/speakers"', 'to="/hotwords"', 'to="/settings"']) {
    assert.ok(app.includes(route), 'Missing global route ' + route)
  }
  assert.doesNotMatch(app, /title="Live Captions"|title="Summaries"|summaryHref|meetingHref/)
  assert.equal((app.match(/class="workspace-nav-item"/g) || []).length, 5)
})

test('meeting uses a single command bar and keeps all existing tools reachable', () => {
  const meeting = read('src/pages/MeetingPage.vue')
  const start = meeting.indexOf('<Teleport to="#synodos-global-context">')
  const end = meeting.indexOf('</Teleport>', meeting.indexOf('<Teleport to="#synodos-global-actions">'))
  const header = meeting.slice(start, end + '</Teleport>'.length)
  assert.ok(start >= 0 && end > start, 'Global meeting toolbar teleport must exist')
  const app = read('src/App.vue')
  assert.match(app, /id="synodos-global-context"/)
  assert.match(app, /id="synodos-global-actions"/)
  assert.match(header, /<Teleport to="#synodos-global-actions">/)
  assert.doesNotMatch(meeting, /<header class="compact-meeting-header/)
  assert.match(header, /class="synodos-quick-tool synodos-recordings-tool"/)
  for (const operation of [
    "activeReviewTool = 'speakers'", "saveMeeting()", "recording ? stop() : start()",
    "activeMeetingTool === 'recordings' ? 'transcript' : 'recordings'", "selectMeetingTool('settings')",
    "exportMeeting('md')", "exportMeeting('json')", "copyTranscript",
    "activeMeetingTool === 'review' ? 'transcript' : 'review'",
  ]) assert.ok(header.includes(operation), 'Missing command bar action ' + operation)
  assert.match(header, /aria-label="Recognition language"/)
  assert.ok(header.indexOf('synodos-recordings-tool') < header.indexOf('synodos-meeting-more'))
  assert.doesNotMatch(header.slice(header.indexOf('synodos-meeting-more')), /> Recordings\s*<\/button>/)
  assert.doesNotMatch(meeting, /class="synodos-participant-stage"|class="synodos-top-controls"|class="synodos-session-bar"/)
  assert.doesNotMatch(meeting, /class="meeting-tool-sidebar"/)
  assert.match(meeting, /class="synodos-meeting-center"/)
  assert.match(meeting, /<IntelligencePanel v-if="activeReviewTool === 'overview'" compact/)
  assert.doesNotMatch(meeting, /<video\b/)
})

test('overlap diagnostic message is not displayed in the transcript', () => {
  const meeting = read('src/pages/MeetingPage.vue')
  assert.doesNotMatch(meeting, /Overlapping voices detected in|Untranscribed words are never fabricated|transcript-overlap-warning/)
  assert.match(meeting, /para\.overlapping/)
  assert.match(meeting, /buildTranscriptParagraphs/)
})

test('compact meeting style is content-first and keeps an optional side assistant', () => {
  const css = read('src/assets/synodos-theme.css')
  assert.match(css, /--mop-primary: #eb7100/i)
  assert.match(css, /--mop-accent: #f37021/i)
  assert.match(css, /--synodos-paper: #faf8f5/i)
  assert.match(css, /grid-template-rows: 52px minmax\(0, 1fr\)/)
  assert.match(css, /grid-template-columns:minmax\(0,1fr\) minmax\(310px,25%\)/)
  assert.match(css, /meeting-workspace:not\(\.tool-panel-open\)/)
  assert.match(css, /\.synodos-meeting-more-popover/)
  assert.match(css, /@media\(max-width:760px\)/)
  assert.match(css, /prefers-reduced-motion:reduce/)
  assert.doesNotMatch(css, /#68389a|#8850c9|#4b196f/i)
  assert.doesNotMatch(css, /synodos-participant-stage|synodos-top-controls/)
  assert.equal((css.match(/\{/g) || []).length, (css.match(/\}/g) || []).length)
})

test('meeting settings and backend-connected screens are still available', () => {
  const route = read('src/router/index.ts')
  const settings = read('src/pages/SettingsPage.vue')
  assert.match(route, /path: '\/settings'/)
  assert.match(settings, /fetch\('\/api\/health'\)/)
  assert.match(settings, /Open meetings/)
  assert.match(read('src/main.ts'), /synodos-theme\.css/)
})
