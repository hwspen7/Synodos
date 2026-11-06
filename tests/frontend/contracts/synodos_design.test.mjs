import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const read = (name) => readFileSync(path.join(root, name), 'utf8')

test('Synodos keeps a distinct Greek-inspired identity throughout the app', () => {
  const app = read('src/App.vue')
  const logo = read('src/components/SynodosMark.vue')
  const favicon = read('public/favicon.svg')
  const html = read('index.html')
  assert.match(app, /SynodosMark/)
  assert.match(app, /synodos-topbar/)
  assert.match(app, /Meet <b>·<\/b> Transcribe <b>·<\/b> Understand/)
  assert.match(logo, /fill="#EB7100"/)
  assert.match(favicon, /#EB7100/)
  assert.doesNotMatch(logo, /8850c9|4c1d83|68389a/i)
  assert.match(html, /<title>Synodos/)
  assert.doesNotMatch(html, /<title>Meeting of Pioneers/)
})

test('meeting follows reference layout with live data, not invented participants', () => {
  const meeting = read('src/pages/MeetingPage.vue')
  assert.match(meeting, /class="synodos-participant-stage"/)
  assert.match(meeting, /const meetingCast = computed/)
  assert.match(meeting, /reviewSpeakers\.value\.map/)
  assert.match(meeting, /participantNames\.value\.map/)
  assert.match(meeting, /meetingCast\.slice\(0, 3\)/)
  assert.match(meeting, /class="synodos-session-bar"/)
  assert.match(meeting, /class="synodos-meeting-center"/)
  assert.match(meeting, /class="synodos-transcript-heading"/)
  assert.match(meeting, /<IntelligencePanel v-if="activeReviewTool === 'overview'" compact/)
  assert.doesNotMatch(meeting, /<video\b/)
})

test('new navigation and settings remain connected to real workflows', () => {
  const app = read('src/App.vue')
  const route = read('src/router/index.ts')
  const prefs = read('src/pages/SettingsPage.vue')
  assert.match(app, /to="\/speakers"/)
  assert.match(app, /to="\/hotwords"/)
  assert.match(app, /settingsHref/)
  assert.match(route, /path: '\/settings'/)
  assert.match(prefs, /fetch\('\/api\/health'\)/)
  assert.match(prefs, /Open meetings/)
  assert.match(read('src/main.ts'), /synodos-theme\.css/)
})

test('responsive design style rules remain balanced', () => {
  const css = read('src/assets/synodos-theme.css')
  assert.match(css, /--mop-primary: #eb7100/i)
  assert.match(css, /--mop-accent: #f37021/i)
  assert.match(css, /--synodos-paper: #faf8f5/i)
  assert.doesNotMatch(css, /#68389a|#8850c9|#4b196f/i)
  assert.match(css, /@media \(max-width:760px\)/)
  assert.match(css, /prefers-reduced-motion:reduce/)
  assert.equal((css.match(/\{/g) || []).length, (css.match(/\}/g) || []).length)
})
