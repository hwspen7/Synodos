import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const read = (name) => readFileSync(path.join(root, name), 'utf8')

test('global workspace has accessible primary routes and a responsive sidebar', () => {
  const shell = read('src/App.vue')
  for (const route of ['to="/"', 'to="/#meetings"', 'to="/speakers"', 'to="/hotwords"']) {
    assert.ok(shell.includes(route), 'Missing navigation route: ' + route)
  }
  assert.match(shell, /aria-label="Workspace navigation"/)
  assert.match(shell, /workspace-mobile-menu/)
  assert.match(shell, /workspace-mobile-scrim/)
  assert.match(shell, /sidebarCollapsed/)
})

test('Synodos uses warm Hermes/Chegg-inspired orange with restrained neutrals', () => {
  const tokens = read('src/assets/synodos-theme.css')
  const app = read('src/App.vue')
  const favicon = read('public/favicon.svg')
  assert.match(tokens, /--mop-primary:\s*#eb7100/i)
  assert.match(tokens, /--mop-accent:\s*#f37021/i)
  assert.match(tokens, /--synodos-ink:\s*#292521/i)
  assert.doesNotMatch(tokens, /#68389a|#8850c9|#4b196f/i)
  assert.doesNotMatch(tokens, /radial-gradient\(|linear-gradient\(/)
  assert.match(app, /<SynodosMark /)
  assert.match(app, /Synodos/)
  assert.match(favicon, /#EB7100/i)
  assert.doesNotMatch(favicon, /8850c9|4c1d83/i)
})

test('meeting controls preserve all critical operations after redesign', () => {
  const meeting = read('src/pages/MeetingPage.vue')
  for (const action of [
    'recording ? stop() : start()', 'saveMeeting()', "exportMeeting('md')",
    "exportMeeting('json')", 'copyTranscript', 'applySpeechSettings()',
    'showParticipants = true', 'showHotwords = true',
    'buildTranscriptParagraphs', 'loadSavedTranscript',
  ]) assert.ok(meeting.includes(action), 'Missing meeting action: ' + action)
  assert.match(meeting, /meeting-config-grid/)
  assert.match(meeting, /recording-card-grid/)
  assert.match(meeting, /Technical diagnostics/)
})

test('all library and creation screens continue to render their functional forms', () => {
  const checks = [
    ['src/pages/HomePage.vue', 'createMeeting'],
    ['src/pages/SpeakersPage.vue', 'SpeakerVectorAdder'],
    ['src/pages/HotwordsPage.vue', 'saveEdit'],
    ['src/pages/SpeakerNewPage.vue', 'createSpeaker'],
    ['src/pages/HotwordNewPage.vue', 'createLibrary'],
  ]
  for (const [file, needle] of checks) {
    assert.ok(read(file).includes(needle), 'Missing screen action: ' + file)
  }
})

test('CSS blocks and Vue UI templates remain balanced', () => {
  for (const file of ['src/assets/workspace-system.css', 'src/assets/home-refresh.css', 'src/assets/meeting-refresh.css']) {
    const css = read(file)
    assert.equal((css.match(/\{/g) || []).length, (css.match(/\}/g) || []).length, 'Unbalanced CSS: ' + file)
  }
  for (const file of ['src/App.vue', 'src/pages/MeetingPage.vue', 'src/pages/HomePage.vue']) {
    const vue = read(file)
    for (const tag of ['div','button','section','article']) {
      const start = (vue.match(new RegExp('<' + tag + '(?=[\\s>])', 'g')) || []).length
      const end = (vue.match(new RegExp('</' + tag + '>', 'g')) || []).length
      assert.equal(start, end, 'Unbalanced ' + tag + ' in ' + file)
    }
  }
})

test('home welcome is editorial, responsive, and keeps its existing actions', () => {
  const page = read('src/pages/HomePage.vue')
  const css = read('src/assets/home-refresh.css')
  const tokens = read('src/assets/synodos-theme.css')
  assert.match(page, /id="home-hero-title"/)
  assert.match(page, /Talk freely\./)
  assert.match(page, /Leave with <em>clarity\.<\/em>/)
  assert.match(page, /class="home-hero-preview"/)
  assert.match(page, /class="home-preview-demo-label">PREVIEW/)
  assert.match(page, /class="home-hero-visual" aria-hidden="true"/)
  assert.match(page, /class="home-hero-primary"/)
  assert.match(page, /@click="showCreate = true"/)
  assert.match(page, /href="#meetings"/)
  assert.match(page, /<main class="page-shell">/)
  assert.match(css, /grid-template-columns:minmax\(0,1\.12fr\) minmax\(350px,\.88fr\)/)
  assert.match(css, /@media\(max-width:950px\)/)
  assert.match(css, /@media\(max-width:680px\)/)
  assert.match(css, /prefers-reduced-motion:reduce/)
  assert.doesNotMatch(tokens, /\.synodos-shell \.home-compact-hero h1 \{/)
})
