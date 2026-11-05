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

test('design system uses warm orange palette rather than previous purple primary', () => {
  const base = read('src/assets/main.css')
  const shared = read('src/assets/workspace-system.css')
  assert.match(base, /--mop-primary:\s*#eb7100/i)
  assert.match(shared, /--ui-orange:#eb7100/i)
  assert.doesNotMatch(shared, /#68389a/i)
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
