import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const page = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const css = readFileSync(path.join(root, 'src/assets/meeting-tool-layout.css'), 'utf8')

test('Synodos meeting header, real speaker cards and recording bar preserve actions', () => {
  const header = page.slice(page.indexOf('<header class="compact-meeting-header">'), page.indexOf('</header>') + 9)
  const session = page.slice(page.indexOf('class="synodos-session-bar"'), page.indexOf('class="meeting-workspace"'))
  for (const value of [
    'meetingTitle', 'statusText', 'meeting-header-context',
    'translationTarget', 'durableSeconds', 'workspace-export-menu',
    'showParticipants = true',
  ]) assert.ok(header.includes(value), 'Missing header content: ' + value)
  for (const value of ["saveMeeting()", "recording ? stop() : start()", 'supportedLanguages']) {
    assert.ok(session.includes(value), 'Missing session action: ' + value)
  }
  assert.match(page, /class="synodos-participant-stage"/)
  assert.match(page, /meetingCast\.slice\(0, 3\)/)
  assert.doesNotMatch(page, /class="meeting-workspace-summary"/)
  assert.match(header, /\{\{ paragraphs\.length \}\} turns/)
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
