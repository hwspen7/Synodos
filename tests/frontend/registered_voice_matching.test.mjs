import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const read = file => readFileSync(path.join(root, file), 'utf8')
const page = read('src/pages/MeetingPage.vue')
const types = read('src/types.ts')
const api = read('src/api/client.ts')
const ws = read('api/websocket.py')

test('meeting settings expose explicit biometric matching consent', () => {
  assert.match(page, /const autoMatchSaved = ref\(true\)/)
  assert.match(page, /v-model="autoMatchSaved"/)
  assert.match(page, /auto_match_saved: autoMatchSaved\.value/)
  assert.match(page, /auto_match_saved: String\(autoMatchSaved\.value\)/)
  assert.match(page, /meeting\.ui_settings\?\.auto_match_saved/)
  assert.match(api, /auto_match_saved: boolean/)
})

test('WebSocket round trip carries the effective match mode', () => {
  assert.match(ws, /auto_match_saved: Optional\[bool\] = Query/)
  assert.match(ws, /auto_match_saved=auto_match_enabled/)
  assert.match(ws, /"auto_match_active": session\.processor\._auto_profile_mode/)
  assert.match(types, /auto_match_active\?: boolean/)
  assert.match(page, /autoMatchSaved \? 'Auto-match eligible voices from your saved library'/)
})

test('the transcript shows a matched voice name with honest provenance', () => {
  assert.match(page, /Voice match/)
  assert.match(page, /registered_voiceprint/)
  assert.match(page, /matched_profile_id: s\.matched_profile_id/)
  assert.match(types, /speaker_match_similarity\?: number \| null/)
})

test('voice recognition changes do not change recording and diarization controls', () => {
  for (const critical of [
    'buildTranscriptParagraphs', 'expandContentFirstSegments',
    'selectMeetingTool', "recording ? stop() : start()",
    'saveMeeting()', 'applySpeechSettings()', 'openIdentityEvidence',
  ]) assert.ok(page.includes(critical), critical)
})

test('speaker library exposes enrollment eligibility without claiming verification', () => {
  const speakerPage = read('src/pages/SpeakersPage.vue')
  assert.match(speakerPage, /Auto-match eligible/)
  assert.match(speakerPage, /Needs a clean CAM\+\+ sample/)
  assert.match(speakerPage, /vector\.duration_s >= 2/)
})

test('saved meeting segments expose the same voice-match fields as live segments', () => {
  const live = types.slice(types.indexOf('export interface TranscriptSegment {'), types.indexOf('export interface ServerTranscript'))
  const saved = types.slice(types.indexOf('export interface MeetingSegment {'), types.indexOf('export interface HealthInfo'))
  for (const field of ['matched_profile_id', 'matched_profile_name', 'speaker_match_similarity']) {
    assert.match(live, new RegExp('\\b' + field + '\\?:'))
    assert.match(saved, new RegExp('\\b' + field + '\\?:'))
    assert.ok(page.includes(field + ': s.' + field), 'Saved transcript mapper omitted ' + field)
  }
})
