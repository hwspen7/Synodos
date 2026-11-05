import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const read = file => readFileSync(path.join(root, file), 'utf8')
const meeting = read('src/pages/MeetingPage.vue')
const panel = read('src/components/SpeakerContributionPanel.vue')
const api = read('api/intelligence.py')
const worker = read('intelligence/worker.py')
const store = read('intelligence/store.py')

test('Speaker review includes an exclusive per-speaker detail and retains manual review', () => {
  assert.match(meeting, /import SpeakerContributionPanel/)
  assert.match(meeting, /selectedSpeakerId = ref<number \| null>\(null\)/)
  assert.match(meeting, /:speaker-id="selectedSpeakerId"/)
  assert.match(meeting, /:selected-segment-id="identityReviewSegmentId"/)
  assert.match(meeting, /@reviewed="loadSavedTranscript"/)
  assert.match(meeting, /@jump-to="jumpToSpeakerStatement"/)
  assert.match(meeting, /speakerDetailTab\.value = 'identity'/)
})

test('Speaker detail tabs provide scoped AI, originals and identity corrections', () => {
  const nav = panel.slice(panel.indexOf('<nav class="speaker-contribution-tabs"'), panel.indexOf('</nav>', panel.indexOf('<nav class="speaker-contribution-tabs"')))
  assert.equal((nav.match(/<button\b/g) ?? []).length, 3)
  for (const label of ['Insights','Statements','Identity']) {
    assert.ok(nav.includes('>' + label + '</button>'), label)
  }
  assert.match(panel, /filter\(s => s\.speaker_id === props\.speakerId/)
  assert.match(panel, /<Phase2IdentityReviewPanel v-if="identitySegmentId !== null"/)
  assert.match(panel, /@click="emit\('jump-to', segment\.id\)"/)
})

test('Speaker AI requests include speaker ID and saved revision, never all-speaker payload', () => {
  assert.match(panel, /kind: 'speaker_insights'/)
  assert.match(panel, /speaker_id: props\.speakerId, source_revision: status\.value\.source_revision/)
  assert.match(api, /choose_speaker_turns\(turns, sid\)/)
  assert.match(worker, /analyze_speaker\(segments, speaker_id, gateway\)/)
  assert.match(store, /'speaker_id': int\(s\.get\('speaker_id'\) or -1\)/)
})

test('Every AI output section exposes cited source links', () => {
  assert.match(panel, /insightContent\.summary_source_ids/)
  assert.match(panel, /item\.source_ids/)
  assert.match(panel, /@click="emit\('jump-to', sourceId\)"/)
  assert.match(meeting, /class="paragraph"[\s\S]*?paragraph-highlighted/)
})

test('Speaker insight changes are tied to source revisions and recording state', () => {
  assert.match(panel, /!props\.recording/)
  assert.match(panel, /artifact && !artifact\.stale/)
  assert.match(panel, /The transcript changed since this analysis/)
  assert.match(api, /current_revision != rev/)
  assert.match(worker, /current_revision != revision/)
})

test('Vue tags remain structurally balanced', () => {
  for (const source of [meeting, panel]) {
    for (const tag of ['section', 'div', 'button', 'nav', 'article', 'template', 'header']) {
      const opens = [...source.matchAll(new RegExp('<' + tag + '(?=[\\s>])', 'g'))].length
      const closes = [...source.matchAll(new RegExp('</' + tag + '>', 'g'))].length
      assert.equal(opens, closes, 'Unbalanced ' + tag)
    }
  }
})
