import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { test } from 'node:test'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const read = file => readFileSync(path.join(root, file), 'utf8')
const meeting = read('src/pages/MeetingPage.vue')
const intelligence = read('src/components/IntelligencePanel.vue')
const styles = read('src/assets/meeting-tool-layout.css')
const icons = read('src/components/AppIcon.vue')
const theme = read('src/assets/synodos-theme.css')
const corrections = read('src/components/ContextCorrectionPanel.vue')

test('review starts with an immediately usable overview, not a workflow chooser', () => {
  assert.match(meeting, /const activeReviewTool = ref<ReviewTool>\('overview'\)/)
  assert.doesNotMatch(meeting, /review-tool-welcome|Open AI workspace/)
  assert.match(meeting, /<IntelligencePanel v-if="activeReviewTool !== 'correction' && activeReviewTool !== 'speakers'"/)
  assert.match(meeting, /:section="intelligenceSection"/)
  assert.match(intelligence, /aria-label="Meeting AI tools"/)
  assert.match(intelligence, /Generate summary/)
  assert.doesNotMatch(intelligence, /Extract next steps/)
  assert.match(intelligence, /@click="analyze\(\)"/)
  assert.match(intelligence, /briefSummary\.value\?\.content \?\? briefExtraction\.value\?\.content/)
  assert.doesNotMatch(intelligence, /<section v-if="briefSummary \|\| briefExtraction" class="review-insights-section"/)
  assert.match(intelligence, /<section class="review-insights-section">\s*<h4>Decisions/)
  assert.match(intelligence, /<section class="review-insights-section">\s*<h4>Next steps/)
  assert.match(intelligence, /v-if="jobs\.length"/)
  assert.doesNotMatch(intelligence, /<h3>Meeting overview<\/h3>/)
  assert.match(intelligence, /briefSummary\?\.content\.summary/)
  assert.match(intelligence, /briefFindings\.decisions/)
  assert.match(intelligence, /briefFindings\.actions/)
})

test('AI abilities are integrated into the Assistant without a second workspace', () => {
  assert.doesNotMatch(intelligence, /showAdvanced|More AI tools|AI Workspace|Choose a workflow/)
  assert.match(intelligence, /const modes: Array/)
  assert.match(intelligence, /aria-label="Analysis type"/)
  assert.doesNotMatch(intelligence, /view === 'knowledge'/)
  assert.match(intelligence, /view === 'translation'/)
  assert.doesNotMatch(intelligence, /view === 'media'/)
  for (const action of ['queueTranslations', 'cancelJob(job)', "review('actions'", "review('decisions'"]) {
    assert.ok(intelligence.includes(action), 'Missing integrated action ' + action)
  }
})

test('speaker confirmation and optional second model remain accessible in one section', () => {
  assert.match(meeting, /activeReviewTool\.value = 'speakers'/)
  assert.match(meeting, /@click="inspectSpeaker\(speaker\.firstSegmentId\)"/)
  assert.match(meeting, /<SpeakerContributionPanel v-if="selectedSpeakerId !== null"/)
  assert.match(meeting, /<OfflineSpeakerPanel v-if="showIndependentSpeakerAnalysis"/)
  assert.match(meeting, /@reviewed="loadSavedTranscript"/)
})

test('review uses four aligned tabs and supported icons', () => {
  const theme = read('src/assets/synodos-theme.css')
  assert.match(theme, /grid-template-columns:repeat\(4,minmax\(0,1fr\)\)/)
  assert.match(intelligence, /class="review-insights-controls"/)
  for (const icon of ['chevron-right', 'chevron-up', 'chevron-down']) {
    assert.ok(icons.includes("| '" + icon + "'"), 'Missing icon name ' + icon)
    assert.ok(icons.includes("name === '" + icon + "'"), 'Missing SVG path for ' + icon)
  }
})

test('updated component tags and styles are balanced', () => {
  for (const source of [meeting, intelligence]) {
    for (const tag of ['section', 'div', 'button', 'nav', 'article', 'template']) {
      const opening = [...source.matchAll(new RegExp('<' + tag + '(?=[\\s>])', 'g'))].length
      const closing = [...source.matchAll(new RegExp('</' + tag + '>', 'g'))].length
      assert.equal(opening, closing, 'Unbalanced ' + tag)
    }
  }
  assert.equal((styles.match(/\{/g) || []).length, (styles.match(/\}/g) || []).length)
})

test('Assistant expands independently and presents unified analysis sections', () => {
  const theme = read('src/assets/synodos-theme.css')
  assert.match(meeting, /'assistant-panel-open': activeMeetingTool === 'review'/)
  assert.match(theme, /\.meeting-workspace\.tool-panel-open\.assistant-panel-open \{\s*grid-template-columns:minmax\(0,1fr\) 500px;/)
  assert.match(theme, /\.meeting-workspace\.tool-panel-open\.assistant-panel-open \{\s*grid-template-columns:minmax\(0,1fr\) 470px;/)
  assert.match(intelligence, /class="review-insights-results"/)
  assert.match(intelligence, /\.intelligence-inline\.review-insights \.review-insights-results \{/)
  assert.match(intelligence, /\.review-insights-results \.review-insights-section h4:before \{\s*content:none;/)
  assert.match(theme, /\.review-insights-results \.review-insights-section h4 \{\s*display:flex;align-items:center;justify-content:flex-start;/)
  assert.match(intelligence, /<span class="visually-hidden">Analysis type<\/span>/)
  for (const heading of ['Summary', 'Decisions', 'Next steps']) {
    assert.match(intelligence, new RegExp('<h4>' + heading))
  }
  assert.doesNotMatch(intelligence, /Extract next steps/)
})

test('Corrections has one compact toolbar and retains explicit accept and dismiss', () => {
  assert.match(meeting, /class="review-focus-panel review-correction-panel"/)
  assert.doesNotMatch(meeting, /<h3>Transcript corrections<\/h3>/)
  assert.match(corrections, /class="context-actions context-review-toolbar"/)
  assert.match(corrections, /Suggestions only\. Changes require your approval\./)
  assert.match(corrections, /@click="startReview\(false\)"/)
  assert.match(corrections, /@click="accept\(item\)"/)
  assert.match(corrections, /@click="dismiss\(item\)"/)
  assert.match(theme, /\.review-correction-panel \.context-review-toolbar \{/)
})
