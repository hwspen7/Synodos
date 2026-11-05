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

test('review starts with an immediately usable overview, not a workflow chooser', () => {
  assert.match(meeting, /const activeReviewTool = ref<ReviewTool>\('overview'\)/)
  assert.doesNotMatch(meeting, /review-tool-welcome|Open AI workspace/)
  assert.match(meeting, /<IntelligencePanel v-if="activeReviewTool === 'overview'" compact/)
  assert.match(intelligence, /v-if="props.compact"/)
  assert.match(intelligence, /aria-label="Meeting review overview"/)
  assert.match(intelligence, /Generate summary/)
  assert.match(intelligence, /Extract next steps/)
  assert.match(intelligence, /briefSummary\?\.content\.summary/)
  assert.match(intelligence, /briefFindings\.decisions/)
  assert.match(intelligence, /briefFindings\.actions/)
})

test('advanced workflows remain accessible only after an explicit expansion', () => {
  assert.match(intelligence, /showAdvanced = ref\(false\)/)
  assert.match(intelligence, /v-if="!props.compact \|\| showAdvanced"/)
  assert.match(intelligence, /@click="showAdvanced = !showAdvanced"/)
  assert.match(intelligence, /More AI tools/)
  assert.match(intelligence, /const modes: Array/)
  assert.match(intelligence, /const views: Array/)
})

test('speaker confirmation and optional second model remain accessible in one section', () => {
  assert.match(meeting, /activeReviewTool\.value = 'speakers'/)
  assert.match(meeting, /@click="inspectSpeaker\(speaker\.firstSegmentId\)"/)
  assert.match(meeting, /<SpeakerContributionPanel v-if="selectedSpeakerId !== null"/)
  assert.match(meeting, /<OfflineSpeakerPanel v-if="showIndependentSpeakerAnalysis"/)
  assert.match(meeting, /@reviewed="loadSavedTranscript"/)
})

test('review uses a three-part responsive tab bar and supported icons', () => {
  assert.match(styles, /grid-template-columns:repeat\(3,minmax\(0,1fr\)\)/)
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
