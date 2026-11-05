import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const page = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const css = readFileSync(path.join(root, 'src/assets/meeting-tool-layout.css'), 'utf8')

test('recording errors render as floating notices outside transcript region', () => {
  assert.doesNotMatch(page, /class="meeting-action-notice"/)
  assert.match(page, /class="meeting-floating-alert"/)
  assert.match(page, /aria-live="assertive"/)
  assert.ok(page.indexOf('class="meeting-floating-alert"') > page.indexOf('</main>'))
  assert.match(page, /@click="error = ''"/)
})

test('alerts preserve access to settings and recordings', () => {
  assert.match(page, /@click="selectMeetingTool\('settings'\)">Settings/)
  assert.match(page, /@click="selectMeetingTool\('recordings'\)">Recordings/)
})

test('notification is fixed without consuming workspace height', () => {
  assert.match(css, /\.meeting-floating-alert\s*\{\s*position:fixed/)
  assert.match(css, /@media\(max-width:720px\)/)
  assert.equal((css.match(/\{/g)||[]).length, (css.match(/\}/g)||[]).length)
})
