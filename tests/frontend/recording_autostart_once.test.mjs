import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const page = readFileSync(path.join(root, 'src/pages/MeetingPage.vue'), 'utf8')
const home = readFileSync(path.join(root, 'src/pages/HomePage.vue'), 'utf8')

test('autostart is one-shot, not persistent across reloads', () => {
  const mount = page.slice(page.indexOf('onMounted(async () => {'),page.indexOf('onBeforeUnmount('))
  const consumed = mount.indexOf('await router.replace({ query: remainingQuery')
  const init = mount.indexOf('await loadMeetingConfig()')
  const connect = mount.indexOf('if (startOnce) connect(true)')
  assert.ok(consumed >= 0 && consumed < init, 'Consume URL flag before async load')
  assert.ok(connect > init, 'Start only after initialization')
  assert.match(mount, /const startOnce = route.query.autostart === '1'/)
})

test('socket ready callback starts mic only after explicit connect(true)', () => {
  assert.match(page, /if \(autostart\) void startMic\(\)/)
  assert.doesNotMatch(page, /autostart \|\| route.query.autostart/)
})

test('explicit create and resume paths still use one-shot launch intent', () => {
  assert.match(home, /query: \{ autostart: '1' \}/)
  assert.match(home, /\?autostart=1/)
})
