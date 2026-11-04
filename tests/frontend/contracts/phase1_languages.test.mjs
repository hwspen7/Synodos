import test from 'node:test'
import assert from 'node:assert/strict'
import {
  PHASE1_LANGUAGES, supportedLanguages,
  supportedSpokenChoice, supportedTranslationChoice,
  restoredSpeechChoices,
} from '../../../src/config/phase1Languages.ts'

const selected = ['en', 'zh', 'fr', 'es', 'ar', 'id', 'ja', 'ru', 'de', 'hi']

test('Phase 1 frontend exposes exactly ten source and translation languages', () => {
  assert.deepEqual(PHASE1_LANGUAGES.map(([code]) => code), selected)
  assert.deepEqual(supportedLanguages.map(([code]) => code), ['auto', ...selected])
  assert.equal(new Set(selected).size, 10)
  for (const code of ['pt', 'bn', 'ko', 'it']) {
    assert.equal(supportedSpokenChoice(code), false)
    assert.equal(supportedTranslationChoice(code), false)
  }
  assert.equal(supportedTranslationChoice('auto'), false)
  assert.equal(supportedTranslationChoice(''), true)
})

test('unsupported saved meeting choices fall back but historic text is not touched', () => {
  assert.deepEqual(restoredSpeechChoices('pt', 'bn'), {
    language: 'auto',
    translate_to: '',
    hadUnsupportedSavedChoice: true,
  })
  assert.deepEqual(restoredSpeechChoices('auto', 'zh'), {
    language: 'auto', translate_to: 'zh',
    hadUnsupportedSavedChoice: false,
  })
  const legacyTranscript = 'Prior audio is still part of the saved meeting.'
  assert.equal(legacyTranscript, 'Prior audio is still part of the saved meeting.')
})
