/** Phase 1 product support, not the complete language range of Whisper.
 * Historical transcripts may still contain other language codes.
 */
export const PHASE1_LANGUAGES = [
  ['en', 'English'],
  ['zh', 'Chinese'],
  ['fr', 'French'],
  ['es', 'Spanish'],
  ['ar', 'Arabic'],
  ['id', 'Indonesian'],
  ['ja', 'Japanese'],
  ['ru', 'Russian'],
  ['de', 'German'],
  ['hi', 'Hindi'],
] as const

const codes = new Set<string>(PHASE1_LANGUAGES.map(([code]) => code))

export const supportedLanguages: ReadonlyArray<readonly [string, string]> = [
  ['auto', 'Detect automatically'],
  ...PHASE1_LANGUAGES,
]

/** Only a new recording choice, never a filter on saved transcript text. */
export function supportedSpokenChoice(value: unknown): value is string {
  return typeof value === 'string' && (value === 'auto' || codes.has(value))
}

export function supportedTranslationChoice(value: unknown): value is string {
  return typeof value === 'string' && (value === '' || codes.has(value))
}

export function restoredSpeechChoices(source: unknown, target: unknown) {
  return {
    language: supportedSpokenChoice(source) ? source : 'auto',
    translate_to: supportedTranslationChoice(target) ? target : '',
    hadUnsupportedSavedChoice: (
      (source != null && !supportedSpokenChoice(source)) ||
      (target != null && !supportedTranslationChoice(target))
    ),
  }
}
