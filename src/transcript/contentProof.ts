/** Content-first frontend integrity guard, independent of Vue or Vite aliases.
 *
 * Validates the displayed words as a single ordered, text-conserving
 * projection of the source. A source caption remains canonical on failure.
 */
export interface TextCue {
  text: string
  start_ms: number
  end_ms: number
}

export function isExactContentProjection(
  sourceText: string,
  sourceStartMs: number,
  sourceEndMs: number,
  cues: readonly TextCue[],
): boolean {
  if (
    !Number.isInteger(sourceStartMs) || !Number.isInteger(sourceEndMs) ||
    sourceStartMs < 0 || sourceEndMs <= sourceStartMs ||
    !Array.isArray(cues) || cues.length < 2 || cues.length > 32
  ) return false
  for (let index = 0; index < cues.length; index++) {
    const cue = cues[index]!
    if (
      typeof cue.text !== 'string' || !cue.text.trim() ||
      !Number.isInteger(cue.start_ms) ||
      !Number.isInteger(cue.end_ms) ||
      cue.start_ms < sourceStartMs - 150 ||
      cue.end_ms > sourceEndMs + 150 ||
      cue.end_ms <= cue.start_ms ||
      (index > 0 && cue.start_ms < cues[index - 1]!.end_ms)
    ) return false
  }
  // Treat Unicode characters and punctuation as literal source evidence.
  // Strip only external whitespace; never normalize away a meaningful word.
  return cues.map((cue) => cue.text).join('').trim() === sourceText.trim()
}
