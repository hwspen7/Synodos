/** Presentation-only sentence boundaries for live/confirmed captions.
 *
 * This code never changes saved transcript text, timestamps, revisions,
 * or speaker identities. A visual line is NOT a new timed speaker turn.
 */
const TITLE_ABBREVIATIONS =
  /(?:^|\s)(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e|u\.s|u\.k)\.$/i
const ABBREVIATIONS = new Set([
  'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'st.',
  'vs.', 'etc.', 'e.g.', 'i.e.', 'u.s.', 'u.k.',
])

function fallbackSentenceBoundaries(original: string): string[] {
  const result: string[] = []
  let current = ''
  for (let index = 0; index < original.length; index++) {
    const char = original.charAt(index)
    current += char
    // Do not treat abbreviations, initials or decimal points as sentence ends.
    if (!'.!?\u3002\uff01\uff1f'.includes(char)) continue
    const cjkEnd = '\u3002\uff01\uff1f'.includes(char)
    const next = original.charAt(index + 1)
    if (!cjkEnd && next && !/\s/.test(next)) continue
    const token = (current.match(/(?:^|\s)(\S+)$/)?.[1] || '').toLowerCase()
    if (!cjkEnd && ABBREVIATIONS.has(token)) continue
    if (char === '.' && /\d$/.test(current.slice(0, -1)) && /^\d/.test(next)) continue
    while (index + 1 < original.length &&
           /["'\u201d\u2019\uff09\])}]/.test(original.charAt(index + 1))) {
      current += original.charAt(++index)
    }
    if (current.trim()) result.push(current.trim())
    current = ''
    while (index + 1 < original.length &&
           /\s/.test(original.charAt(index + 1))) index++
  }
  if (current.trim()) result.push(current.trim())
  return result
}

function nativeSentenceBoundaries(original: string): string[] {
  if (typeof Intl.Segmenter !== 'function') return fallbackSentenceBoundaries(original)
  const pieces = [...new Intl.Segmenter(undefined, { granularity: 'sentence' })
    .segment(original)].map((item) => item.segment.trim()).filter(Boolean)
  const repaired: string[] = []
  for (const piece of pieces) {
    const previous = repaired[repaired.length - 1]
    if (previous && TITLE_ABBREVIATIONS.test(previous)) {
      repaired[repaired.length - 1] = previous + ' ' + piece
    } else {
      repaired.push(piece)
    }
  }
  return repaired
}

export function splitDisplaySentences(text: string): string[] {
  const original = String(text || '').trim()
  if (!original) return []
  const result: string[] = []
  // Preserve explicit upstream line boundaries in addition to punctuation.
  for (const line of original.split(/\r?\n/u)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    // Keep a complete grammatical sentence in one display element.
    // Responsive CSS wraps long text without inventing new paragraphs.
    result.push(...nativeSentenceBoundaries(trimmed))
  }
  return result.length ? result : [original]
}
