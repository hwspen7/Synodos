/** Presentation-only line breaks for live/confirmed captions.
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

/** Hard cap on visual line length when ASR does not emit punctuation.
 *
 * Prefer phrase punctuation or word boundaries. Never invent a word,
 * speaker turn, punctuation mark, or a timestamp for a visual chunk.
 */
function wrapLongSentence(sentence: string, maxChars = 100): string[] {
  const lines: string[] = []
  let rest = Array.from(sentence.trim())
  while (rest.length > maxChars) {
    const minBreak = Math.floor(maxChars * 0.48)
    let breakAt = -1
    // Prefer a clause boundary that still yields a reasonably sized line.
    for (let index = minBreak; index < maxChars && index < rest.length; index++) {
      if (/[,;\uff0c\uff1b\uff1a\u3001:]/u.test(rest[index] ?? '') &&
          (index + 1 === rest.length || /[\s\u3400-\u9fff]/u.test(rest[index + 1] ?? ''))) {
        breakAt = index + 1
      }
    }
    if (breakAt < 0) {
      // Choose the last word boundary to avoid breaking Latin words.
      for (let index = minBreak; index < maxChars && index < rest.length; index++) {
        if (/\s/u.test(rest[index] ?? '')) breakAt = index
      }
    }
    if (breakAt < 0) {
      // Long CJK text without spaces can break between code points.
      // For Latin strings with no safe break, keep the full token intact.
      const unit = rest.slice(0, maxChars).join('')
      if (/^[^\s\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+$/u.test(unit)) {
        break
      }
      breakAt = maxChars
    }
    const line = rest.slice(0, breakAt).join('').trim()
    if (!line) break
    lines.push(line)
    rest = rest.slice(breakAt)
    while (rest[0] && /\s/u.test(rest[0])) rest.shift()
  }
  if (rest.length) lines.push(rest.join('').trim())
  return lines.filter(Boolean)
}

export function splitDisplaySentences(text: string): string[] {
  const original = String(text || '').trim()
  if (!original) return []
  const result: string[] = []
  // Preserve explicit upstream line boundaries in addition to punctuation.
  for (const line of original.split(/\r?\n/u)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    for (const sentence of nativeSentenceBoundaries(trimmed)) {
      result.push(...wrapLongSentence(sentence))
    }
  }
  return result.length ? result : [original]
}
