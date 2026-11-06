/** Separate terms without breaking names and phrases at spaces. */
export function parseHotwords(text: string): string[] {
  return [...new Set(text.split(/[,;\r\n\uFF0C\u3001\uFF1B]+/u)
    .map((item) => item.trim().replace(/\s+/g, ' '))
    .filter(Boolean))]
}
