/** Display-only warm, distinguishable earth tones for speaker IDs. */

export function speakerColor(id: number): string {
  const colors = ['#AD4E11', '#8C5B32', '#B6661D', '#735944', '#C66B29', '#996B41']
  return colors[((id % colors.length) + colors.length) % colors.length]!
}
