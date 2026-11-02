/** Assign display colors for speaker IDs provided by the backend. */

export function speakerColor(id: number): string {
  const colors = ['#5b8def', '#e07b54', '#5fbf8f', '#b58be0', '#d9b44a', '#6fc3d6']
  return colors[id % colors.length]!
}
