/** 说话人展示颜色：聚类由后端完成，前端只负责把 speaker_id 映射成颜色。 */

export function speakerColor(id: number): string {
  const colors = ['#5b8def', '#e07b54', '#5fbf8f', '#b58be0', '#d9b44a', '#6fc3d6']
  return colors[id % colors.length]!
}
