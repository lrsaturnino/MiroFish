import service from './index'

/**
 * 获取网页研究进度
 *
 * 后端短轮询端点：读取 `agent_research.jsonl` 与 `agent_research.meta.json`
 * 返回 `{processed_agents, total_active_agents, last_agent_id, last_ts, malformed_count}`。
 * 项目目录不存在时返回 404；路径越界尝试返回 400；二者都用 `{success: false, error: ...}` 错误信封。
 *
 * @param {string} projectId
 */
export const getResearchProgress = (projectId) => {
  return service.get(`/api/research/progress/${projectId}`)
}
