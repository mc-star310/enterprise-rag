"""搜索工具：把自研的 Bing 网页搜索包装成框架工具（7.5 自定义工具开发）

这一节的核心思想：**任何一个 Python 函数，只要套上 Tool 的壳，就能被 Agent 调用。**
你自研的 Bing 搜索原本只是一个普通函数，模型根本看不到它。包装成 Tool 子类后，
它被塞进 ToolRegistry，名字和用法会出现在 ReAct 的提示词里，模型就能自主决定
"这个问题我该不该搜一下"。

工程上的两个关键改进（都是踩过坑才改的）：
  1. **优先用 Bing 的 RSS 接口**（?format=rss），而不是爬 HTML。
     爬 HTML 依赖 li.b_algo 这种 class 名，页面一改就废，而且连续请求会被降级
     （实测：第一次结果很好，第二三次全是官网首页）。RSS 是结构化数据，稳定得多。
  2. **HTML 抓取作为兜底**：RSS 挂了才走老路，保证工具不会整体失效。

对外工具名：web_search
ReAct 调用格式：web_search[关键词]
"""

import requests
from bs4 import BeautifulSoup

from ..base import Tool, ToolParameter

# 伪装成浏览器，避免被搜索引擎判定为爬虫
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class SearchTool(Tool):
    """基于 Bing 的免费网页搜索工具（无需任何 API Key）。"""

    def __init__(self, max_results: int = 3, max_length: int = 800):
        super().__init__(
            name="web_search",
            description=(
                "一个基于Bing的网页搜索工具。当你需要获取实时信息、最新事件、"
                "或不确定的事实性问题时使用。输入搜索关键词，返回相关标题与摘要。"
                "提示：英文关键词通常能搜到更精确的结果。"
            ),
        )
        self.max_results = max_results
        self.max_length = max_length  # 截断，防止搜索结果把上下文撑爆
        self._cache: dict[str, str] = {}  # 简单缓存：同一个问题不重复联网

    # ---------- 主入口 ----------
    def run(self, parameters: dict) -> str:
        query = parameters.get("query", "").strip()
        try:
            max_results = int(parameters.get("max_results", self.max_results))
        except (TypeError, ValueError):
            max_results = self.max_results
        max_results = max(1, min(max_results, 10))

        if not query:
            return "搜索失败：缺少 query 参数。"

        # 缓存命中就直接返回（省时间、也降低被反爬的概率）
        if query in self._cache:
            return self._cache[query]

        # 先试 RSS，拿不到再退回 HTML 抓取
        result = self._search_rss(query, max_results)
        if not result or "未找到" in result:
            result = self._search_html(query, max_results)

        self._cache[query] = result
        return result

    # ---------- 方案 A：Bing RSS（首选，结构化）----------
    def _search_rss(self, query: str, max_results: int) -> str:
        try:
            resp = requests.get(
                "https://www.bing.com/search",
                params={"q": query, "format": "rss", "count": str(max_results * 2)},
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "xml")
            items = soup.select("item")
            if not items:
                return ""

            snippets = []
            for item in items:
                title = item.select_one("title")
                desc = item.select_one("description")
                if not (title and desc):
                    continue
                t = title.get_text(strip=True)
                d = desc.get_text(strip=True)
                if not d:
                    continue
                snippets.append(f"{t}: {d}")
                if len(snippets) >= max_results:
                    break

            if not snippets:
                return "未找到相关搜索结果。"
            return self._join(snippets)
        except Exception as e:
            return f"搜索时发生错误(RSS): {e}"

    # ---------- 方案 B：HTML 抓取（兜底）----------
    def _search_html(self, query: str, max_results: int) -> str:
        try:
            resp = requests.get(
                "https://www.bing.com/search",
                params={"q": query},
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            snippets = []
            for item in soup.select("li.b_algo"):
                title_tag = item.select_one("h2")
                body_tag = (
                    item.select_one(".b_caption p")
                    or item.select_one(".b_algoSlug_icon + p")
                    or item.select_one("p")
                )
                if not (title_tag and body_tag):
                    continue
                t = title_tag.get_text(strip=True)
                d = body_tag.get_text(strip=True)
                if d:
                    snippets.append(f"{t}: {d}")
                if len(snippets) >= max_results:
                    break

            if not snippets:
                return "未找到相关搜索结果（可能是页面结构变化或被反爬拦截）。"
            return self._join(snippets)
        except Exception as e:
            # 工具层必须自己吞掉异常，返回字符串而不是抛错，否则整个 ReAct 循环会崩
            return f"搜索时发生错误(HTML): {e}"

    def _join(self, snippets: list[str]) -> str:
        result = " | ".join(snippets)
        if len(result) > self.max_length:
            result = result[: self.max_length] + "..."
        return result

    # ---------- 参数声明：告诉模型该怎么调 ----------
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="str",
                description="搜索关键词，英文关键词通常结果更精确",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="int",
                description="返回结果条数，默认 3，最多 10",
                required=False,
                default=3,
            ),
        ]
