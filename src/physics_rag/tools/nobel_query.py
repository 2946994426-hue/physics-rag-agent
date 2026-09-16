"""工具4：诺贝尔奖查询 —— 实时查询诺贝尔物理学奖数据。"""

import requests
from langchain_core.tools import tool


NOBEL_API = "https://api.nobelprize.org/2.1/nobelPrizes"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _search_nobel(query: str) -> str:
    """查询诺贝尔 API 并返回格式化结果。"""
    # 尝试从查询中提取年份
    params = {"nobelPrizeCategory": "phy", "limit": 300, "sort": "asc"}
    resp = requests.get(NOBEL_API, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    prizes = resp.json().get("nobelPrizes", [])

    q_lower = query.lower()
    results = []

    for prize in prizes:
        year = prize["awardYear"]
        laureates = prize.get("laureates", [])
        all_text = year

        for l in laureates:
            name_en = l.get("knownName", {}).get("en", "")
            full_en = l.get("fullName", {}).get("en", "")
            motivation = l.get("motivation", {}).get("en", "")
            all_text += " " + name_en + " " + full_en + " " + motivation

        # 匹配查询
        if any(kw in all_text.lower() for kw in q_lower.split()) or q_lower in all_text.lower():
            results.append(prize)

    if not results:
        # 返回最近的几届
        results = prizes[-5:]

    # 格式化
    lines = []
    for prize in results[:10]:
        year = prize["awardYear"]
        laureates = prize.get("laureates", [])
        names = [l.get("knownName", {}).get("en", "?") for l in laureates]
        motivations = []
        for l in laureates:
            m = l.get("motivation", {}).get("en", "")
            if m:
                motivations.append(f"{l.get('knownName', {}).get('en', '?')}: {m}")

        lines.append(f"## {year} 年")
        lines.append(f"获奖者：{'、'.join(names)}")
        for m in motivations:
            lines.append(f"- {m}")
        lines.append("")

    return "\n".join(lines) if lines else "未找到相关诺贝尔物理学奖信息。"


def create_nobel_query_tool():
    """创建诺贝尔奖查询工具。

    Returns:
        LangChain Tool 实例。
    """

    @tool
    def nobel_query(query: str) -> str:
        """查询诺贝尔物理学奖信息。可以按年份、获奖者姓名、获奖领域等搜索。

        示例查询：
        - "2023年诺贝尔物理学奖"
        - "爱因斯坦获得诺贝尔奖"
        - "量子力学相关的诺贝尔奖"
        - "近年来诺贝尔物理学奖获奖者"

        Args:
            query: 自然语言查询。

        Returns:
            格式化的诺贝尔奖查询结果。
        """
        return _search_nobel(query)

    return nobel_query
