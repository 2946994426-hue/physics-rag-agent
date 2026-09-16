"""诺贝尔奖 API 数据抓取器 —— 拉取物理学奖数据并生成 Markdown 知识文件。"""

import json
import time
from pathlib import Path

import requests

NOBEL_API = "https://api.nobelprize.org/2.1/nobelPrizes"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_physics_prizes() -> list[dict]:
    """从诺贝尔 API 获取全部物理学奖数据。"""
    all_prizes = []
    limit = 50
    offset = 0

    while True:
        resp = requests.get(
            NOBEL_API,
            params={
                "nobelPrizeCategory": "phy",
                "limit": limit,
                "offset": offset,
                "sort": "asc",
            },
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        prizes = data.get("nobelPrizes", [])
        if not prizes:
            break
        all_prizes.extend(prizes)
        offset += limit
        time.sleep(0.3)  # 温和限速

    return all_prizes


def _format_prize(prize: dict) -> str:
    """将单条诺贝尔奖数据格式化为 Markdown。"""
    year = prize["awardYear"]
    category = prize.get("category", {}).get("en", "Physics")
    prize_amount = prize.get("prizeAmountAdjusted", 0)
    date = prize.get("dateAwarded", "")

    laureates = prize.get("laureates", [])
    names = [l.get("knownName", {}).get("en", "未知") for l in laureates]
    names_str = "、".join(names)

    lines = [
        f"# {year} 年诺贝尔物理学奖",
        "",
        f"**获奖者**：{names_str}",
        f"**颁奖日期**：{date}",
        f"**奖金**：{prize_amount:,} SEK（调整后）",
        "",
        "## 获奖理由",
        "",
    ]

    for l in laureates:
        name = l.get("knownName", {}).get("en", "未知")
        full_name = l.get("fullName", {}).get("en", "")
        motivation = l.get("motivation", {}).get("en", "暂无")
        portion = l.get("portion", "1")
        lines.append(f"### {name}")
        if full_name and full_name != name:
            lines.append(f"全名：{full_name}")
        lines.append(f"份额：{portion}")
        lines.append(f"获奖理由：{motivation}")
        lines.append("")

    lines.extend([
        "---",
        f"*数据来源：Nobel Prize API ({year})*",
    ])

    return "\n".join(lines)


def fetch_and_save(output_dir: str = "./data/nobel_prizes") -> int:
    """拉取全量诺贝尔物理学奖数据并保存为 Markdown 文件。

    Args:
        output_dir: 输出目录。

    Returns:
        保存的文件数量。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prizes = fetch_physics_prizes()
    count = 0

    for prize in prizes:
        year = prize["awardYear"]
        markdown = _format_prize(prize)
        file_path = out / f"nobel_{year}.md"
        file_path.write_text(markdown, encoding="utf-8")
        count += 1

    # 生成索引文件
    _generate_index(prizes, out)

    return count


def _generate_index(prizes: list[dict], out_dir: Path) -> None:
    """生成诺贝尔物理学奖总索引。"""
    lines = [
        "# 诺贝尔物理学奖总览",
        "",
        f"共 {len(prizes)} 届（1901 - 至今）",
        "",
        "| 年份 | 获奖者 | 获奖理由 |",
        "|------|--------|----------|",
    ]

    for prize in prizes:
        year = prize["awardYear"]
        laureates = prize.get("laureates", [])
        names = "、".join(l.get("knownName", {}).get("en", "?") for l in laureates)
        motivation = ""
        if laureates:
            motivation = laureates[0].get("motivation", {}).get("en", "")[:80]

        lines.append(f"| {year} | {names} | {motivation}... |")

    (out_dir / "nobel_index.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    n = fetch_and_save()
    print(f"✅ 已抓取 {n} 条诺贝尔物理学奖记录")
