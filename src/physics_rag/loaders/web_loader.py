"""网页文档加载器。"""

import re

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdownify import markdownify as md


class WebLoader:
    """抓取网页 URL 并转换为 Markdown 文本。"""

    def load(self, source: str) -> list[Document]:
        """加载网页。

        Args:
            source: 可以是 URL 字符串，也可以是 .url 文件路径。
                    .url 文件第一行应为 http(s):// 格式的 URL。

        Returns:
            包含单个 Document 的列表。
        """
        url = self._resolve_url(source)
        title, content = self._fetch(url)
        doc = Document(
            page_content=content,
            metadata={
                "source": url,
                "title": title,
                "file_type": "web",
            },
        )
        return [doc]

    def _resolve_url(self, source: str) -> str:
        """从 source 中解析出 URL。"""
        path = Path(source) if not source.startswith("http") else None
        stripped = source.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            return stripped
        if path and path.suffix == ".url":
            content = path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("http://") or line.startswith("https://"):
                    return line
        raise ValueError(f"无法从 source 解析出有效 URL: {source}")

    def _fetch(self, url: str, timeout: int = 30) -> tuple[str, str]:
        """抓取网页并返回 (title, markdown_content)。"""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # 提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 移除无用的标签
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # 尝试找到正文区域
        main = soup.find("main") or soup.find("article") or soup.body
        if main is None:
            main = soup

        markdown = md(str(main), heading_style="ATX", strip=["img"])
        # 压缩多余空行
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        return title, markdown.strip()


from pathlib import Path  # noqa: E402
