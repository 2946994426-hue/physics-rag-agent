"""文档加载器模块 —— 统一的加载器工厂，支持 PDF / Markdown / 网页。"""

from pathlib import Path

from langchain_core.documents import Document

from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .web_loader import WebLoader


def load_documents(data_dir: str) -> list[Document]:
    """扫描 data 目录，按文件类型分发给对应加载器，返回所有文档。

    Args:
        data_dir: 知识库根目录路径。

    Returns:
        所有加载到的 LangChain Document 列表。
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_docs: list[Document] = []
    pdf_loader = PDFLoader()
    md_loader = MarkdownLoader()
    web_loader = WebLoader()

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            docs = pdf_loader.load(str(file_path))
        elif suffix in (".md", ".txt"):
            docs = md_loader.load(str(file_path))
        elif suffix == ".url":
            # .url 文件内容是一个网址，交给 web loader
            docs = web_loader.load(str(file_path))
        else:
            continue

        all_docs.extend(docs)

    return all_docs


__all__ = ["load_documents", "PDFLoader", "MarkdownLoader", "WebLoader"]
