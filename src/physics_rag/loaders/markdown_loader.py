"""Markdown / 纯文本文档加载器。"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class MarkdownLoader:
    """加载 .md 和 .txt 文件，每文件生成一个 Document。"""

    def load(self, file_path: str) -> list[Document]:
        """加载文本文件。

        Args:
            file_path: 文件绝对路径。

        Returns:
            包含单个 Document 的列表。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()

        # 补充 metadata
        for doc in docs:
            doc.metadata["source"] = str(path)
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = path.suffix.lstrip(".")

            # 从目录结构中提取分类信息
            parent_dir = path.parent.name
            if parent_dir and parent_dir != "physics_history_db":
                doc.metadata["category"] = parent_dir

        return docs
