"""PDF 文档加载器。"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """使用 PyPDF 解析 PDF 文件，逐页生成 Document。"""

    def load(self, file_path: str) -> list[Document]:
        """加载 PDF 文件并返回各页的 Document。

        Args:
            file_path: PDF 文件的绝对路径。

        Returns:
            Document 列表，每个元素对应 PDF 的一页。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        loader = PyPDFLoader(str(path))
        docs = loader.load()

        # 补充文件来源 metadata
        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(path)
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = "pdf"

        return docs
