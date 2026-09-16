"""FAISS 向量库封装 —— 索引构建、持久化、检索。"""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """管理 FAISS 向量索引的完整生命周期。"""

    def __init__(self, persist_path: str, embeddings: Embeddings):
        """初始化向量库管理器。

        Args:
            persist_path: FAISS 索引持久化目录。
            embeddings: 嵌入模型实例。
        """
        self.persist_path = persist_path
        self.embeddings = embeddings
        self._store: FAISS | None = None

    @property
    def store(self) -> FAISS | None:
        """获取当前加载的 FAISS 向量库实例。"""
        return self._store

    def build_index(self, docs: list[Document]) -> None:
        """从文档列表构建 FAISS 索引。

        Args:
            docs: 文档列表（已切片）。
        """
        self._store = FAISS.from_documents(docs, self.embeddings)
        self._ensure_dir()
        self._store.save_local(self.persist_path)

    def load_index(self) -> bool:
        """从磁盘加载现有索引。

        Returns:
            成功加载返回 True，索引不存在返回 False。
        """
        if not os.path.exists(os.path.join(self.persist_path, "index.faiss")):
            return False

        self._store = FAISS.load_local(
            self.persist_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """语义相似度检索。

        Args:
            query: 查询字符串。
            top_k: 返回的文档数量。

        Returns:
            相似度降序排列的文档列表。

        Raises:
            RuntimeError: 索引未加载时调用。
        """
        if self._store is None:
            raise RuntimeError("索引未加载，请先调用 build_index() 或 load_index()")

        return self._store.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 5):
        """将 FAISS 向量库转为 LangChain Retriever 接口。

        Args:
            top_k: 检索数量。

        Returns:
            LangChain Retriever 实例。

        Raises:
            RuntimeError: 索引未加载。
        """
        if self._store is None:
            raise RuntimeError("索引未加载，请先调用 build_index() 或 load_index()")
        return self._store.as_retriever(search_kwargs={"k": top_k})

    def count(self) -> int:
        """返回已索引的文档数。"""
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def is_loaded(self) -> bool:
        """检查索引是否已加载。"""
        return self._store is not None

    def _ensure_dir(self) -> None:
        """确保索引目录存在。"""
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
