"""Embedding 模型封装 —— 直接用 SentenceTransformer 避开兼容问题。"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class BGEEmbeddings(Embeddings):
    """BGE-M3 嵌入模型，直接封装 SentenceTransformer，兼容 LangChain 接口。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档文本。"""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """嵌入查询文本。"""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


def create_embeddings(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> BGEEmbeddings:
    """创建 BGE Embeddings 实例。

    Args:
        model_name: HuggingFace 模型名。
        device: 运行设备。

    Returns:
        BGEEmbeddings 实例，实现 LangChain Embeddings 接口。
    """
    return BGEEmbeddings(model_name=model_name, device=device)
