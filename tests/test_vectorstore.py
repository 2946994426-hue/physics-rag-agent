"""测试 FAISS 向量库模块。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document

from physics_rag.config import settings
from physics_rag.embeddings import create_embeddings
from physics_rag.vectorstore import VectorStoreManager


def test_vector_store_init(temp_faiss_dir):
    """测试向量库初始化。"""
    embeddings = create_embeddings(device="cpu")
    store = VectorStoreManager(temp_faiss_dir, embeddings)
    assert not store.is_loaded()
    assert store.count() == 0


def test_build_and_search(temp_faiss_dir):
    """测试索引构建和检索。"""
    embeddings = create_embeddings(device="cpu")
    store = VectorStoreManager(temp_faiss_dir, embeddings)

    docs = [
        Document(
            page_content="阿尔伯特·爱因斯坦在1905年提出了狭义相对论。",
            metadata={"source": "einstein.md", "category": "relativity"},
        ),
        Document(
            page_content="艾萨克·牛顿在1687年发表了万有引力定律。",
            metadata={"source": "newton.md", "category": "classical_physics"},
        ),
        Document(
            page_content="马克斯·普朗克在1900年提出了量子假说。",
            metadata={"source": "planck.md", "category": "quantum"},
        ),
    ]

    store.build_index(docs)
    assert store.is_loaded()
    assert store.count() == 3

    # 搜索
    results = store.search("狭义相对论", top_k=2)
    assert len(results) == 2
    # 第一个应该是最相关的
    assert "爱因斯坦" in results[0].page_content or "相对论" in results[0].page_content


def test_save_and_load(temp_faiss_dir):
    """测试索引持久化和加载。"""
    embeddings = create_embeddings(device="cpu")

    # 构建并保存
    store1 = VectorStoreManager(temp_faiss_dir, embeddings)
    docs = [
        Document(
            page_content="量子力学是描述微观粒子运动规律的理论。",
            metadata={"source": "quantum.md"},
        ),
    ]
    store1.build_index(docs)
    assert store1.count() == 1

    # 新建 store 并加载
    store2 = VectorStoreManager(temp_faiss_dir, embeddings)
    assert store2.load_index()
    assert store2.count() == 1

    results = store2.search("量子力学", top_k=1)
    assert len(results) == 1
    assert "量子力学" in results[0].page_content


def test_search_without_index(temp_faiss_dir):
    """测试未加载索引时搜索抛出异常。"""
    embeddings = create_embeddings(device="cpu")
    store = VectorStoreManager(temp_faiss_dir, embeddings)

    try:
        store.search("test", top_k=1)
        assert False, "应该抛出 RuntimeError"
    except RuntimeError:
        pass  # 预期行为


def test_as_retriever(temp_faiss_dir):
    """测试转为 LangChain Retriever。"""
    embeddings = create_embeddings(device="cpu")
    store = VectorStoreManager(temp_faiss_dir, embeddings)

    docs = [Document(page_content="测试内容", metadata={"source": "test.md"})]
    store.build_index(docs)

    retriever = store.as_retriever(top_k=1)
    results = retriever.invoke("测试")
    assert len(results) == 1
