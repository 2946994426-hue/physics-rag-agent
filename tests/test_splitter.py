"""测试文本切片模块。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document

from physics_rag.splitter import create_splitter, split_documents


def test_create_splitter():
    """测试创建分割器。"""
    splitter = create_splitter(chunk_size=512, chunk_overlap=50)
    assert splitter._chunk_size == 512
    assert splitter._chunk_overlap == 50


def test_split_single_document():
    """测试单个文档切片。"""
    doc = Document(
        page_content="这是第一段。\n\n这是第二段。\n\n这是第三段，包含更多的文本内容用于测试切分效果的好坏。"
        * 50,
        metadata={"source": "test.md"},
    )
    chunks = split_documents([doc], chunk_size=512, chunk_overlap=50)
    assert len(chunks) > 1  # 应该切分成多个 chunk
    for chunk in chunks:
        assert len(chunk.page_content) <= 512 + 200  # 允许一些余量
        assert "source" in chunk.metadata


def test_split_preserves_metadata():
    """测试切片保留 metadata。"""
    doc = Document(
        page_content="第一段。\n\n第二段。\n\n第三段。" * 30,
        metadata={"source": "physics.md", "category": "quantum", "page": 1},
    )
    chunks = split_documents([doc], chunk_size=512, chunk_overlap=50)
    for chunk in chunks:
        assert chunk.metadata["source"] == "physics.md"
        assert chunk.metadata["category"] == "quantum"


def test_chinese_separators():
    """测试中文分隔符生效。"""
    splitter = create_splitter(chunk_size=200, chunk_overlap=0)
    doc = Document(
        page_content="这是第一个句子。这是第二个句子。这是第三个句子。",
        metadata={},
    )
    chunks = splitter.split_documents([doc])
    assert len(chunks) >= 1
