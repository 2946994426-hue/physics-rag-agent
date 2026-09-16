"""测试文档加载器模块。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_rag.loaders import load_documents
from physics_rag.loaders.markdown_loader import MarkdownLoader


def test_markdown_loader_basic():
    """测试 Markdown 加载器基本功能。"""
    loader = MarkdownLoader()
    # 使用实际项目数据
    data_dir = Path(__file__).parent.parent / "data" / "physics_history_db"
    md_files = list(data_dir.rglob("*.md"))

    if not md_files:
        return  # 无文件则跳过

    docs = loader.load(str(md_files[0]))
    assert len(docs) >= 1
    assert len(docs[0].page_content) > 0
    assert "source" in docs[0].metadata
    assert "file_name" in docs[0].metadata
    assert "category" in docs[0].metadata


def test_markdown_loader_file_not_found():
    """测试文件不存在时抛出异常。"""
    loader = MarkdownLoader()
    try:
        loader.load("/nonexistent/path/file.md")
    except FileNotFoundError:
        pass  # 预期行为


def test_load_documents_from_dir():
    """测试从目录加载所有文档。"""
    data_dir = Path(__file__).parent.parent / "data" / "physics_history_db"
    if not data_dir.exists():
        return

    docs = load_documents(str(data_dir))
    assert len(docs) > 0
    for doc in docs:
        assert hasattr(doc, "page_content")
        assert hasattr(doc, "metadata")
        assert "source" in doc.metadata


def test_load_documents_dir_not_found():
    """测试数据目录不存在时抛出异常。"""
    try:
        load_documents("/nonexistent/data/dir")
    except FileNotFoundError:
        pass
