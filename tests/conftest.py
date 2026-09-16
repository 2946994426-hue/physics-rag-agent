"""pytest 全局 fixtures。"""

import os
import tempfile
from pathlib import Path

import pytest

# 确保不使用真实 API key
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-placeholder")


@pytest.fixture
def sample_docs_dir():
    """创建包含示例 Markdown 文件的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sub = root / "classical_physics"
        sub.mkdir()

        (sub / "test.md").write_text(
            "# 测试文档\n\n这是一个测试段落。\n\n这是第二个段落。",
            encoding="utf-8",
        )

        yield str(root)


@pytest.fixture
def temp_faiss_dir():
    """创建临时 FAISS 索引目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_data_dir():
    """创建包含测试数据的临时数据目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sub = root / "test_category"
        sub.mkdir(parents=True)
        (sub / "doc1.md").write_text(
            "# 文档1\n\n这是第一个测试文档的内容。",
            encoding="utf-8",
        )
        (sub / "doc2.md").write_text(
            "# 文档2\n\n这是第二个测试文档，包含更多内容。",
            encoding="utf-8",
        )
        yield str(root)
