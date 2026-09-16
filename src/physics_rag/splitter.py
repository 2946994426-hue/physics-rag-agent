"""文本切片模块 —— 针对中英文物理文本优化的递归字符分割器。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 分隔符优先级：从粗粒度到细粒度，中英文都覆盖
PHYSICS_SEPARATORS = [
    "\n\n",   # 段落分隔
    "\n",     # 行分隔
    "。",     # 中文句号
    ". ",     # 英文句号 + 空格
    "！",     # 中文感叹号
    "! ",     # 英文感叹号
    "？",     # 中文问号
    "? ",     # 英文问号
    "；",     # 中文分号
    "; ",     # 英文分号
    "，",     # 中文逗号
    ", ",     # 英文逗号
    "、",     # 中文顿号
    " ",      # 空格
    "",       # 字符级兜底
]


def create_splitter(
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """创建针对物理学文本优化的递归字符分割器。

    Args:
        chunk_size: 每个切片的最大字符数。
        chunk_overlap: 相邻切片之间的重叠字符数。

    Returns:
        配置好的 RecursiveCharacterTextSplitter 实例。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=PHYSICS_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Document]:
    """将文档列表切片为更小的 chunks。

    Args:
        docs: 原始文档列表。
        chunk_size: 切片大小。
        chunk_overlap: 切片重叠。

    Returns:
        切片后的文档列表。
    """
    splitter = create_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
