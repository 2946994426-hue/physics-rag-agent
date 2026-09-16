"""工具1：知识库检索 —— 在物理学史知识库中进行语义搜索。"""

from langchain_core.tools import tool

from ..config import settings
from ..vectorstore import VectorStoreManager


def create_knowledge_search_tool(vector_store: VectorStoreManager):
    """创建知识库检索工具（闭包绑定向量库实例）。

    Args:
        vector_store: 已加载索引的 VectorStoreManager。

    Returns:
        LangChain Tool 实例。
    """

    @tool
    def knowledge_search(query: str) -> str:
        """在物理学史知识库中搜索与查询相关的资料。

        Args:
            query: 自然语言搜索查询，例如 "牛顿万有引力定律的发现过程"
                   或 "量子力学的发展阶段"。

        Returns:
            格式化的参考资料，包含文档内容和来源标注。
        """
        docs = vector_store.search(query, top_k=settings.top_k)

        if not docs:
            return "未在知识库中找到相关资料。"

        blocks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知来源")
            category = doc.metadata.get("category", "")
            label = f"[{i}] 📄 {source}"
            if category:
                label += f" [{category}]"
            blocks.append(f"{label}\n{doc.page_content}\n")

        return "\n".join(blocks)

    return knowledge_search
