"""工具3：人物关系分析 —— 分析物理学家之间的学术关系。"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import settings
from ..prompts import RELATIONSHIP_SYSTEM_PROMPT
from ..vectorstore import VectorStoreManager


def create_relationship_analysis_tool(vector_store: VectorStoreManager):
    """创建人物关系分析工具。

    Args:
        vector_store: 已加载索引的 VectorStoreManager。

    Returns:
        LangChain Tool 实例。
    """

    @tool
    def analyze_relationships(query: str) -> str:
        """分析物理学家之间的学术关系（师承、合作、理论继承、学术争论等）。

        Args:
            query: 分析请求，例如 "分析牛顿和爱因斯坦的学术关系"、\
"量子力学发展过程中玻尔和爱因斯坦的论战"、\
"伽利略对牛顿的影响"。

        Returns:
            结构化的关系分析报告，包含师承、合作、争论等维度。
        """
        # 1. 检索相关资料
        docs = vector_store.search(query, top_k=settings.top_k + 3)
        if not docs:
            return f"知识库中未找到与「{query}」相关的资料，无法进行分析。"

        context = "\n\n".join(
            f"[来源: {d.metadata.get('file_name', '?')}]\n{d.page_content}"
            for d in docs
        )

        # 2. 使用 LLM 生成关系分析
        llm = ChatOpenAI(
            model=settings.deepseek_model,
            openai_api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.5,
            max_tokens=2048,
            request_timeout=60,
        )

        prompt = RELATIONSHIP_SYSTEM_PROMPT.format(context=context)
        user_msg = f"请分析以下物理学人物/理论之间的关系：{query}"

        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ])

        return response.content

    return analyze_relationships
