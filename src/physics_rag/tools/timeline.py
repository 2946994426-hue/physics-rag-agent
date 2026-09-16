"""工具2：时间线生成 —— 为物理学主题生成科学发展时间线。"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import settings
from ..prompts import TIMELINE_SYSTEM_PROMPT
from ..vectorstore import VectorStoreManager


def create_timeline_tool(vector_store: VectorStoreManager):
    """创建时间线生成工具。

    Args:
        vector_store: 已加载索引的 VectorStoreManager。

    Returns:
        LangChain Tool 实例。
    """

    @tool
    def generate_timeline(topic: str) -> str:
        """为指定的物理学主题生成科学发展的详细时间线。

        Args:
            topic: 物理学主题名称，例如 "量子力学发展史"、\
"相对论的建立"、"经典力学的发展"。

        Returns:
            按年份排序的时间线，Markdown 格式，包含关键事件、人物和简要说明。
        """
        # 1. 检索相关资料
        docs = vector_store.search(topic, top_k=settings.top_k + 3)
        if not docs:
            return f"知识库中未找到与「{topic}」相关的资料，无法生成时间线。"

        context = "\n\n".join(
            f"[来源: {d.metadata.get('file_name', '?')}]\n{d.page_content}"
            for d in docs
        )

        # 2. 使用 LLM 生成时间线
        llm = ChatOpenAI(
            model=settings.deepseek_model,
            openai_api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,  # 低温度获得更精确的时间线
            max_tokens=2048,
            request_timeout=60,
        )

        prompt = TIMELINE_SYSTEM_PROMPT.format(context=context)
        user_msg = f"请为「{topic}」生成详细的发展时间线。"

        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ])

        return response.content

    return generate_timeline
