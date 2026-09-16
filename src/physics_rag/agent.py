"""Agent 编排 —— 使用 LangChain 1.3+ create_agent API 整合工具集。"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import settings
from .prompts import PHYSICS_AGENT_PROMPT
from .tools import (
    create_knowledge_search_tool,
    create_nobel_query_tool,
    create_relationship_analysis_tool,
    create_timeline_tool,
)
from .vectorstore import VectorStoreManager


def create_physics_agent(vector_store: VectorStoreManager):
    """创建物理学研究助手 Agent（LangChain 1.3+ 新 API）。

    Args:
        vector_store: 已加载索引的向量库。

    Returns:
        CompiledStateGraph 实例。
    """
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        openai_api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.5,
        max_tokens=2048,
        request_timeout=60,
    )

    # 创建工具集
    tools = [
        create_knowledge_search_tool(vector_store),
        create_timeline_tool(vector_store),
        create_relationship_analysis_tool(vector_store),
        create_nobel_query_tool(),
    ]

    # LangChain 1.3+ 新 API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=PHYSICS_AGENT_PROMPT,
    )

    return agent


def run_agent(query: str, vector_store: VectorStoreManager) -> str:
    """执行一次 Agent 查询。

    Args:
        query: 用户问题。
        vector_store: 已加载索引的向量库。

    Returns:
        Agent 的完整回答。
    """
    agent = create_physics_agent(vector_store)

    # 新 API：invoke 接受 {"messages": [...]} 格式
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}],
    })

    # 提取最后一条 AI 消息
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content

    return "Agent 未能生成回答，请重试。"
