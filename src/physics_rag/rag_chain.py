"""RAG 问答链 —— 使用 LangChain LCEL 构建端到端问答管道。"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import settings
from .prompts import PHYSICS_RAG_SYSTEM_PROMPT, PHYSICS_RAG_USER_PROMPT
from .vectorstore import VectorStoreManager


def _format_docs(docs) -> str:
    """将检索到的文档列表格式化为上下文字符串。"""
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", doc.metadata.get("source", "未知来源"))
        category = doc.metadata.get("category", "")
        header = f"[参考 {i}] 📄 {source}"
        if category:
            header += f" ({category})"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n" + "─" * 60 + "\n\n".join(blocks)


def create_llm() -> ChatOpenAI:
    """创建 DeepSeek LLM 客户端（OpenAI 兼容 API）。"""
    return ChatOpenAI(
        model=settings.deepseek_model,
        openai_api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.7,
        max_tokens=2048,
        request_timeout=60,
    )


def build_rag_chain(vector_store: VectorStoreManager):
    """构建 RAG 问答链。

    Args:
        vector_store: 已加载索引的 VectorStoreManager 实例。

    Returns:
        可调用的 LangChain Runnable 链。

    Raises:
        RuntimeError: 向量库未加载。
    """
    if not vector_store.is_loaded():
        raise RuntimeError("向量库未加载，请先执行索引构建")

    llm = create_llm()
    retriever = vector_store.as_retriever(top_k=settings.top_k)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PHYSICS_RAG_SYSTEM_PROMPT),
        ("human", PHYSICS_RAG_USER_PROMPT),
    ])

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str, vector_store: VectorStoreManager) -> dict:
    """执行一次 RAG 问答。

    Args:
        question: 用户问题。
        vector_store: 已加载的向量库。

    Returns:
        {"answer": str, "sources": list[dict]} — 包含答案和引用来源。
    """
    chain = build_rag_chain(vector_store)

    # 先检索来源（独立获取，用于展示）
    docs = vector_store.search(question, top_k=settings.top_k)

    # 生成回答
    answer = chain.invoke(question)

    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("file_name", doc.metadata.get("source", ""))
        if source and source not in seen:
            seen.add(source)
            sources.append({
                "file": source,
                "category": doc.metadata.get("category", ""),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

    return {"answer": answer, "sources": sources}
