"""Streamlit Web UI —— 物理学史智能知识库 Agent 系统。"""

import sys
from pathlib import Path

# 将 src 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from physics_rag.agent import run_agent
from physics_rag.config import settings
from physics_rag.embeddings import create_embeddings
from physics_rag.loaders import load_documents
from physics_rag.rag_chain import ask
from physics_rag.splitter import split_documents
from physics_rag.vectorstore import VectorStoreManager

# ── 页面配置 ──────────────────────────────────────────────────

st.set_page_config(
    page_title="物理学史智能知识库",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 初始化 Session State ─────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

if "mode" not in st.session_state:
    st.session_state.mode = "RAG 模式"

# ── 辅助函数 ──────────────────────────────────────────────────


def init_vector_store() -> VectorStoreManager:
    """初始化向量库并尝试加载已有索引。"""
    embeddings = create_embeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    store = VectorStoreManager(settings.resolved_faiss_path, embeddings)
    return store


def build_index(store: VectorStoreManager) -> int:
    """从数据目录构建 FAISS 索引。返回索引的文档数。"""
    with st.spinner("正在加载文档..."):
        docs = load_documents(settings.resolved_data_dir)
    st.info(f"加载了 {len(docs)} 个原始文档")

    with st.spinner("正在切片..."):
        chunks = split_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    st.info(f"切片为 {len(chunks)} 个文本块")

    with st.spinner("正在构建 FAISS 索引（首次运行会下载 embedding 模型，约 2GB）..."):
        store.build_index(chunks)
    return store.count()


# ── 侧边栏 ────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏛️ 物理学史智能知识库")
    st.caption("Powered by RAG + Agent + DeepSeek")

    st.divider()

    # ── 数据管理 ──
    st.subheader("📂 数据管理")

    if st.button("🔨 重建索引", use_container_width=True):
        store = init_vector_store()
        count = build_index(store)
        st.session_state.vector_store = store
        st.session_state.index_loaded = True
        st.success(f"索引构建成功！共 {count} 条记录")

    # 尝试自动加载已有索引
    if not st.session_state.index_loaded:
        store = init_vector_store()
        if store.load_index():
            st.session_state.vector_store = store
            st.session_state.index_loaded = True

    if st.session_state.index_loaded:
        store = st.session_state.vector_store
        st.metric("已索引文档块", store.count())
    else:
        st.warning("⚠️ 索引尚未构建，请点击「重建索引」")

    st.divider()

    # ── 模式选择 ──
    st.subheader("⚙️ 模式选择")
    mode = st.radio(
        "选择问答模式",
        ["RAG 模式", "Agent 模式"],
        help=(
            "RAG 模式：快速检索+问答，适合事实性问题\n\n"
            "Agent 模式：可调用知识检索、时间线生成、人物关系分析、诺贝尔奖查询等工具，适合复杂分析"
        ),
    )
    st.session_state.mode = mode

    st.divider()

    # ── 示例问题 ──
    st.subheader("📊 示例问题")

    example_questions = {
        "RAG 模式": [
            "牛顿和爱因斯坦对经典力学的发展有什么区别？",
            "量子力学的发展经历了哪些阶段？",
            "伽利略对现代科学方法有什么贡献？",
            "2024年诺贝尔物理学奖颁给了谁？",
        ],
        "Agent 模式": [
            "帮我分析量子力学从普朗克到薛定谔的发展路线",
            "分析牛顿、伽利略和爱因斯坦之间的学术传承关系",
            "请给出相对论从狭义到广义的发展时间线",
            "爱因斯坦获得了哪年的诺贝尔奖？获奖理由是什么？",
        ],
    }

    for q in example_questions.get(mode, example_questions["RAG 模式"]):
        if st.button(q, use_container_width=True):
            st.session_state.current_question = q

    st.divider()

    # ── 配置状态 ──
    st.subheader("🔧 配置状态")
    st.write(f"- LLM: {settings.deepseek_model}")
    st.write(f"- Embedding: {settings.embedding_model.split('/')[-1]}")
    st.write(f"- Top-K: {settings.top_k}")
    st.write(f"- Chunk: {settings.chunk_size} / Overlap: {settings.chunk_overlap}")

# ── 主区域：对话框 ────────────────────────────────────────────

st.title("🏛️ 物理学史智能知识库 Agent")
st.caption("基于 RAG + LangChain Agent，探索物理学的伟大历程")

# 显示当前模式
mode_badge = "🔄 Agent 模式" if st.session_state.mode == "Agent 模式" else "📖 RAG 模式"
st.info(f"当前模式：**{mode_badge}** — 可直接输入问题或点击左侧示例")

st.divider()

# 显示对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考资料"):
                for s in msg["sources"]:
                    category_tag = f" `[{s['category']}]`" if s.get("category") else ""
                    st.markdown(f"**{s['file']}**{category_tag}")
                    st.caption(s["snippet"])

# 用户输入
user_input = st.chat_input("请输入您想了解的物理学史问题...")

# 处理预设问题（从侧边栏按钮触发）
if "current_question" in st.session_state and st.session_state.current_question:
    user_input = st.session_state.current_question
    st.session_state.current_question = None

if user_input:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成回答
    if not st.session_state.index_loaded:
        with st.chat_message("assistant"):
            st.error("⚠️ 索引尚未构建！请先在侧边栏点击「🔨 重建索引」")
    else:
        store = st.session_state.vector_store
        with st.chat_message("assistant"):
            try:
                if st.session_state.mode == "RAG 模式":
                    with st.spinner("正在检索知识库..."):
                        result = ask(user_input, store)
                    st.markdown(result["answer"])
                    sources = result.get("sources", [])
                else:
                    with st.spinner("Agent 正在思考..."):
                        answer = run_agent(user_input, store)
                    st.markdown(answer)
                    sources = []
            except Exception as e:
                error_msg = f"⚠️ 出错了：{e}"
                st.error(error_msg)
                if "RAG 模式" in st.session_state.mode:
                    result = {"answer": error_msg, "sources": []}
                else:
                    answer = error_msg
                sources = []

        # 保存回答
        content = (
            result["answer"]
            if st.session_state.mode == "RAG 模式"
            else answer
        )
        msg_data = {"role": "assistant", "content": content}
        if sources:
            msg_data["sources"] = sources
            with st.expander("📚 参考资料"):
                for s in sources:
                    category_tag = f" `[{s['category']}]`" if s.get("category") else ""
                    st.markdown(f"**{s['file']}**{category_tag}")
                    st.caption(s["snippet"])

        st.session_state.messages.append(msg_data)

# ── 底部操作 ──
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("🔝 回到顶部", use_container_width=True):
        pass  # Streamlit 自动回到顶部
