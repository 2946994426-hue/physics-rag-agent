# 🏛️ 物理学史智能知识库 Agent 系统

基于 LangChain + RAG + Agent 构建的物理学历史资料智能知识库系统，支持对物理学家、重要理论、科学事件及发展历程的智能检索与问答。

## ✨ 功能特性

- 📖 **RAG 问答**：基于 FAISS 向量检索 + DeepSeek 大模型，从物理学史知识库中检索并生成答案
- 🤖 **Agent 模式**：支持多工具调用 —— 知识检索、时间线生成、人物关系分析
- 🌐 **Streamlit Web UI**：美观的对话式交互界面
- 📂 **多格式支持**：PDF、Markdown、网页内容加载
- 🔍 **语义搜索**：BGE-M3 多语言嵌入模型，支持中英文混合检索

## 🚀 快速开始

### 1. 环境准备

```bash
# Python >= 3.11
pip install -e .
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 3. 构建知识库索引

```bash
# 首次运行需要下载 BGE-M3 模型（约 2GB）
python -c "
from physics_rag.embeddings import create_embeddings
from physics_rag.loaders import load_documents
from physics_rag.splitter import split_documents
from physics_rag.vectorstore import VectorStoreManager
from physics_rag.config import settings

embeddings = create_embeddings(device='cpu')
docs = load_documents(settings.data_dir)
chunks = split_documents(docs)
store = VectorStoreManager(settings.faiss_index_path, embeddings)
store.build_index(chunks)
print(f'索引完成！共 {store.count()} 条记录')
"
```

### 4. 启动 Web UI

```bash
streamlit run app.py
```

## 📁 项目结构

```
physics-rag-agent/
├── app.py                  # Streamlit 主入口
├── src/physics_rag/
│   ├── config.py           # 配置管理
│   ├── loaders/            # 文档加载器
│   ├── splitter.py         # 文本切片
│   ├── embeddings.py       # Embedding 模型
│   ├── vectorstore.py      # FAISS 向量库
│   ├── rag_chain.py        # RAG 问答链
│   ├── agent.py            # Agent 编排
│   ├── tools/              # Agent 工具集
│   │   ├── knowledge_search.py  # 知识库检索
│   │   ├── timeline.py         # 时间线生成
│   │   └── relationship.py     # 人物关系分析
│   └── prompts.py          # Prompt 模板
├── data/physics_history_db/ # 知识库
│   ├── classical_physics/   # 经典物理学
│   ├── relativity/          # 相对论
│   ├── quantum/             # 量子力学
│   └── modern_physics/      # 现代物理学
└── tests/                   # 测试
```

## 🛠️ 技术栈

- **框架**: LangChain (RAG + Agent 编排)
- **LLM**: DeepSeek Chat API
- **Embedding**: BAAI/bge-m3 (多语言)
- **向量库**: FAISS
- **UI**: Streamlit
