# 🏛️ 物理学史智能知识库 Agent

基于 **LangChain + RAG + Agent** 构建的物理学史知识库智能问答系统，支持对物理学家、重要理论、科学事件及发展历程的语义检索与多步推理问答。

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C.svg)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/vector%20store-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)](#-测试)

> 一个不只是"能问答"的知识库——它会在需要时**自己决定调用哪个工具**，去检索文献、拼时间线、理人物关系，甚至实时查诺贝尔奖官网。

---

## 📖 这个项目做了什么

通用大模型回答物理学史问题时有两个典型毛病：**一是会编**（幻觉，把不确定的年份和人物关系说得斩钉截铁），**二是信息旧**（训练数据截止后就不知道了）。

这个项目用两种手段解决：

| 问题 | 方案 |
|---|---|
| 会编 | **RAG 检索增强**——答案必须基于知识库文献生成，并给出可点开的来源引用 |
| 信息旧 | **Agent 工具调用**——诺贝尔奖数据实时从官方 API 拉取，不依赖模型记忆 |

---

## ✨ 核心功能

### 双模式问答

界面提供两种模式，一键切换：

| 模式 | 机制 | 适合 |
|---|---|---|
| **📖 RAG 模式** | LCEL 检索链：向量检索 → 拼上下文 → 生成 → 输出解析 | 事实性问题（"薛定谔方程是什么"） |
| **🔄 Agent 模式** | ReAct Agent 自主规划，按需调用工具 | 复杂分析（"普朗克到玻尔的理论演进脉络"） |

### 4 个 Agent 工具

Agent 会**根据问题自行判断**调用哪一个，也可以多步串联：

- **🔍 知识库检索** — 基于 FAISS 的语义搜索，支持中英文混合查询
- **📅 时间线生成** — 自动梳理科学史事件脉络
- **🕸️ 人物关系分析** — 分析师承、合作、学术论战等多维关系
- **🏅 诺贝尔奖查询** — 接入 Nobel Prize API，实时查询 125 届物理学奖

### 知识库

**133 篇**文献，两个来源：

- **7 篇手写专题** — 经典力学、相对论、量子力学、现代物理学四大板块（伽利略、牛顿、爱因斯坦、普朗克、玻尔、薛定谔等）
- **126 篇自动生成** — 调用 [Nobel Prize API v2.1](https://api.nobelprize.org/2.1/) 拉取全部 125 届诺贝尔物理学奖，自动生成结构化 Markdown 入库

---

## 🚀 快速开始

### 1. 克隆并安装

```bash
git clone https://github.com/2946994426-hue/physics-rag-agent.git
cd physics-rag-agent
pip install -e ".[dev]"
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入 [DeepSeek](https://platform.deepseek.com/) 的 API Key：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 项目通过 **OpenAI 兼容接口**调用 DeepSeek，因此把 `DEEPSEEK_BASE_URL` 换成任意兼容 OpenAI 协议的服务端点也可以直接跑。

### 3. 启动

```bash
streamlit run app.py
```

浏览器打开后，在**侧边栏点击「🔨 重建索引」**，首次会自动下载 BGE-M3 模型（约 2GB）并构建向量索引，完成后即可开始提问。

---

## 🏗️ 架构

```text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    │  RAG / Agent 模式 │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │   RAG 链路       │          │   Agent 编排     │
     │   LCEL Chain    │          │  create_agent   │
     └────────┬────────┘          └────────┬────────┘
              │                            │
              │                   ┌────────┴────────┐
              │                   ▼                 ▼
              │            ┌────────────┐   ┌─────────────┐
              │            │ 知识库检索  │   │ 诺贝尔奖查询 │
              │            │ 时间线生成  │   │ 人物关系分析 │
              │            └─────┬──────┘   └──────┬──────┘
              │                  │                 │
              ▼                  ▼                 ▼
     ┌─────────────────────────────────┐   ┌──────────────┐
     │   FAISS 向量库 (BGE-M3 1024维)   │   │ Nobel API    │
     └────────────────┬────────────────┘   └──────────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  133 篇知识库文献  │
            └───────────────────┘
```

### 目录结构

```text
physics-rag-agent/
├── app.py                       # Streamlit 主入口（双模式 UI）
├── src/physics_rag/
│   ├── config.py                # pydantic-settings 全局配置
│   ├── agent.py                 # Agent 编排（create_agent + 4 工具）
│   ├── rag_chain.py             # RAG 问答链（LCEL）+ 来源引用
│   ├── prompts.py               # Prompt 模板集中管理
│   ├── embeddings.py            # BGE-M3 Embedding 封装
│   ├── vectorstore.py           # FAISS 索引构建 / 持久化 / 检索
│   ├── splitter.py              # 中英文递归字符分割器
│   ├── nobel_fetcher.py         # Nobel Prize API 抓取与 Markdown 生成
│   ├── loaders/                 # 文档加载器
│   │   ├── pdf_loader.py        #   PDF (pypdf)
│   │   ├── markdown_loader.py   #   Markdown
│   │   └── web_loader.py        #   网页 (BeautifulSoup + markdownify)
│   └── tools/                   # Agent 工具集
│       ├── knowledge_search.py  #   知识库语义检索
│       ├── timeline.py          #   科学史时间线生成
│       ├── relationship.py      #   人物关系分析
│       └── nobel_query.py       #   诺贝尔奖实时查询
├── data/                        # 知识库（133 篇 Markdown）
├── tests/                       # pytest 测试
└── pyproject.toml
```

---

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| 编排框架 | LangChain 1.3（LCEL + `create_agent`） |
| 大模型 | DeepSeek Chat（OpenAI 兼容接口） |
| Embedding | BAAI/bge-m3（多语言，1024 维） |
| 向量库 | FAISS（CPU） |
| Web UI | Streamlit |
| 配置管理 | pydantic-settings |
| 文档解析 | pypdf / BeautifulSoup / markdownify |
| 测试 | pytest |

---

## 🔧 实现要点

几个开发过程中值得记录的问题和取舍：

**1. LangChain 1.3 的 API 迁移**

项目最初基于旧版 API 编写，适配 1.3 时踩了两个坑：
- `create_react_agent` 已被 `create_agent` 取代，返回的是 `CompiledStateGraph`，调用方式改为 `agent.invoke({"messages": [...]})`，取结果需要从 `messages` 列表里倒序找最后一条 AI 消息
- `openai_api_base` 参数更名为 `base_url`

**2. 中英文混合的分割策略**

纯英文的递归分割器按 `\n\n` / `.` 切分，处理中文文献时会把完整句子拦腰截断。分隔符列表补入了中文标点（`。`、`；`、`，`），并采用 `chunk_size=1024 / overlap=200` 的参数——chunk 偏大是因为物理学史文献段落语义完整度高，切太碎反而丢失上下文。

**3. Embedding 模型的选择**

选用 BGE-M3 而不是英文模型，核心原因是知识库同时包含中文手写文献和英文诺贝尔奖资料，需要**跨语言检索**能力（用中文问"爱因斯坦拿了哪年诺奖"要能召回英文条目）。

**4. 配置与密钥分离**

所有可调参数（模型、chunk 大小、top-k、路径）集中在 `config.py` 用 pydantic-settings 管理，优先级为 **环境变量 > .env > 默认值**。`.env` 已在 `.gitignore` 中排除，仓库里只保留 `.env.example`。

---

## 🧪 测试

```bash
pytest -q
```

```text
13 passed
├── test_loaders.py      4 passed    # PDF / Markdown / 网页加载器
├── test_splitter.py     4 passed    # 分割器边界情况
└── test_vectorstore.py  5 passed    # FAISS 构建 / 持久化 / 检索
```

---

## ⚠️ 已知限制

- **索引需手动构建** — 首次运行必须点侧边栏的「重建索引」，否则问答功能不可用（会给出明确提示）
- **Embedding 跑在 CPU 上** — 首次建索引较慢，且需下载约 2GB 模型
- **知识库规模有限** — 133 篇文献覆盖主要人物与理论，冷门细节可能检索不到；RAG 模式下遇到知识库外的问题会明确说明，不会硬编答案

## 🗺️ 后续计划

- [ ] 知识图谱可视化（人物关系用 Mermaid 渲染）
- [ ] 补充更多物理学教材与论文 PDF
- [ ] Docker 容器化部署
- [ ] 多用户会话隔离
- [ ] 检索效果评估集（Recall@k / 答案准确率）

---

## 📄 License

尚未指定。如需开源复用，建议补充 MIT 或其他许可证文件。
