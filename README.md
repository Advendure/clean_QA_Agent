# clean_QA_智能客服

基于 RAG（检索增强生成）技术的扫地机器人智能客服系统，支持知识库问答、使用报告生成等功能。

## 功能特性

- **智能问答**：基于 RAG 技术，从扫地机器人知识库中检索相关资料进行回答
- **混合检索**：BM25 关键词检索 + 稠密向量检索 + RRF 融合 + ReRanker 精排
- **智能体对话**：基于 ReAct 框架的智能体，可调用多种工具完成复杂任务
- **使用报告**：根据用户使用数据自动生成个性化使用报告
- **流式输出**：支持对话内容流式展示，提升用户体验

## 技术栈

- **智能体框架**：LangChain
- **前端界面**：Streamlit
- **大语言模型**：通义千问 qwen3-max
- **Embedding 模型**：DashScope text-embedding-v4
- **向量数据库**：ChromaDB
- **重排序**：qwen3-rerank

## 项目结构

```
clean_QA_智能客服/
├── app.py                  # Streamlit 主入口
├── agent/                  # 智能体模块
│   ├── react_agent.py      # ReAct 智能体实现
│   └── tools/              # 工具定义
│       ├── agent_tools.py  # 业务工具
│       └── middleware.py   # 中间件
├── rag/                    # RAG 模块
│   ├── rag_service.py      # RAG 服务（混合检索+重排）
│   ├── vector_store.py     # 向量存储
│   ├── bm25_store.py       # BM25 检索
│   └── reranker.py         # 重排序
├── model/                  # 模型工厂
│   └── factory.py          # 聊天模型/Embedding 模型
├── utils/                  # 工具函数
│   ├── config_handler.py   # 配置处理
│   ├── file_handler.py     # 文件处理
│   ├── logger_handler.py   # 日志处理
│   ├── path_tool.py        # 路径工具
│   └── prompt_loader.py    # 提示词加载
├── config/                 # 配置文件
│   ├── agent.yml           # 智能体配置
│   ├── chroma.yml          # 向量数据库配置
│   ├── rag.yml             # RAG 配置
│   └── prompts.yml         # 提示词路径配置
├── data/                   # 知识库数据
│   ├── external/           # 外部数据
│   └── *.txt / *.pdf       # 知识库文档
├── prompts/                # 提示词模板
│   ├── main_prompt.txt     # 主提示词
│   ├── rag_summarize.txt   # RAG 提示词
│   └── report_prompt.txt   # 报告提示词
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key

### 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：如需生成 requirements.txt，可使用 `pip freeze > requirements.txt`

### 配置 API Key

设置环境变量：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key

# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key
```

### 准备知识库

将扫地机器人相关的文档（txt 或 pdf 格式）放入 `data/` 目录下。

支持的文件类型可在 `config/chroma.yml` 中配置。

### 运行项目

```bash
streamlit run app.py
```

启动后在浏览器访问 `http://localhost:8501` 即可使用。

## 配置说明

主要配置文件位于 `config/` 目录下：

### rag.yml - RAG 配置

```yaml
chat_model_name: qwen3-max          # 聊天模型名称
embedding_model_name: text-embedding-v4  # Embedding 模型名称

hybrid_search:
  enable: true                      # 是否启用混合检索
  bm25_top_n: 50                    # BM25 检索数量
  dense_top_n: 50                   # 稠密向量检索数量
  fusion_top_n: 20                  # 融合后保留数量

rrf:
  k: 60                             # RRF 融合参数

reranker:
  enable: true                      # 是否启用重排序
  provider: "dashscope"
  model: "qwen3-rerank"
  top_k: 3                          # 最终返回数量
  top_n: 20                         # 重排序输入数量
```

### chroma.yml - 向量数据库配置

```yaml
collection_name: agent              # 集合名称
persist_directory: chroma_db        # 持久化目录
k: 3                                # 默认检索数量
data_path: data                     # 知识库数据目录
chunk_size: 200                     # 文档分块大小
chunk_overlap: 20                   # 分块重叠大小
```

## License

MIT
