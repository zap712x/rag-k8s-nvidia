# rag-k8s-nvidia

基于 RAG（检索增强生成）的 Kubernetes / NVIDIA 文档智能问答系统。
从官方文档出发，完成 数据切块 → 向量建库 → 语义检索 → LLM 生成 → 来源溯源 的完整链路，
并以测试视角持续打磨检索与生成质量。

## 架构流程

    官方文档(.md) → 切块(chunk) → Embedding → Chroma 向量库
                                                   ↑
    用户提问 → 向量检索 Top-K → 拼 Prompt → LLM 生成 → 答案 + 引用来源

## 技术栈

| 环节 | 选型 | 理由 |
|------|------|------|
| Embedding | BAAI/bge-small-en-v1.5（384 维） | 免费、本地运行、结果可复现 |
| 向量数据库 | Chroma（PersistentClient） | 轻量、零部署、适合单机项目 |
| LLM | Kimi（kimi-k2.6，OpenAI 兼容 API） | 国内直连、按量计费 |
| 语言/环境 | Python 3.13，sentence-transformers / openai / python-dotenv | |

## 目录结构

    ├── ingest/
    │   └── chunk_markdown.py    # Markdown 文档切块
    ├── retriever/
    │   ├── build_index.py       # Embedding + 写入 Chroma 向量库（4690 块）
    │   ├── search.py            # 检索模块：search(query, top_k)
    │   ├── search_demo.py       # 检索效果演示
    │   └── embedding_demo.py    # Embedding 原理演示
    ├── generator/
    │   ├── generate.py          # 在线生成链路：检索 → Prompt → LLM → 答案+来源
    │   └── test_api.py          # LLM API 连通性测试
    ├── docs/                    # 原始文档（K8s 官方文档，NVIDIA 文档待接入）
    ├── data/                    # 向量库数据（gitignore）
    └── .env                     # API Key（gitignore）

## 快速开始

1. 安装依赖

       pip install sentence-transformers chromadb openai python-dotenv

2. 配置 API Key（platform.moonshot.cn 申请），在项目根目录创建 `.env`：

       MOONSHOT_API_KEY=sk-你的key

3. 离线建库（首次运行）

       python ingest/chunk_markdown.py
       python retriever/build_index.py

4. 提问

       python generator/generate.py

## 当前进展

- [x] v0.1 全链路打通：4690 个文档块建库，检索 → 生成 → 溯源可用
- [x] v0.2 回归测试集：固化典型问题，每次改动可回归验证
- [x] v0.3 拒答机制优化：修复"资料相关却拒答"的一致性问题
- [ ] v0.4 Reranker 二级精排：解决向量距离扎堆导致的误召回
- [ ] v0.5 RAGAS 自动评测体系
- [ ] v0.6 接入 NVIDIA 文档，支持多知识库

## 测试中发现的质量问题（持续记录）

第一轮功能测试（5 个典型问题）发现：

1. **拒答行为不一致**：Top-1 距离相同（0.315）的两个问题，一个拒答一个回答，
   拒答决策无客观标准，纯靠模型主观判断
2. **误拒答**：Top-1 来源实际相关（距离 0.315），模型仍回答"根据现有资料无法回答"
3. **距离区分度低**：回答正确与拒答的问题，距离区间高度重叠（0.27~0.55），
   暂无法单靠距离设拒答阈值 → 计划引入 Reranker

## License

MIT