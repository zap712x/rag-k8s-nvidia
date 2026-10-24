# RAG 项目 · 阶段一：数据处理（Ingestion & Chunking）设计文档

> 项目：K8s / NVIDIA 文档智能问答系统（RAG）
> 阶段目标：把原始文档语料加工成"可直接向量化"的标准化文本块（chunks）
> 环境：MacBook Pro M1 Max / CPU / Python 3.13 + venv / PyCharm

---

## 1. 需求分析

RAG 系统的回答质量，上限在检索，检索的上限在数据。本阶段的需求可以拆成四条：

| 需求 | 说明 |
|---|---|
| R1 语料获取 | 获取权威、结构化的 K8s 官方英文文档（后续补 NVIDIA GPU 文档） |
| R2 文本切块 | 把长文档切成大小合适的块（chunk），块太大超出 Embedding 模型输入上限，块太小语义不完整 |
| R3 内容还原 | K8s 文档中的 Hugo 短代码占位符（`{{% code_sample %}}`）没有实际内容，必须替换成真实 YAML 示例 |
| R4 产物标准化 | 输出统一的 `.jsonl` 格式，每块带元数据（来源文件、标题），供下游 Embedding / 检索 / 评估使用 |

**关键约束**：计划使用的 Embedding 模型 BGE 系列输入上限为 **512 token**。英文约 4 字符 ≈ 1 token，因此单块字符数上限定为 **2000 字符**，留出余量。

## 2. 技术方案与选型

### 2.1 数据分层

```
data/
├── raw/          # 原始语料，只进不出，永不修改（immutable）
└── processed/    # 加工产物，任何时刻都可以用脚本重新生成
```

**设计理由**：原始数据与派生数据分离。切分策略以后一定会调整（这是 RAG 调优的常态），只要 raw 不动，processed 可以随时推倒重来。这也是数据工程的基本规范。

### 2.2 处理流水线

```
kubernetes/website 仓库 (git clone --depth 1)
        │
        ▼
[1] parse_markdown()        解析 front matter（YAML 头）→ 提取 title 等元数据 + 正文 body
        │
        ▼
[2] resolve_code_samples()  正则匹配 {{% code_sample file="..." %}} 占位符
                            → 到 content/en/examples/ 读真实 YAML → 以 ```yaml 代码块内联进正文
        │
        ▼
[3] split_by_headers()      一级切分：按 Markdown 标题（# / ## / ###）切，保持语义结构完整
                            → 过滤 ≤ 50 字符的碎片块
        │
        ▼
[4] split_long_chunk()      二级切分：超过 2000 字符的块做滑动窗口切分（窗口 2000 / 重叠 200）
                            → 保证不触碰 Embedding 模型的 token 上限
        │
        ▼
[5] 输出 chunks.jsonl       每行一个 JSON：{"text": ..., "metadata": {"source": ..., "title": ...}}
```

### 2.3 切分策略选型

| 候选策略 | 思路 | 结论 |
|---|---|---|
| 固定长度切分 | 每 N 字符硬切 | 会切断句子和代码块，语义破碎，弃用 |
| 递归切分 | 按段落→句子逐级切 | 通用但对结构化文档浪费标题信息 |
| **按标题切分（采用）** | 以 Markdown 标题为天然边界 | K8s 文档结构规范，标题即语义单元，且标题可存入元数据辅助检索 |
| 语义切分 | 用模型判断语义边界 | 效果最好但成本高，本阶段不引入 |

**两级切分的必要性**：纯按标题切分时，实测最长块达 **29845 字符**（如 `custom-resource-definitions.md`，全文标题稀疏）。这类块远超 512 token 上限，会被模型截断、丢失尾部内容。因此增加滑动窗口二次切分作为"保险丝"：绝大多数块走一级切分保持完整语义，少数超长块兜底切小。重叠 200 字符是为了避免答案恰好横跨切分点时上下文断裂。

## 3. 遇到的问题与解决

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | 首次运行 `总块数: 0` 且 `min() iterable argument is empty` | 相对路径依赖运行目录（PyCharm Run 的工作目录与预期不符），文档目录没定位到 | 用 `Path(__file__).resolve()` 以脚本自身位置为锚点拼绝对路径，并加 `exists()` 防御性检查，目录不存在直接抛出带明确信息的 `FileNotFoundError` |
| 2 | 手动写 `Path("~/data/...")` 找不到目录 | `~` 是 shell 的展开特性，Python 的 Path 不认识；且路径层级缺失 | 放弃手写绝对路径，统一用 `__file__` 锚定，代码换机器也能跑 |
| 3 | 切换 venv 解释器后 `ModuleNotFoundError: No module named 'yaml'` | venv 是隔离环境，之前装在系统 Python 里的 pyyaml 在 venv 中不可见 | 激活 venv 后重新 `pip install pyyaml`；从此形成习惯：装包前先确认终端有 `(.venv)` 前缀 |
| 4 | 最长块 29845 字符，疑似处理 bug | 先用数据诊断（打印 top-5 最长块及来源），确认是原文档本身标题稀疏导致，属设计内的必然情况而非 bug | 增加 `split_long_chunk()` 滑动窗口二次切分；关键细节：`if end >= len(text): break` 防止窗口到头后死循环 |
| 5 | 块内出现 `{{% code_sample %}}` 占位符，无实际内容 | K8s 文档用 Hugo 短代码引用外部示例文件，正文里只有指针 | `resolve_code_samples()`：正则捕获文件相对路径 → 到 examples 目录读取真实 YAML → 替换为 Markdown 代码块；文件不存在时替换为空串，不中断流程 |

> 方法论沉淀：问题 4 体现了"数据驱动排障"——不猜，先把异常数据打出来看分布和来源，再决定改不改代码。这套思路后续会贯穿评估（Evaluation）阶段。

## 4. 产出与质量指标

产物：`data/processed/chunks.jsonl`

| 指标 | 数值 | 说明 |
|---|---|---|
| 总块数 | **4690** | 覆盖 concepts / tasks 两大目录 |
| 最小块长 | 51 字符 | 碎片过滤阈值生效 |
| 最大块长 | **2000 字符** | 二次切分硬约束生效，约 500 token，在 BGE 512 上限内 |
| 平均块长 | 890 字符 | 分布健康，大多数块保持了标题级完整语义 |
| 元数据 | source + title | 每个块可追溯回源文件和所属章节 |

**指标演进**（策略调优的痕迹，也是面试的讲点）：

```
仅标题切分:      4014 块 / 最大 29845 / 平均 981   ← 发现超长块问题
+ 二次切分:      4618 块 / 最大 2000  / 平均 878   ← 长度收敛
+ YAML 内联:     4690 块 / 最大 2000  / 平均 890   ← 内容补全后块数微增（变长的块被重新切分）
```

## 5. 与下一阶段（Embedding）的接口

- 输入：逐行读取 `chunks.jsonl`，取 `text` 字段向量化
- 元数据 `source` / `title` 不进入向量，但随向量一起入库，用于检索结果溯源和答案引用
- 下一阶段的 Embedding 模型必须与本阶段的 2000 字符预算匹配（BGE-small-en-v1.5，512 token 上限）

---

## 附：核心代码结构（ingest/chunk_markdown.py）

```
parse_markdown(path)          # front matter 解析，yaml.YAMLError 容错
resolve_code_samples(body)    # Hugo 短代码 → 真实 YAML 内联
split_by_headers(body)        # 一级：按 # / ## / ### 标题切分，过滤 ≤50 字符
split_long_chunk(text, 2000, 200)   # 二级：滑动窗口兜底，防死循环 break
main()                        # 遍历目录 → 串联流水线 → 统计打印 → 写 JSONL
```
