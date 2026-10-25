"""
第一阶段：K8s 文档加载与切分（纯手写版）
目标：理解 RAG 的第一块积木 —— chunking
"""
import re
import yaml
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = PROJECT_ROOT / "data/raw/k8s-website/content/en/docs"

if not DOCS_ROOT.exists():
    raise FileNotFoundError(f"文档目录不存在: {DOCS_ROOT}")

# 只用核心目录
TARGET_DIRS = ["concepts", "tasks"]


def parse_markdown(path: Path):
    """把 Markdown 拆成：元信息(front matter) + 正文"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta, body = {}, text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2]
            except yaml.YAMLError:
                pass
    return meta, body


def split_by_headers(body: str):
    """按 Markdown 标题(##)切分，标题本身保留在块开头"""
    chunks, current = [], []

    for line in body.splitlines():
        if re.match(r"^#{1,3}\s", line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())

    # 过滤掉太短的碎片（比如空行块）
    return [c for c in chunks if len(c) > 50]

def split_long_chunk(text: str, max_chars: int = 2000, overlap: int = 200):
    """超长块二次切分：按固定长度滑动窗口切，相邻块保留重叠"""
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        parts.append(text[start:end].strip())
        if end >= len(text):
            break                    # 已经切到末尾，必须退出，否则死循环
        start = end - overlap        # 下一次从"回退overlap"的位置开始
    return [p for p in parts if len(p) > 50]

# examples 目录 = content/en/examples（DOCS_ROOT 是 content/en/docs，往上一层就是 en）
EXAMPLES_ROOT = DOCS_ROOT.parent / "examples"

def resolve_code_samples(body: str):
    """把 {{% code_sample file="xxx" %}} 占位符替换成真实文件内容"""
    pattern = re.compile(r'\{\{%\s*code_sample\s+file="([^"]+)"[^%]*%\}\}')

    def _replace(match):
        rel_path = match.group(1)                       # 例如 "controllers/job.yaml"
        real_file = EXAMPLES_ROOT / rel_path
        if real_file.exists():
            content = real_file.read_text(encoding="utf-8", errors="ignore").strip()
            lang = "yaml" if rel_path.endswith((".yaml", ".yml")) else ""
            return f"\n```{lang}\n{content}\n```\n"     # 包成标准 Markdown 代码块
        return ""                                       # 找不到文件就删掉占位符

    return pattern.sub(_replace, body)


def main():
    all_chunks = []
    for sub in TARGET_DIRS:
        for md_file in (DOCS_ROOT / sub).rglob("*.md"):
            meta, body = parse_markdown(md_file)
            body = resolve_code_samples(body)
            title = meta.get("title", md_file.stem)

            for chunk in split_by_headers(body):
                for piece in split_long_chunk(chunk):  # ← 新增：超长块二次切分
                    all_chunks.append({
                        "text": piece,  # ← 注意这里从 chunk 改成 piece
                        "metadata": {
                            "source": str(md_file.relative_to(DOCS_ROOT)),
                            "title": title,
                        },
                    })

    # 打印统计 + 看一个样本
    print(f"总块数: {len(all_chunks)}")
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"块长度: 最小 {min(lengths)} / 最大 {max(lengths)} "
          f"/ 平均 {sum(lengths)//len(lengths)} 字符")
    print("\n===== 样例 =====")
    sample = all_chunks[len(all_chunks)//2]
    print(f"来源: {sample['metadata']['source']}")
    print(f"标题: {sample['metadata']['title']}")
    print(f"内容前300字:\n{sample['text'][:300]}")


    # 把切分结果存盘，供下一步向量化使用
    out_path = PROJECT_ROOT / "data/processed/chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 目录不存在就创建
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n已保存到: {out_path}")


if __name__ == "__main__":
    main()

