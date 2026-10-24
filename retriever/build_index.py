import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"
DB_DIR = PROJECT_ROOT / "data/vector_db"          # 向量数据库存放位置

# 1. 读取 chunks.jsonl，每行一个 JSON
chunks = []
with CHUNKS_PATH.open(encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
print(f"读取到 {len(chunks)} 个块")

# 2. 加载 Embedding 模型（今天 demo 用过的那个，已缓存，不用重新下载）
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 3. 把所有块的 text 批量转成向量
#    4690 个块在 CPU 上大约要几分钟，show_progress_bar 能看到进度条
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
print(f"向量形状: {embeddings.shape}")   # 应该是 (4690, 384)

# 4. 写入 Chroma 向量数据库（本地文件库，持久化在 data/vector_db/）
client = chromadb.PersistentClient(path=str(DB_DIR))
# 如果之前建过同名集合，先删掉重来，保证索引可重复构建
try:
    client.delete_collection("k8s_docs")
except Exception:
    pass
collection = client.create_collection("k8s_docs")

collection.add(
    ids=[str(i) for i in range(len(chunks))],       # 每块的唯一编号
    embeddings=embeddings.tolist(),                  # 向量
    documents=texts,                                 # 原文
    metadatas=[c["metadata"] for c in chunks],       # source + title
)
print(f"已写入 {collection.count()} 条向量到 {DB_DIR}")
