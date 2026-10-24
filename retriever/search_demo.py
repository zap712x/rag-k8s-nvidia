import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data/vector_db"

# 同一个模型：问题必须和库里的块用同一个模型 encode，否则向量不在同一个"语义空间"
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 打开已有的向量库（PersistentClient 读的是同一个目录）
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection("k8s_docs")

# 模拟用户提问
question = "What is a Deployment in Kubernetes?"
q_emb = model.encode([question])          # 问题 → 向量 (1, 384)

# query：在 4690 个向量里找最相似的 5 个
results = collection.query(query_embeddings=q_emb.tolist(), n_results=3)

# results 是字典：ids / documents / metadatas / distances，每个都是"每问一组"
n = len(results["ids"][0])   # 实际返回了几条，就循环几次
for i in range(n):
    dist = results["distances"][0][i]     # Chroma 返回的是距离，越小越相似
    meta = results["metadatas"][0][i]
    text = results["documents"][0][i]
    print(f"\n===== Top {i+1} | 距离 {dist:.3f} | {meta['source']} | {meta['title']} =====")
    print(text[:200])
