"""向量检索模块：对外暴露 search() 函数"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data/vector_db"

# 模块加载时初始化一次，之后每次 search() 复用，不用重复加载模型
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection("k8s_docs")


def search(query, top_k=3):
    """在向量库中检索与问题最相似的文档块。

    Args:
        query (str): 用户问题。
        top_k (int): 返回的块数量。

    Returns:
        list[dict]: 每项含 text / source / title / distance，
                    按相似度从高到低排序。
    """
    q_emb = model.encode([query])
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "title": results["metadatas"][0][i]["title"],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    # 直接运行本文件时自测；被 import 时这段不会执行
    for h in search("What is a Deployment in Kubernetes?"):
        print(f"{h['distance']:.3f} | {h['source']} | {h['title']}")
