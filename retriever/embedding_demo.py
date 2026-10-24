import os
# 模型要从 HuggingFace 下载，先设置国内镜像，否则下载会卡死
# 注意：必须写在 import sentence_transformers 之前，否则不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer, util

# 加载 BGE 句向量模型（首次运行会自动下载，约 100MB）
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 三句话：前两句意思相同但用词不同，第三句是无关话题
sentences = [
    "A Pod is the smallest deployable unit in Kubernetes",
    "Pods are the basic building blocks of K8s workloads",
    "How to configure GPU drivers on NVIDIA DGX systems",
]

# encode：把每句话变成一个向量（一串浮点数）
embeddings = model.encode(sentences)
print(f"向量形状: {embeddings.shape}")  # 应该是 (3, 384)：3 句话，每句 384 维

# cos_sim：余弦相似度，范围 [-1, 1]，越接近 1 语义越相似
print(f"句1 vs 句2（同义不同词）: {util.cos_sim(embeddings[0], embeddings[1]).item():.3f}")
print(f"句1 vs 句3（无关话题）  : {util.cos_sim(embeddings[0], embeddings[2]).item():.3f}")
