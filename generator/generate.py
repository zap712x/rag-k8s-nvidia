"""在线生成链路：检索 → 拼 prompt → 调 LLM → 输出答案+来源"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 让本项目能 import 到 retriever 包
sys.path.append(str(Path(__file__).resolve().parent.parent))
from retriever.search import search

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

PROMPT_TEMPLATE = """你是 Kubernetes 文档问答助手。请根据下面给出的资料回答用户问题。

要求：
1. 只要资料中包含与问题相关的内容，就基于资料尽力回答，并在回答中标注引用编号，如 [1]
2. 只有当资料与问题完全无关时，才回答"根据现有资料无法回答"
3. 不要编造资料中没有的信息

【资料】
{context}

【问题】
{question}
"""



def answer(question, top_k=3):
    """完整的 RAG 问答流程。

    Args:
        question (str): 用户问题。
        top_k (int): 检索的块数量。

    Returns:
        tuple: (答案文本, 检索到的块列表)。
    """
    hits = search(question, top_k=top_k)

    # 把检索结果拼成编号列表，塞进 prompt 的【资料】位置
    context = "\n\n".join(
        f"[{i}] 来源：{h['source']} / {h['title']}\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    resp = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, hits


if __name__ == "__main__":
    q = "Who created kubernetes?"
    ans, hits = answer(q)
    print("【回答】\n" + ans)
    print("\n【来源】")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['source']} / {h['title']}（距离 {h['distance']:.3f}）")
