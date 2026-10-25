"""最小 API 测试：确认 key 可用、模型能返回"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 里的 MOONSHOT_API_KEY

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)

resp = client.chat.completions.create(
    model="kimi-k2.6",
    messages=[{"role": "user", "content": "用一句话解释什么是 RAG"}],
)

print(resp.choices[0].message.content)