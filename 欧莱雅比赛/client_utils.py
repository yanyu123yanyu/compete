import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

_client = None


def get_client():
    """懒加载，避免重复创建连接"""
    global _client
    if _client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("没有找到 DASHSCOPE_API_KEY，请检查 .env 文件")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client


def chat(user_input, system_prompt=None, model="qwen-plus", temperature=0.7):
    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            if i == max_retries - 1:
                return f"[调用失败] {type(e).__name__}: {e}"
            time.sleep(2 * (i + 1))