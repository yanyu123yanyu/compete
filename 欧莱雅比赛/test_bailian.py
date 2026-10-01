from client_utils import chat

result = chat(
    "帮我总结一下这个产品的核心卖点",
    system_prompt="你是一名专业的品牌文案顾问，回答要简洁、有条理。",
)
print(result)