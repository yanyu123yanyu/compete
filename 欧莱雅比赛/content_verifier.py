# content_verifier.py
# 赛题二「信任守护师」内容鉴真引擎

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 从 prompts 模块导入提示词
try:
    from prompts_track2 import (
        CONTENT_VERIFIER_PROMPT,
        IMAGE_VERIFICATION_PROMPT,
        RISK_ASSESSMENT_PROMPT,
        AGENT_FOLLOWUP_PROMPT,
        COT_TEMPLATE,
    )
except ImportError:
    # 如果 prompts_track2 不可用，使用内置默认值
    CONTENT_VERIFIER_PROMPT = "请分析这段美妆内容是否存在虚假宣传、AI生成、夸大功效等问题。"
    IMAGE_VERIFICATION_PROMPT = "请分析这张图片是否存在AI生成、拼接篡改等痕迹。"
    RISK_ASSESSMENT_PROMPT = "请根据分析结果给出风险评级和处置建议。"
    AGENT_FOLLOWUP_PROMPT = "请根据分析结果与用户互动。"
    COT_TEMPLATE = ""

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 客户端管理
# ============================================================
_client = None


def get_client():
    """懒加载客户端"""
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


# ============================================================
# 图片处理工具
# ============================================================
def encode_image(image_path: str) -> str:
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime_map = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


# ============================================================
# 核心鉴真函数
# ============================================================
def verify_content(
    text: str,
    image_path: str = None,
    model: str = "qwen3.8-flash",
    use_cot: bool = True,
) -> dict:
    """
    美妆内容鉴真主函数

    参数:
        text: 待检测的文案内容（种草笔记/商品评价等）
        image_path: 可选的图片路径（产品图/对比图/成分表等）
        model: 使用的模型名称
        use_cot: 是否启用思维链推理（更详细的分析但消耗更多token）

    返回:
        dict: 包含鉴真结果的字典
    """
    client = get_client()

    # 构建 messages
    messages = []

    # 如果有图片，先进行图片鉴伪（多模态）
    image_analysis = None
    if image_path and os.path.exists(image_path):
        image_analysis = analyze_image(image_path, model)

    # 构建内容鉴真请求
    prompt = CONTENT_VERIFIER_PROMPT

    # 如果启用思维链，附加推理模板
    if use_cot and COT_TEMPLATE:
        prompt = COT_TEMPLATE + "\n\n" + prompt

    # 附加图片分析结果（如果有）
    if image_analysis:
        prompt += "\n\n【图片鉴伪结果】\n" + json.dumps(image_analysis, ensure_ascii=False, indent=2)

    # 构建用户消息
    user_content = "请对以下内容进行鉴真分析：\n\n" + text

    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": user_content})

    # 调用模型
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    result_text = response.choices[0].message.content

    # 尝试解析 JSON 输出
    result = _parse_json_result(result_text)

    # 附加图片分析结果
    if image_analysis:
        result["image_analysis"] = image_analysis

    return result


def verify_content_with_image(
    text: str,
    image_path: str,
    model: str = "qwen3.8-flash",
) -> dict:
    """
    多模态鉴真：同时传入文案和图片，让多模态模型直接分析

    注意：需要模型支持多模态（如 qwen-vl-plus, qwen3.8-flash）

    参数:
        text: 待检测的文案
        image_path: 图片路径
        model: 多模态模型名称
    """
    client = get_client()

    # 编码图片
    b64_image = encode_image(image_path)
    mime_type = get_image_mime(image_path)

    # 构建多模态消息
    messages = [
        {
            "role": "system",
            "content": CONTENT_VERIFIER_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:" + mime_type + ";base64," + b64_image
                    },
                },
                {
                    "type": "text",
                    "text": "请对以下美妆内容进行鉴真分析：\n\n" + text,
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )

    result_text = response.choices[0].message.content

    return _parse_json_result(result_text)


def analyze_image(image_path: str, model: str = "qwen3.8-flash") -> dict:
    """
    单独分析图片是否存在AI生成/篡改痕迹

    参数:
        image_path: 图片文件路径
        model: 多模态模型名称
    """
    client = get_client()

    b64_image = encode_image(image_path)
    mime_type = get_image_mime(image_path)

    messages = [
        {
            "role": "system",
            "content": IMAGE_VERIFICATION_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:" + mime_type + ";base64," + b64_image
                    },
                },
                {"type": "text", "text": "请分析这张图片的真实性。"},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )

    result_text = response.choices[0].message.content
    return _parse_json_result(result_text)


def assess_risk(analysis_result: dict) -> dict:
    """
    根据鉴真分析结果进行风险评级和处置建议

    参数:
        analysis_result: verify_content 返回的分析结果字典
    """
    client = get_client()

    # 构建风险评估请求
    prompt = RISK_ASSESSMENT_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "以下是内容鉴真分析结果：\n\n" + json.dumps(analysis_result, ensure_ascii=False, indent=2) + "\n\n请给出最终风险评级和处置建议。",
        },
    ]

    response = client.chat.completions.create(
        model="qwen-plus",  # 风险评估不需要多模态，用 qwen-plus 省额度
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    result_text = response.choices[0].message.content
    result = _parse_json_result(result_text)

    # 【新增】一致性维度风险分层
    consistency = analysis_result.get("text_image_consistency", "")
    exaggerated = analysis_result.get("exaggerated_claims") or []
    endorsements = analysis_result.get("fake_endorsements") or []

    if consistency in ("一致", "consistent"):
        result["consistency_risk"] = "低" if not exaggerated and not endorsements else "中"
    else:
        result["consistency_risk"] = "高"

    return result



def agent_followup(
    analysis_result: dict,
    user_message: str = None,
    model: str = "qwen-plus",
) -> str:
    """
    Agent 多轮对话：根据鉴真结果与用户互动

    参数:
        analysis_result: 鉴真分析结果
        user_message: 用户当前输入（如果为None则输出总结）
        model: 模型名称
    """
    client = get_client()

    messages = [
        {"role": "system", "content": AGENT_FOLLOWUP_PROMPT},
    ]

    if user_message:
        messages.append({"role": "user", "content": user_message})

    messages.append(
        {
            "role": "system",
            "content": "【鉴真分析结果】\n" + json.dumps(analysis_result, ensure_ascii=False, indent=2),
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ============================================================
# 工具函数
# ============================================================
def _parse_json_result(text: str) -> dict:
    """尝试从模型输出中解析 JSON"""
    text = text.strip()

    # 去掉 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 解析失败，返回原始文本
        return {"raw_output": text, "parse_error": "JSON解析失败，请检查模型输出格式"}


def print_report(result: dict):
    """格式化打印鉴真报告"""
    print("\n" + "=" * 60)
    print("  美妆内容鉴真报告")
    print("=" * 60)

    # AI生成概率
    ai_prob = result.get("ai_generated_probability", "未评估")
    print("\n[AI生成检测] 概率等级: " + str(ai_prob))

    # 夸大功效
    exaggeration = result.get("exaggerated_claims", [])
    if exaggeration:
        print("\n[夸大功效检测] 发现 " + str(len(exaggeration)) + " 处可疑:")
        for i, claim in enumerate(exaggeration, 1):
            print("  " + str(i) + ". " + str(claim))
    else:
        print("\n[夸大功效检测] 未发现明显夸大")

    # 虚假背书
    fake = result.get("fake_endorsements", [])
    if fake:
        print("\n[虚假背书检测] 发现 " + str(len(fake)) + " 处可疑:")
        for i, item in enumerate(fake, 1):
            print("  " + str(i) + ". " + str(item))
    else:
        print("\n[虚假背书检测] 未发现明显虚假背书")

    # 图片篡改
    tampering = result.get("image_tampering_suspected", None)
    if tampering is not None:
        print("\n[图片鉴伪] 疑似篡改: " + ("是" if tampering else "否"))

    # 图文一致性
    consistency = result.get("text_image_consistency", "未评估")
    print("\n[图文一致性] 校验结果: " + str(consistency))

    # 风险等级
    risk = result.get("risk_level", "未评估")
    print("\n[综合风险等级] " + str(risk))

    # 一致性维度风险
    consistency_risk = result.get("consistency_risk", "未评估")
    print("\n[一致性维度风险] " + str(consistency_risk))

    # 推理过程
    reasoning = result.get("reasoning", {})
    if reasoning:
        print("\n[推理依据]")
        for key, value in reasoning.items():
            label = key.replace("_", " ").title()
            print("  " + label + ": " + str(value))

    # 处置建议
    suggestions = result.get("suggestions", [])
    if suggestions:
        print("\n[处置建议]")
        for i, sug in enumerate(suggestions, 1):
            print("  " + str(i) + ". " + str(sug))

    print("\n" + "=" * 60)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys

    print("美妆内容鉴真助手 v1.0")
    print("赛题二「信任守护师」")
    print()

    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 纯文本鉴真: python content_verifier.py <文案内容>")
        print("  2. 图文鉴真:   python content_verifier.py <文案内容> <图片路径>")
        print()
        print("示例:")
        print('  python content_verifier.py "这款精华液七天美白三个度，亲测有效！"')
        print('  python content_verifier.py "亲测七天白三个度" product.jpg')
        sys.exit(1)

    text = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("待检测文案: " + text[:80] + "...")
    if image_path:
        print("图片路径: " + image_path)
    print()

    try:
        if image_path and os.path.exists(image_path):
            # 多模态模式
            result = verify_content_with_image(text, image_path)
        else:
            # 纯文本模式
            result = verify_content(text)

        print_report(result)

    except Exception as e:
        print("[错误] " + type(e).__name__ + ": " + str(e))
        print()
        print("可能的原因:")
        print("  1. 检查 .env 文件中 DASHSCOPE_API_KEY 是否正确")
        print("  2. 如果使用了多模态功能，确认模型已开通（如 qwen-vl-plus）")
        print("  3. 图片文件路径是否正确")
        sys.exit(1)
