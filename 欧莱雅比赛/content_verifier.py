# content_verifier.py
# 赛题二「信任守护师」内容鉴真引擎

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

# 从 prompts 模块导入提示词
try:
    from prompts_track2 import (
        AGENT_FOLLOWUP_PROMPT,
        CONTENT_VERIFIER_PROMPT,
        COT_TEMPLATE,
        IMAGE_VERIFICATION_PROMPT,
        RISK_ASSESSMENT_PROMPT,
    )
except ImportError:
    # 如果 prompts_track2 不可用，使用内置默认值
    CONTENT_VERIFIER_PROMPT = "请分析这段美妆内容是否存在虚假宣传、AI生成、夸大功效等问题。"
    IMAGE_VERIFICATION_PROMPT = "请分析这张图片是否存在AI生成、拼接篡改等痕迹。"
    RISK_ASSESSMENT_PROMPT = "请根据分析结果给出风险评级和处置建议。"
    AGENT_FOLLOWUP_PROMPT = "请根据分析结果与用户互动。"
    COT_TEMPLATE = ""

load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================
# 常量配置
# ============================================================
DEFAULT_TEXT_MODEL = "qwen3.8-flash"
DEFAULT_REASONING_MODEL = "qwen-plus"

# ============================================================
# 客户端管理
# ============================================================
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """懒加载客户端，避免重复初始化。"""
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
    """将图片文件编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型。"""
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
    image_path: Optional[str] = None,
    model: str = DEFAULT_TEXT_MODEL,
    use_cot: bool = True,
) -> dict:
    """
    美妆内容鉴真主函数。

    参数:
        text: 待检测的文案内容（种草笔记/商品评价等）
        image_path: 可选的图片路径（产品图/对比图/成分表等）
        model: 使用的模型名称
        use_cot: 是否启用思维链推理（更详细的分析但消耗更多token）

    返回:
        dict: 包含鉴真结果的字典
    """
    if not text or not str(text).strip():
        raise ValueError("text 不能为空")

    image_analysis = None
    if image_path and os.path.exists(image_path):
        image_analysis = analyze_image(image_path, model=model)

    prompt = CONTENT_VERIFIER_PROMPT
    if use_cot and COT_TEMPLATE:
        prompt = f"{COT_TEMPLATE}\n\n{prompt}"

    if image_analysis:
        prompt += "\n\n【图片鉴伪结果】\n" + json.dumps(image_analysis, ensure_ascii=False, indent=2)

    user_content = "请对以下内容进行鉴真分析：\n\n" + text
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]

    result_text = _chat_completion(model=model, messages=messages, max_tokens=4096)
    result = _parse_json_result(result_text)

    if image_analysis:
        result["image_analysis"] = image_analysis

    return result


def verify_content_with_image(
    text: str,
    image_path: str,
    model: str = DEFAULT_TEXT_MODEL,
) -> dict:
    """
    多模态鉴真：同时传入文案和图片，让多模态模型直接分析。

    注意：需要模型支持多模态（如 qwen-vl-plus, qwen3.8-flash）
    """
    if not text or not str(text).strip():
        raise ValueError("text 不能为空")
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    b64_image = encode_image(image_path)
    mime_type = get_image_mime(image_path)

    messages = [
        {"role": "system", "content": CONTENT_VERIFIER_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
                {"type": "text", "text": "请对以下美妆内容进行鉴真分析：\n\n" + text},
            ],
        },
    ]

    result_text = _chat_completion(model=model, messages=messages, max_tokens=4096)
    return _parse_json_result(result_text)


def analyze_image(image_path: str, model: str = DEFAULT_TEXT_MODEL) -> dict:
    """单独分析图片是否存在AI生成/篡改痕迹。"""
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    b64_image = encode_image(image_path)
    mime_type = get_image_mime(image_path)

    messages = [
        {"role": "system", "content": IMAGE_VERIFICATION_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
                {"type": "text", "text": "请分析这张图片的真实性。"},
            ],
        },
    ]

    result_text = _chat_completion(model=model, messages=messages, max_tokens=2048)
    return _parse_json_result(result_text)


def assess_risk(analysis_result: dict) -> dict:
    """根据鉴真分析结果进行风险评级和处置建议。"""
    if not isinstance(analysis_result, dict):
        raise ValueError("analysis_result 必须是字典")

    prompt = RISK_ASSESSMENT_PROMPT
    user_content = (
        "以下是内容鉴真分析结果：\n\n"
        + json.dumps(analysis_result, ensure_ascii=False, indent=2)
        + "\n\n请给出最终风险评级和处置建议。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]

    result_text = _chat_completion(
        model=DEFAULT_REASONING_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )
    result = _parse_json_result(result_text)

    # 一致性维度风险分层
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
    user_message: Optional[str] = None,
    model: str = DEFAULT_REASONING_MODEL,
) -> str:
    """Agent 多轮对话：根据鉴真结果与用户互动。"""
    if not isinstance(analysis_result, dict):
        raise ValueError("analysis_result 必须是字典")

    messages = [{"role": "system", "content": AGENT_FOLLOWUP_PROMPT}]

    if user_message:
        messages.append({"role": "user", "content": user_message})

    messages.append(
        {
            "role": "system",
            "content": "【鉴真分析结果】\n" + json.dumps(analysis_result, ensure_ascii=False, indent=2),
        }
    )

    return _chat_completion(model=model, messages=messages, max_tokens=1024, temperature=0.7)


# ============================================================
# 工具函数
# ============================================================
def _chat_completion(
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 0.3,
) -> str:
    """统一封装模型调用，减少重复代码并提升可维护性。"""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _parse_json_result(text: str) -> dict:
    """尝试从模型输出中解析 JSON。"""
    if text is None:
        return {"raw_output": "", "parse_error": "模型返回为空"}

    parsed = str(text).strip()

    # 去掉 markdown 代码块标记
    if parsed.startswith("```"):
        lines = parsed.split("\n")
        if len(lines) >= 3:
            parsed = "\n".join(lines[1:-1]) if lines[0].startswith("```") else parsed

    parsed = parsed.strip()
    if not parsed:
        return {"raw_output": "", "parse_error": "模型返回为空"}

    # 尝试直接解析 JSON
    try:
        return json.loads(parsed)
    except json.JSONDecodeError:
        # 尝试提取最外层 JSON 对象
        start = parsed.find("{")
        end = parsed.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(parsed[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {"raw_output": parsed, "parse_error": "JSON解析失败，请检查模型输出格式"}


def print_report(result: dict):
    """格式化打印鉴真报告。"""
    print("\n" + "=" * 60)
    print("  美妆内容鉴真报告")
    print("=" * 60)

    ai_prob = result.get("ai_generated_probability", result.get("ai_generated", "未评估"))
    print("\n[AI生成检测] 概率等级: " + str(ai_prob))

    exaggeration = result.get("exaggerated_claims", [])
    if exaggeration:
        print("\n[夸大功效检测] 发现 " + str(len(exaggeration)) + " 处可疑:")
        for i, claim in enumerate(exaggeration, 1):
            print("  " + str(i) + ". " + str(claim))
    else:
        print("\n[夸大功效检测] 未发现明显夸大")

    fake = result.get("fake_endorsements", [])
    if fake:
        print("\n[虚假背书检测] 发现 " + str(len(fake)) + " 处可疑:")
        for i, item in enumerate(fake, 1):
            print("  " + str(i) + ". " + str(item))
    else:
        print("\n[虚假背书检测] 未发现明显虚假背书")

    tampering = result.get("image_tampering_suspected", result.get("image_tampering", None))
    if tampering is not None:
        print("\n[图片鉴伪] 疑似篡改: " + ("是" if tampering else "否"))

    consistency = result.get("text_image_consistency", "未评估")
    print("\n[图文一致性] 校验结果: " + str(consistency))

    risk = result.get("risk_level", "未评估")
    print("\n[综合风险等级] " + str(risk))

    consistency_risk = result.get("consistency_risk", "未评估")
    print("\n[一致性维度风险] " + str(consistency_risk))

    reasoning = result.get("reasoning", {})
    if reasoning:
        print("\n[推理依据]")
        for key, value in reasoning.items():
            label = key.replace("_", " ").title()
            print("  " + label + ": " + str(value))

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

    print("美妆内容鉴真助手 v1.1")
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
