# content_verifier.py
# 赛题二「信任守护师」内容鉴真引擎

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    retry = None
    retry_if_exception_type = None
    stop_after_attempt = None
    wait_exponential = None

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:  # pragma: no cover
    Console = None
    Panel = None

# 从 prompts 模块导入提示词
try:
    from prompts_track2 import (
        AGENT_FOLLOWUP_PROMPT,
        CONTENT_VERIFIER_PROMPT,
        COT_TEMPLATE,
        IMAGE_VERIFICATION_PROMPT,
        RISK_ASSESSMENT_PROMPT,
    )
except ImportError:  # pragma: no cover
    CONTENT_VERIFIER_PROMPT = "请分析这段美妆内容是否存在虚假宣传、AI生成、夸大功效等问题。"
    IMAGE_VERIFICATION_PROMPT = "请分析这张图片是否存在AI生成、拼接篡改等痕迹。"
    RISK_ASSESSMENT_PROMPT = "请根据分析结果给出风险评级和处置建议。"
    AGENT_FOLLOWUP_PROMPT = "请根据分析结果与用户互动。"
    COT_TEMPLATE = ""

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_TEXT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3.8-flash")
DEFAULT_REASONING_MODEL = os.getenv("REASONING_MODEL", "qwen-plus")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
DEFAULT_RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF_BASE", "2"))
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

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
            timeout=DEFAULT_TIMEOUT,
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
) -> dict:
    """美妆内容鉴真主函数。"""
    text = _normalize_text(text)
    if not text:
        raise ValueError("text 不能为空")

    image_analysis = None
    if image_path and os.path.exists(image_path):
        image_analysis = analyze_image(image_path, model=model, max_tokens=2048)

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

    result_text = _chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = _parse_json_result(result_text)

    if image_analysis:
        result["image_analysis"] = image_analysis

    return result


def verify_content_with_image(
    text: str,
    image_path: str,
    model: str = DEFAULT_TEXT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
) -> dict:
    """多模态鉴真：同时传入文案和图片，让多模态模型直接分析。"""
    text = _normalize_text(text)
    if not text:
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

    result_text = _chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_json_result(result_text)


def analyze_image(
    image_path: str,
    model: str = DEFAULT_TEXT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> dict:
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

    result_text = _chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_json_result(result_text)


def assess_risk(
    analysis_result: dict,
    model: str = DEFAULT_REASONING_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> dict:
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
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = _parse_json_result(result_text)

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
    max_tokens: int = 1024,
    temperature: float = 0.7,
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

    return _chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ============================================================
# 工具函数
# ============================================================
def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _should_retry_exception(exc: Exception) -> bool:
    if exc is None:
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code in {429, 500, 502, 503, 504}
    message = str(exc).lower()
    if "429" in message or "rate limit" in message:
        return True
    if "500" in message or "502" in message or "503" in message or "504" in message:
        return True
    if "timeout" in message or "temporarily unavailable" in message:
        return True
    return False


def _execute_with_retry(func, *args, **kwargs):
    """统一重试逻辑：3 次指数退避，429/5xx 才重试。"""
    retries = int(os.getenv("RETRY_COUNT", str(DEFAULT_RETRY_COUNT)))
    backoff_base = float(os.getenv("RETRY_BACKOFF_BASE", str(DEFAULT_RETRY_BACKOFF)))

    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt >= retries or not _should_retry_exception(exc):
                raise
            sleep_seconds = backoff_base ** attempt
            logging.warning(
                "API 调用失败，%s 秒后重试 (%s/%s): %s",
                sleep_seconds,
                attempt + 1,
                retries,
                exc,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("重试逻辑结束，但未返回结果")


def _chat_completion(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
) -> str:
    """统一封装模型调用，减少重复代码并提升可维护性。"""
    client = get_client()

    def _request():
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("模型返回内容为空")
        return content

    return _execute_with_retry(_request)


def _parse_json_result(text: str) -> dict:
    """尝试从模型输出中解析 JSON。"""
    if text is None:
        return {"raw_output": "", "parse_error": "模型返回为空"}

    parsed = str(text).strip()

    if parsed.startswith("```"):
        lines = parsed.split("\n")
        if len(lines) >= 3:
            if lines[0].startswith("```"):
                parsed = "\n".join(lines[1:-1])

    parsed = parsed.strip()
    if not parsed:
        return {"raw_output": "", "parse_error": "模型返回为空"}

    try:
        result = json.loads(parsed)
        if isinstance(result, dict):
            return result
        return {"raw_output": result, "parse_error": "JSON 解析成功，但顶层数据不为对象"}
    except json.JSONDecodeError:
        start = parsed.find("{")
        end = parsed.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(parsed[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {"raw_output": parsed, "parse_error": "JSON解析失败，请检查模型输出格式"}


def print_report(result: dict):
    """格式化打印鉴真报告。"""
    if not isinstance(result, dict):
        raise ValueError("result 必须是字典")

    if Console is not None:
        console = Console()
        console.print(Panel.fit("美妆内容鉴真报告", style="bold cyan"))
        console.print(f"[bold]AI生成检测:[/bold] {result.get('ai_generated_probability', result.get('ai_generated', '未评估'))}")
        console.print(f"[bold]夸大功效:[/bold] {result.get('exaggerated_claims', [])}")
        console.print(f"[bold]虚假背书:[/bold] {result.get('fake_endorsements', [])}")
        console.print(f"[bold]图片篡改:[/bold] {result.get('image_tampering_suspected', result.get('image_tampering', '未评估'))}")
        console.print(f"[bold]图文一致性:[/bold] {result.get('text_image_consistency', '未评估')}")
        console.print(f"[bold]综合风险:[/bold] {result.get('risk_level', '未评估')}")
        console.print(f"[bold]一致性维度风险:[/bold] {result.get('consistency_risk', '未评估')}")
        if result.get("suggestions"):
            console.print("[bold]处置建议:[/bold]")
            for i, suggestion in enumerate(result.get("suggestions", []), 1):
                console.print(f"  {i}. {suggestion}")
        return

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="赛题二：内容鉴真与风险识别工具")
    parser.add_argument("text", nargs="?", help="待检测文案内容")
    parser.add_argument("image_path", nargs="?", help="待检测图片路径（可选）")
    parser.add_argument("--image", dest="image_arg", help="待检测图片路径（可选）")
    parser.add_argument("--model", default=os.getenv("DEFAULT_MODEL", DEFAULT_TEXT_MODEL), help="AI 模型名称")
    parser.add_argument("--reasoning-model", default=os.getenv("REASONING_MODEL", DEFAULT_REASONING_MODEL), help="风险评估模型名称")
    parser.add_argument("--no-cot", action="store_true", help="关闭思维链模式")
    parser.add_argument("--output-json", help="将分析结果写入 JSON 文件")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))), help="单次请求最大 token 数")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.3")), help="采样温度")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    text = _normalize_text(args.text)
    image_path = args.image_arg or args.image_path

a    if not text:
        parser.error("请提供待检测文案，或通过 --image 传递图片路径")

    if image_path and not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    try:
        if image_path:
            result = verify_content_with_image(
                text=text,
                image_path=image_path,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            result = verify_content(
                text=text,
                image_path=None,
                model=args.model,
                use_cot=not args.no_cot,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

        print_report(result)

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[JSON输出] 已保存到: {output_path}")

        return 0

    except Exception as exc:  # pragma: no cover
        logging.exception("内容鉴真执行失败")
        print(f"\n[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
