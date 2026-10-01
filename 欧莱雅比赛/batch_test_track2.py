# batch_test_track2.py
# 赛道二：多组「文案 + 图片」批量鉴真，输出对比表格

import sys
from content_verifier import verify_content_with_image, verify_content

# ============ 测试用例配置 ============
# 每组包含：用例名、预期结果、文案、图片路径（None 表示纯文本）

# 调试开关：需要查看模型原始返回时改成 True，提交前保持 False
DEBUG_RAW_OUTPUT = False

CASES = [
    {
        "name": "用例1-正向一致",
        "expect": "一致 / 低风险",
        "text": (
            "这组测评整理了三款热门美妆产品：稿定水感防晒霜，推荐指数较高，"
            "价格100元左右，主打透明氧化锌防护，物理性低敏；贝玲妃玫瑰胭脂水，"
            "价格100元左右，质地水一般透明；稿定持妆粉底液，推荐指数高，"
            "价格100元左右，妆效偏奶油肌，性价比不错。"
        ),
        "image": "Test2.jpg",
    },
    {
        "name": "用例2-细节冲突",
        "expect": "不一致 / 高风险",
        "text": (
            "这款热门美妆产品测评推荐指数很高，性价比不错，价格只要 59.9 元。"
            "推荐理由是它的质地非常清爽，适合油皮使用，而且包装上明确标注了"
            "含有烟酰胺成分，确实是一款值得入手的入门级精华。"
        ),
        "image": "Test2.jpg",
    },
    {
        "name": "用例3-夸大功效",
        "expect": "不一致 / 高风险",
        "text": (
            "全网爆款！这款贵妇级抗老面霜原价 2999 元，现在限时特价 999 元。"
            "推荐理由是它含有珍稀深海鱼子酱提取物，能在一周内彻底祛除皱纹、"
            "逆转肌龄，连明星都在偷偷用，绝对是医美级的抗衰神器！"
        ),
        "image": "Test2.jpg",
    },
    {
        "name": "用例4-纯文本基线",
        "expect": "无图片，仅文本判断",
        "text": "七天美白三个度，七天焕新肌底，敏感肌也能放心用。",
        "image": None,
    },
]

def run_one(case):
    """执行单个用例，返回结果字典。异常时返回错误信息，不中断整体流程。"""
    result = {
        "name": case["name"],
        "expect": case["expect"],
        "risk": "-",
        "consistency": "-",
        "consistency_risk": "-",
        "exaggerated": "-",
        "endorsement": "-",
        "error": "",
    }
    try:
        if case["image"]:
            data = verify_content_with_image(case["text"], case["image"])
        else:
            data = verify_content(case["text"])

        # 【调试】仅在开关打开时输出原始返回
        if DEBUG_RAW_OUTPUT and "用例1" in case["name"]:
            print("【调试-用例1 原始返回】")
            print("data 类型：", type(data))
            print("data 的键：", list(data.keys()) if isinstance(data, dict) else "非字典")
            print("data 完整内容：", data)

        # 【修改】风险等级：字段名兼容（多名字兜底），避免模型返回不同字段名导致取不到
        risk = (
            data.get("risk_level")
            or data.get("risk")
            or data.get("risk_grade")
            or data.get("riskLevel")
            or "-"
        )
        if not case["image"] and risk != "-":
            risk = f"{risk}（仅文本维度）"
        result["risk"] = risk

        # 【新增】一致性维度风险：从 data 中取值，兼容多种字段名
        consistency_risk = (
            data.get("consistency_risk")
            or data.get("consistencyRisk")
            or "-"
        )
        result["consistency_risk"] = consistency_risk

        # 【修改】字段名对齐 content_verifier 实际返回的 text_image_consistency
        result["consistency"] = (
            data.get("text_image_consistency")
            or data.get("image_text_consistency")
            or "-"
        )

        # 【新增】纯文本用例单独标注，避免评审误读
        if not case["image"]:
            result["consistency"] = "不适用（纯文本）"

        # 【新增】解析失败检测：如果 data 为空或不是字典，标记错误
        if not isinstance(data, dict) or not data:
            result["error"] = "解析失败或返回为空"
            return result

        # 【新增】JSON 解析失败提示
        if data.get("parse_error"):
            result["error"] = "JSON解析失败"

        # 【新增】夸大项 / 背书项计数
        exc = data.get("exaggerated_claims") or []
        end = data.get("fake_endorsements") or []
        result["exaggerated"] = len(exc) if isinstance(exc, list) else str(exc)
        result["endorsement"] = len(end) if isinstance(end, list) else str(end)

        # 【新增】打印推理过程，便于定位判定依据
        try:
            reasoning = data.get("reasoning") or {}
            if isinstance(reasoning, dict):
                for k, v in reasoning.items():
                    print(f"    · {k}: {v}")
            else:
                print(f"    · reasoning: {reasoning}")
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)[:40]

    # 【兜底】若模型侧未返回 consistency_risk，本地按一致性字段推算
    if result.get("consistency_risk") in ("-", "", None):
        c = str(result.get("consistency", "")).strip()
        if c in ("一致", "consistent"):
            ex = str(result.get("exaggerated", "")).strip()
            result["consistency_risk"] = "低" if ex in ("0", "-", "") else "中"
        elif c in ("部分一致", "部分符合", "partially_consistent"):
            result["consistency_risk"] = "中"
        elif c in ("不一致", "inconsistent"):
            result["consistency_risk"] = "高"
        else:
            result["consistency_risk"] = "不适用"

        # 【纯文本用例】无图片时一致性维度不参与评估
        if "纯文本" in case.get("name", "") or "纯文本" in str(case.get("expect", "")):
            result["consistency_risk"] = "不适用"

    return result


def print_table(rows):
    """打印对齐的对比表格。"""
    headers = ["用例", "预期", "风险等级", "一致性风险", "图文一致性", "夸大项", "背书项", "异常"]
    widths = [18, 20, 12, 12, 14, 8, 8, 30]

    def fmt(vals):
        cells = []
        for v, w in zip(vals, widths):
            s = str(v)
            if len(s) > w:
                s = s[: w - 1] + "…"
            cells.append(s.ljust(w))
        return " | ".join(cells)

    print("\n" + "=" * 120)
    print("赛道二 多模态鉴真批量测试结果")
    print("=" * 120)
    print(fmt(headers))
    print("-" * 120)
    for r in rows:
        print(fmt([
            r["name"], r["expect"], r["risk"], r["consistency_risk"], r["consistency"],
            r["exaggerated"], r["endorsement"], r["error"] or "无"
        ]))
    print("=" * 120)


def main():
    print(f"共 {len(CASES)} 个用例，开始批量测试...\n")
    rows = []
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] 正在执行：{case['name']}")
        rows.append(run_one(case))
    print_table(rows)
    print("\n批量测试完成！")


if __name__ == "__main__":
    main()
