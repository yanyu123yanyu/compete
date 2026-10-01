# test_track2.py
# 赛题二方案测试脚本
# 放在 欧莱雅比赛 目录下
# 用于快速验证内容鉴真功能

from content_verifier import verify_content, verify_content_with_image, print_report

# ============================================================
# 测试用例 1: 纯文本鉴真（不需要多模态模型）
# ============================================================
def test_text_only():
    """测试纯文本内容鉴真"""

    # 示例1: 典型的夸大功效种草文案
    suspicious_text = """姐妹们！这款精华液真的绝了！用了七天白三个度，毛孔全部隐形！
成分全是进口贵妇成分，比某蓝之谜还厉害！
有皮肤科医生推荐，国家药监局认证，敏感肌也能用！
已经回购第三瓶了，真的不夸张，不好用你来打我！
链接放在下面了，趁还没涨价赶紧冲！"""

    print("=" * 60)
    print("测试1: 纯文本鉴真 - 夸大功效文案")
    print("=" * 60)

    result = verify_content(suspicious_text, model="qwen-plus")
    print_report(result)
    return result


# ============================================================
# 测试用例 2: 图文鉴真（需要多模态模型）
# ============================================================
def test_image(text, image_path):
    """测试图文联合鉴真"""
    print("=" * 60)
    print("测试2: 图文鉴真")
    print("文案: " + text[:60] + "...")
    print("图片: " + image_path)
    print("=" * 60)

    try:
        result = verify_content_with_image(text, image_path, model="qwen3.8-flash")
        print_report(result)
        return result
    except Exception as e:
        print("[多模态调用失败] " + str(e))
        print("提示: 请确认控制台已开通 qwen3.8-flash 或 qwen-vl-plus 模型")
        print("尝试切换到纯文本模式...")
        result = verify_content(text, model="qwen-plus")
        print_report(result)
        return result


# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    import sys

    print("赛题二「信任守护师」方案测试")
    print("-" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        # 图文测试模式
        if len(sys.argv) >= 4:
            text = sys.argv[2]
            image_path = sys.argv[3]
            test_image(text, image_path)
        else:
            print("用法: python test_track2.py --image <文案> <图片路径>")
    else:
        # 纯文本测试
        test_text_only()

    print("\n测试完成!")
