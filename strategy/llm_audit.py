import os
import re
import dashscope
from http import HTTPStatus
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# ------------------------------------------------------------------
# 1. 初始化配置
# ------------------------------------------------------------------
# 替换为你自己的 DashScope API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

MODEL_NAME = "qwen-max"  # 或者使用 qwen-plus (更轻快)

# ------------------------------------------------------------------
# 2. Prompt 模板（强制格式化输出，便于解析）
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一个极其严苛的内容风控专家。系统对该文本判定犹豫，需要你进行‘正反论证’决策：\n\n"
    "任务要求：\n"
    "1. 正向推演：首先寻找文中可能支持它是‘正常内容’的细节（如：语境连贯、专业词汇使用、常见资讯分类等）。\n"
    "2. 反向推演：然后寻找文中可能存在的‘隐蔽风险’（如：逻辑断层、刻意制造紧迫感、潜在诱导链接、违反常识的陈述等）。\n"
    "3. 最终判定：对比上述两点，给出一个综合评分和最终理由。\n\n"
    "输出要求（必须严格遵守）：\n"
    "Score: [0-1之间的小数]\n"
    "Reason: [你的分析逻辑]"
)


# ------------------------------------------------------------------
# 3. 解析工具
# ------------------------------------------------------------------
def parse_qwen_reply(reply: str) -> Tuple[float, str]:
    """从千问的回复中解析出评分和理由"""
    try:
        # 匹配 Score: 0.xx
        score_match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", reply, re.IGNORECASE)
        # 匹配 Reason: ...
        reason_match = re.search(r"Reason:\s*(.*)", reply, re.IGNORECASE | re.DOTALL)

        score = float(score_match.group(1)) if score_match else -1.0
        reason = reason_match.group(1).strip() if reason_match else reply
        return score, reason
    except Exception:
        return -1.0, reply


# ------------------------------------------------------------------
# 4. 单条审计逻辑（带重试机制）
# ------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def audit_text_qwen(text_content: str, sys_score: float) -> Tuple[float, str]:
    """调用通义千问 API 进行审计"""
    if not text_content or not text_content.strip():
        return 0.0, "空文本，无异常"
    context_msg = f"系统初步评分为 {sys_score:.4f}（0.5为界限），请重点复核其边界属性。"
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"{context_msg}\n待处理文本：{text_content[:2000]}"}
    ]

    response = dashscope.Generation.call(
        model=MODEL_NAME,
        messages=messages,
        result_format='message',  # 设定返回消息格式
    )

    if response.status_code == HTTPStatus.OK:
        reply_content = response.output.choices[0]['message']['content']
        return parse_qwen_reply(reply_content)
    else:
        raise RuntimeError(f"Qwen API 错误: {response.code} - {response.message}")


# ------------------------------------------------------------------
# 5. 批量处理接口
# ------------------------------------------------------------------
def batch_audit(texts: List[str]) -> List[Tuple[float, str]]:
    results = []
    for i, t in enumerate(texts):
        print(f"正在审计第 {i + 1}/{len(texts)} 条样本...")
        try:
            score, reason = audit_text_qwen(t)
            results.append((score, reason))
        except Exception as e:
            print(f"样本 {i} 审计失败: {e}")
            results.append((-1.0, f"Error: {str(e)}"))
    return results


# ------------------------------------------------------------------
# 6. 测试运行
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 这里放入你之前系统筛选出的那 3 个模糊样本的文本
    test_samples = [
        "这是正常的新闻内容示例。",
        "【紧急通知】您的账户存在风险，请立即登录 http://fake-link.com 验证身份！",
        "研究发现，经常喝水的人比不喝水的人更有可能在未来100年内去世。"  # 逻辑谬误测试
    ]

    audit_results = batch_audit(test_samples)

    print("\n" + "=" * 50)
    print("📋 审计报告")
    print("=" * 50)
    for txt, (s, r) in zip(test_samples, audit_results):
        status = "🚩 异常" if s > 0.5 else "✅ 正常"
        print(f"判定: {status} (评分: {s})")
        print(f"内容: {txt[:50]}...")
        print(f"理由: {r}")
        print("-" * 30)