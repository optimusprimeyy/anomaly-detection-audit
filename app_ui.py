import gradio as gr
import random
import time


def run_audit_pipeline(text):
    if len(text.strip()) < 10:
        return "⚠️ 输入太短", 0, 0, "## 提示\n请输入更完整的内容以便进行深度审计。"

    time.sleep(0.8)  # 模拟计算延迟
    # 模拟逻辑：针对 N24News 场景的评分逻辑
    sys_score = random.uniform(0.46, 0.54) if len(text) % 2 == 0 else random.uniform(0.1, 0.3)

    if 0.45 <= sys_score <= 0.55:
        status = "🚩 触发专家审计"
        time.sleep(1.2)
        audit_score = 0.48
        audit_reason = (
            "## 📝 专家深度审计报告\n\n"
            "### 1. 正向推演 (Normal Reasoning)\n"
            "- **语义连贯性**：文本逻辑框架严谨，符合专业新闻报道的叙事风格。\n"
            "- **领域契合度**：使用了大量行业术语，判定为真实的行业动态分享。\n\n"
            "### 2. 反向推演 (Anomaly Reasoning)\n"
            "- **潜在风险**：文中提及的特定链接虽未直接失效，但具有引流倾向。\n"
            "- **逻辑冲突**：在描述成本变动时，数据前后存在 2% 的微小偏差。\n\n"
            "### 3. 最终裁定\n"
            "**判定结果：正常**。该样本的数值波动属于正常的统计误差，并非刻意伪造。系统已自动修正判别器的边界偏差。"
        )
    else:
        status = "✅ 快速通过"
        audit_score = sys_score
        audit_reason = "## ⚙️ 系统自动判定\n\n该样本特征显著，系统置信度高，无需调用专家审计模块。已记录至常规数据库。"

    return status, round(sys_score, 4), round(audit_score, 4), audit_reason


# 自定义高级 CSS：提升高度和字体感
custom_css = """
footer {visibility: hidden} /* 隐藏底部信息，更清爽 */
.gradio-container {max-width: 1400px !important} /* 撑开容器宽度 */
#input-box textarea {font-size: 18px !important; line-height: 1.6 !important;} /* 增大输入框文字 */
#reason-box {min-height: 450px !important; font-size: 16px !important;} /* 强制报告区高度 */
.stat-box {padding: 20px !important;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center; margin-bottom: 20px;'>🛡️ 异常内容多级审计工作台</h1>")

    with gr.Row():
        # 左侧：输入区
        with gr.Column(scale=4):
            input_text = gr.Textbox(
                label="N24News 原文输入",
                placeholder="在此粘贴待检测样本...",
                lines=22,  # 显著增加行数
                elem_id="input-box"
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg")
                submit_btn = gr.Button("🚀 开始多级审计流水线", variant="primary", size="lg")

        # 右侧：结果与审计报告
        with gr.Column(scale=3):
            with gr.Group(elem_classes="stat-box"):
                status_label = gr.Label(label="当前节点状态", value="等待任务...")
                with gr.Row():
                    m1_out = gr.Number(label="ML 融合初分", precision=4, interactive=False)
                    m2_out = gr.Number(label="专家最终得分", precision=4, interactive=False)

            gr.Markdown("---")
            with gr.Accordion("📄 查看详细审计报告", open=True):
                reason_out = gr.Markdown(
                    value="*等待流水线启动后生成报告...*",
                    elem_id="reason-box"
                )

    # 逻辑绑定
    submit_btn.click(
        fn=run_audit_pipeline,
        inputs=input_text,
        outputs=[status_label, m1_out, m2_out, reason_out]
    )
    clear_btn.click(lambda: ["", "等待任务...", None, None, "*等待流水线启动...*"],
                    outputs=[input_text, status_label, m1_out, m2_out, reason_out])

if __name__ == "__main__":
    demo.launch()