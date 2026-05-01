# Multi-Tier Anomaly Detection & LLM Audit System
本项目构建了一套面向大规模高维数据的**多级异常检测与深度审计流水线**。系统集成了统计机器学习、粒计算以及大语言模型（LLM）的推演能力，旨在解决传统异常检测模型在“边界样本”上误报率高且判定逻辑不可见的问题。

## 🌟 核心亮点 (Key Highlights)

*   **三级防御架构**：结合有监督 LightGBM 与无监督 GBAE（粒球自编码器）进行双路特征对齐，最后由 LLM 进行专家级终审。
*   **自适应调度策略**：设计了置信度感知调度机制，仅对置信度处于 $[0.45, 0.55]$ 的模糊样本触发审计，有效平衡了检测精度与算力成本。
*   **正反论证 Prompt 审计**：创新性地在 LLM 审计环节引入“双向推演策略”，通过对比正向合理性与反向异常性，消除大模型的决策偏见。
*   **工业级可解释性**：针对高风险样本自动生成语义级的审计报告，涵盖逻辑自洽性、潜在风险分析及最终裁定。

---

## 🏗️ 系统架构 (Architecture)

系统由以下三个核心模块组成：

1.  **初筛层 (Detector Layer)**:
    *   **LightGBM**: 快速捕获已知攻击模式和高维统计特征。
    *   **GBAE (Granular Ball AutoEncoder)**: 利用粒计算理论刻画正常数据的流形，检测未知异构模式。
2.  **调度层 (Fusion & Adaptive Engine)**:
    *   基于动态置信度区间判断样本是否具有“决策歧义”，实现算力的精准分配。
3.  **专家层 (LLM Audit Layer)**:
    *   基于通义千问 (Qwen) 的语义推演模块，输出正反论证逻辑报告。

---
## 📂 数据集说明 (Dataset)
本项目采用 **N24News** 数据集进行验证。该数据集包含 2.4w+ 条来自《纽约时报》的真实新闻，涵盖 24 个细分领域。其复杂的语义背景和高度相似的边界样本，充分验证了本系统在处理“决策歧义”时的优越性。

---

## 📊 实验表现 (Performance)

在 **N24News (2.4w+ samples)** 真实数据集上测试结果如下：

| Metric | Score |
| :--- | :--- |
| **ROC_AUC** | **0.9897** |
| **PR_AUC** | **0.9176** |
| **TNR (Specificity)** | **0.9828** |
| **F1_Score** | **0.8274** |

*注：通过 LLM 审计纠偏，系统在保持高召回的同时，显著提升了特异性 (TNR)，将误报率控制在极低水平。*

---

## 🚀 快速开始 (Quick Start)

### 环境配置
```bash
# 克隆仓库
git clone https://github.com/your-username/anomaly-detection-audit.git
cd anomaly-detection-audit

# 安装依赖
pip install -r requirements.txt
```

### 运行演示界面
本项目提供了基于 Gradio 的交互式审计工作台，可直观体验多级检测流程。
```bash
python app_ui.py
```

---

## 🖥️ 界面展示 (Interface)

系统界面支持全屏沉浸式审计，左侧输入原始文本，右侧实时展示 ML 评分、系统调度状态及 LLM 深度报告。

![Gradio UI Demo](image_5b12ed.png)

---

## 📂 项目结构 (Project Structure)

```text
├── models/             # 模型定义 (GBAE, LGBM Wrapper)
├── strategy/           # 核心策略 (融合引擎, LLM 审计逻辑)
├── utils/              # 工具函数 (指标计算, 预处理)
├── data/               # 数据集管理
├── app_ui.py           # Gradio 可视化工作台
└── main.py             # 离线自动化测试流水线
```

---

## 💡 技术细节 (Technical Deep Dive)

### 自适应触发逻辑 (Adaptive Trigger)
系统并非盲目审计，而是通过以下逻辑进行资源调度：
```python
if 0.45 <= hybrid_score <= 0.55:
    # 触发 LLM 专家审计
    audit_report = expert_audit(text)
else:
    # 快速通道：直接输出 ML 判定结果
    final_output = hybrid_score
```

### 正反论证 Prompt 示例
```markdown
1. 正向推演：寻找支持该文本为“正常内容”的语义细节。
2. 反向推演：识别文中隐蔽的逻辑断层或风险因素。
3. 最终判定：综合权衡输出分值。
```

