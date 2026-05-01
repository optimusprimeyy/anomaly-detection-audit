import numpy as np
import os
import sys
import torch
import warnings
import pandas as pd
# 导入你的组件
from models.gbae_model import GBAE
from models.lgbm_wrapper import LGBMWrapper
from strategy.fusion_engine import AnomalyFusionEngine
from strategy.llm_audit import audit_text_qwen  # 👈 导入你刚刚跑通的千问函数
from utils.metrics import compute_metrics, print_report

warnings.filterwarnings("ignore", category=UserWarning)

# 2. 针对 KMeans 内存泄漏问题的环境变量设置
# 注意：这行必须在 import sklearn 之前执行才有效
os.environ["OMP_NUM_THREADS"] = "1"

def get_adaptive_audit_indices(scores, low=0.45, high=0.55):
    """筛选出系统最犹豫的区间"""
    # 找出评分在 [low, high] 之间的所有索引
    mask = (scores >= low) & (scores <= high)
    indices = np.where(mask)[0]
    return indices

def main():
    # --- 1. 真实数据加载 ---
    data_dir = r"D:\sihanwang\Study\LLM项目\data"
    print(f"⏳ 正在加载 N24News 真实数据...")

    X_train = np.load(os.path.join(data_dir, "n24news_train_emb.npy")).astype(np.float32)
    y_train = np.load(os.path.join(data_dir, "n24news_train_label.npy")).astype(np.int32)
    X_test = np.load(os.path.join(data_dir, "n24news_test_emb.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "n24news_test_label.npy")).astype(np.int32)


    # 在 main 函数加载数据部分修改：
    df_test = pd.read_csv(os.path.join(data_dir, "n24news_test.csv"))
    test_texts = df_test['text'].values  # 提取文字列
    # 📝 重要：加载测试集的原始文本（用于 LLM 审计）
    # 假设你的文本数据存放在 n24news_test_text.npy 或类似文件中

    X_train_normal = X_train[y_train == 0]
    print(f"📊 数据就绪: 训练集(正常) {X_train_normal.shape[0]} | 测试集 {X_test.shape[0]}")

    # --- 2. 模型训练 (保持不变) ---
    print("\n🚀 正在训练 LightGBM 判别器...")
    lgbm = LGBMWrapper(params={"n_estimators": 100, "learning_rate": 0.05, "verbose": -1})
    lgbm.fit(X_train, y_train)

    print("\n🚀 正在启动 GBAE 粒球自编码器...")
    gbae = GBAE(delta=0.1, epochs=50, batch_size=32, latent_dim=16)
    gbae.fit(X_train_normal)

    # --- 3. 策略融合 ---
    print("\n🧠 运行置信度自适应融合策略...")
    engine = AnomalyFusionEngine(lgbm, gbae)
    hybrid_scores = engine.predict_hybrid(X_test)

    # --- 4. 自动化评估 ---
    results = compute_metrics(y_test, hybrid_scores)
    print_report(results, title="N24News 实时检测指标")

    # --- 5. 🛠️ 改进：LLM 专家深度审计 ---
    print("\n🧠 正在根据[0.45, 0.55]区间动态触发LLM审计...")
    audit_indices = get_adaptive_audit_indices(hybrid_scores)
    if len(audit_indices) > 10:
        print(f"⚠️ 模糊样本过多({len(audit_indices)})，仅审计前10个。")
        audit_indices = audit_indices[:10]

    for idx in audit_indices:
        raw_text = test_texts[idx]
        sys_score = hybrid_scores[idx]
        true_label = "异常" if y_test[idx] == 1 else "正常"

        print(f"\n[待审样本 ID: {idx}] | 系统融合评分: {sys_score:.4f} | 真实标签: {true_label}")
        print(f"📄 文本内容提取: {raw_text[:100]}...")

        # 调用千问接口
        print("🤖 通义千问正在深度分析...")
        llm_score, llm_reason = audit_text_qwen(raw_text, sys_score)

        # 输出审计报告
        status_icon = "🚩" if llm_score > 0.5 else "✅"
        print(f"{status_icon} LLM 审计判定: {'异常' if llm_score > 0.5 else '正常'} (得分: {llm_score})")
        print(f"💡 审计理由: {llm_reason}")
        print("-" * 40)

    print("\n🎉 整个检测与审计工作流完成！")


if __name__ == "__main__":
    main()