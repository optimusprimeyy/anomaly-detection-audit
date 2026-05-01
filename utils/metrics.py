
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix

def compute_metrics(y_true, y_score, threshold=0.5):
    """
    计算异常检测的关键指标
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 1. AUC-ROC
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except:
        roc_auc = 0.0
        
    # 2. Precision-Recall AUC (Average Precision)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # 3. 基于阈值的指标 (F1, G-means)
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    
    # 防止混淆矩阵在单类情况下报错
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # 处理全 0 或 全 1 的极端情况
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    g_means = np.sqrt(tpr * tnr)
    
    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "F1_Score": f1,
        "G_Means": g_means,
        "TPR (Recall)": tpr,
        "TNR (Specificity)": tnr
    }

def print_report(results, title="模型性能分析报告"):
    print(f"\n{'='*10} {title} {'='*10}")
    for k, v in results.items():
        print(f"{k:15}: {v:.4f}")
    print('='*40)
