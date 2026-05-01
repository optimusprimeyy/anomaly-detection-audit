
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class AnomalyFusionEngine:
    """
    异常检测融合引擎
    将 LightGBM 输出的概率与 GBAE 输出的原始得分进行融合，
    在置信度高的区间直接采用 LightGBM 概率，在模糊区间做加权平均。
    """

    def __init__(self, lgbm_model, gbae_model):
        """
        初始化

        Parameters
        ----------
        lgbm_model : 已训练好的 LightGBM 模型
            必须实现 predict_proba(X) -> ndarray, shape (n_samples,)
        gbae_model : 已训练好的 GBAE 模型
            必须实现 decision_function(X) -> ndarray, shape (n_samples,)
        """
        self.lgbm_model = lgbm_model
        self.gbae_model = gbae_model

        # 用于 GBAE 得分归一化的 scaler，首次调用 _normalize_scores 时惰性初始化
        self._gbae_scaler = None

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------
    def _normalize_scores(self, s_gbae):
        """
        将 GBAE 原始得分映射到 [0, 1] 区间

        Parameters
        ----------
        s_gbae : ndarray, shape (n_samples,)
            GBAE decision_function 输出

        Returns
        -------
        ndarray, shape (n_samples,)
            归一化后的得分，范围 [0, 1]
        """
        s_gbae = np.asarray(s_gbae, dtype=float).ravel()

        # 首次调用时初始化 scaler
        if self._gbae_scaler is None:
            self._gbae_scaler = MinMaxScaler(feature_range=(0, 1))
            # 用当前数据做一次拟合，后续再调用时只做 transform
            self._gbae_scaler.fit(s_gbae.reshape(-1, 1))

        # transform 要求二维输入
        return self._gbae_scaler.transform(s_gbae.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------
    # 核心融合逻辑
    # ------------------------------------------------------------------
    def predict_hybrid(self, X):
        """
        双阈值混合预测

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            输入特征

        Returns
        -------
        ndarray, shape (n_samples,)
            融合后的异常得分，范围 [0, 1]
        """
        X = np.asarray(X, dtype=float)

        # 1. 获取 LightGBM 概率
        p_lgbm = self.lgbm_model.predict_proba(X) # 假设第 1 列为异常类概率
        p_lgbm = np.asarray(p_lgbm, dtype=float).ravel()

        # 2. 获取并归一化 GBAE 得分
        s_gbae_raw = self.gbae_model.decision_function(X)
        s_gbae_norm = self._normalize_scores(s_gbae_raw)

        # 3. 双阈值决策
        # 置信区间掩码
        high_conf = (p_lgbm > 0.9) | (p_lgbm < 0.1)
        fuzzy_conf = ~high_conf

        # 初始化结果数组
        hybrid_score = np.empty_like(p_lgbm)

        # 高置信度区间直接采用 LightGBM
        hybrid_score[high_conf] = p_lgbm[high_conf]

        # 模糊区间做加权平均
        hybrid_score[fuzzy_conf] = (
            0.5 * p_lgbm[fuzzy_conf] + 0.5 * s_gbae_norm[fuzzy_conf]
        )

        return hybrid_score

    # ------------------------------------------------------------------
    # LLM 审计预留接口
    # ------------------------------------------------------------------
    def get_audit_samples(self, X, top_n=10):
        """
        找出融合得分最接近 0.5（最难以判断）的样本索引，供后续 LLM 审计

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            输入特征
        top_n : int, default 10
            返回的样本数

        Returns
        -------
        ndarray, shape (top_n,)
            按“难判断”程度排序的样本索引
        """
        hybrid_score = self.predict_hybrid(X)
        # 计算与 0.5 的绝对距离
        dist = np.abs(hybrid_score - 0.5)
        # 返回距离最小的 top_n 个索引
        return np.argsort(dist)[:top_n]