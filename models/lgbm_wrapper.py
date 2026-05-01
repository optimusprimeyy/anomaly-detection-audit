import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None):
        # 默认参数中加入 verbose=-1 保持终端整洁
        self.params = params if params is not None else {"verbose": -1}
        self._clf = None
        self._feature_importance = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if n_pos == 0:
            raise ValueError("训练集中没有异常样本（Label=1），无法训练有监督模型。")

        fit_params = self.params.copy()

        # 自动处理不平衡：如果用户没传权重参数，我们根据数据比例自动计算
        if "is_unbalance" not in fit_params and "scale_pos_weight" not in fit_params:
            fit_params["scale_pos_weight"] = n_neg / n_pos

        self._clf = lgb.LGBMClassifier(**fit_params)
        self._clf.fit(X, y)

        self._feature_importance = self._clf.feature_importances_
        return self

    def predict_proba(self, X):
        if self._clf is None:
            raise ValueError("模型尚未训练！")
        X = np.asarray(X, dtype=np.float32)
        return self._clf.predict_proba(X)[:, 1]

    def get_top_features(self, n=10):
        if self._feature_importance is None:
            raise ValueError("模型尚未训练，无法获取特征重要性。")
        # 返回前 n 个最重要的特征维度索引及其分值
        idx = np.argsort(self._feature_importance)[::-1][:n]
        return idx, self._feature_importance[idx]