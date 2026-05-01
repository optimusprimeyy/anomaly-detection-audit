import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import k_means
from tqdm import tqdm


# --- 内部逻辑：粒球生成 (源自 GBshengcheng_v2.py) ---
def get_SD(gb):
    data = gb[:, :-1]
    center = data.mean(0)
    SD = np.sum(((data - center) ** 2).sum(axis=1) ** 0.5)
    return SD


def spilt_ball(gb):
    data = gb[:, :-1]
    cluster = k_means(X=data, init='k-means++', n_clusters=2)[1]
    return [gb[cluster == 0, :], gb[cluster == 1, :]]


def division(gb_list, sample_threshold):
    gb_list_new = []
    for gb in gb_list:
        if gb.shape[0] >= sample_threshold:
            ball_1, ball_2 = spilt_ball(gb)
            SD_original = get_SD(gb)
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(ball_1 if len(ball_1) > 0 else ball_2)
                continue
            if (get_SD(ball_1) + get_SD(ball_2)) < SD_original:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new


# --- 核心模型类 ---
class GBAE(BaseEstimator, OutlierMixin):
    def __init__(self, delta=0.1, latent_dim=16, epochs=100, batch_size=32, lambda_de=0.5, lr=0.001):
        self.delta = delta
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_de = lambda_de
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.center_data_train = None

    def _get_granular_balls(self, X):
        sample_threshold = max(int(self.delta * len(X)), 2)
        index = np.arange(0, X.shape[0], 1)
        data_index = np.insert(X, X.shape[1], values=index, axis=1)
        gb_list = [data_index]
        while True:
            old_len = len(gb_list)
            gb_list = division(gb_list, sample_threshold)
            if len(gb_list) == old_len: break

        centers = []
        sample_to_center = np.zeros((len(X), X.shape[1]))
        for gb in gb_list:
            c = gb[:, :-1].mean(0)
            centers.append(c)
            indices = gb[:, -1].astype(int)
            sample_to_center[indices] = c
        return np.array(centers), sample_to_center

    def fit(self, X, y=None):
        # 1. 粒球划分
        self.center_data_train, sample_centers = self._get_granular_balls(X)

        # 2. 初始化网络 (DisentangledGBAE 结构)
        from .gbae_model import DisentangledGBAE_Net  # 内部定义
        self.model = DisentangledGBAE_Net(X.shape[1], self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 3. 训练循环
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        c_tensor = torch.tensor(sample_centers, dtype=torch.float32).to(self.device)

        self.model.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                idx = permutation[i:i + self.batch_size]
                optimizer.zero_grad()
                rl, rg, zl, zg = self.model(x_tensor[idx], c_tensor[idx])
                loss = F.mse_loss(rl, x_tensor[idx]) + F.mse_loss(rg, c_tensor[idx])
                # 简化版正交损失
                loss += self.lambda_de * torch.mean(torch.sum(F.normalize(zl) * F.normalize(zg), dim=1) ** 2)
                loss.backward()
                optimizer.step()
        return self

    def decision_function(self, X):
        self.model.eval()
        x_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        # 为测试集匹配最近的训练集粒球中心
        test_centers = []
        for x in X:
            dist = np.linalg.norm(self.center_data_train - x, axis=1)
            test_centers.append(self.center_data_train[np.argmin(dist)])
        c_t = torch.tensor(np.array(test_centers), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            rl, rg, zl, zg = self.model(x_t, c_t)
            recon_error = torch.mean((rl - x_t) ** 2, dim=1).cpu().numpy()
            latent_dist = torch.norm(zl - zg, p=2, dim=1).cpu().numpy()

        # 综合评分：重构误差 + 空间偏离
        return recon_error + latent_dist


class DisentangledGBAE_Net(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc_l = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.LeakyReLU(),
                                   nn.Linear(input_dim // 2, latent_dim))
        self.dec_l = nn.Sequential(nn.Linear(latent_dim, input_dim // 2), nn.LeakyReLU(),
                                   nn.Linear(input_dim // 2, input_dim))
        self.enc_g = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.LeakyReLU(),
                                   nn.Linear(input_dim // 2, latent_dim))
        self.dec_g = nn.Sequential(nn.Linear(latent_dim, input_dim // 2), nn.LeakyReLU(),
                                   nn.Linear(input_dim // 2, input_dim))

    def forward(self, x, c):
        zl, zg = self.enc_l(x), self.enc_g(c)
        return self.dec_l(zl), self.dec_g(zg), zl, zg