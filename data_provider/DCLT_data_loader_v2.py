import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns

original_repr = torch.Tensor.__repr__
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}, {self.device}, {self.dtype}}} {original_repr(self)}'

def show_shape():
    torch.Tensor.__repr__ = custom_repr

show_shape()

class GraphContrastDataset(Dataset):
    """
    基于图的多变量时间序列对比学习数据集构造类
    参数说明：
    - data_path: CSV 文件路径，行表示时间步，列表示不同变量（可包含 'date' 列）
    - dtw_matrix: DTW 距离矩阵（numpy 数组或 CSV 路径，形状为 var_count x var_count）
    - k: 构建 top-k 邻居列表（用于互为 k 近邻）
    - P: 每个 anchor 返回的正样本数量
    - N: 每个 anchor 返回的负样本数量
    - sigma_method: 'self_tuning'（自适应尺度） 或 'global'（全局尺度）
    - self_tuning_k: 计算 sigma_i 时使用的 k（仅在 self_tuning 下使用）
    - global_sigma: 全局 sigma 值（可选）
    - use_mutual: 是否仅保留互为 k 近邻的边
    - neg_sampling: 'uniform'（均匀采样） 或 'hard'（困难负样本采样，基于 1 - 相似度）
    """
    def __init__(self,
                 data_path,
                 dtw_matrix=None,  # numpy 数组 或 CSV 文件路径
                 k=10,
                 P=5,
                 N=20,
                 sigma_method='self_tuning',
                 self_tuning_k=10,
                 global_sigma=None,
                 use_mutual=True,
                 neg_sampling='hard',
                 rng_seed=42):
        self.data_path = data_path
        self.k = k # top-k 邻居数
        self.P = P # 正样本数
        self.N = N # 负样本数
        self.sigma_method = sigma_method # 相似度公式里sigma尺度选择方法
        self.self_tuning_k = self_tuning_k # 自适应尺度时使用的k，默认10，以第10个
        self.global_sigma = global_sigma # 全局sigma值，暂时不考虑
        self.use_mutual = use_mutual # 是否使用mutual
        self.neg_sampling = neg_sampling # 负样本采样方式，uniform或hard，均匀或者优先采样不相似样本
        self.rng = np.random.default_rng(rng_seed)
        self.data_length = 0 # 初始化data_length长度

        # 读取并预处理数据
        self._read_data()
        # 读取 DTW 距离并构建相似度矩阵 / 图结构
        if isinstance(dtw_matrix, str) or dtw_matrix is None:
            if dtw_matrix is None:
                raise ValueError("请提供 dtw_matrix（numpy 数组）或 DTW CSV 文件路径")
            self._read_dtw_from_csv(dtw_matrix) # 目前仅使用csv读取
        else:
            # 如果传入的是 numpy 数组
            self.dtw = np.array(dtw_matrix, dtype=float)
        self._build_similarity()
        self._build_knn_graph()

    def _read_data(self):
        df = pd.read_csv(self.data_path)
        # 如果存在 'date' 列，删除
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        # 行表示时间步，列表示不同变量
        self.data_df = df.copy()
        values = df.values  # 形状 (T, num_vars)
        self.data_length = values.shape[0]  # 数据的长度保存
        self.time_steps, self.num_vars = values.shape
        # 按列标准化（z-score）
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(values)  # 形状保持不变
        # fit_transform进行2个操作
        # 1.拟合：对输入数据values中每个变量(列)的均值和标准差，并存储在self.scaler中
        # 2.转换：使用计算出的均值和标准差，对values进行标准化处理，得到scaled
        self.data = scaled  # 保存 numpy 数组

    def _read_dtw_from_csv(self, path):
        df = pd.read_csv(path, header=0, index_col=0)
        self.dtw = df.values.astype(float)
        # 检查 DTW 矩阵是否方阵且变量数量匹配
        if self.dtw.shape[0] != self.dtw.shape[1]:
            raise ValueError("DTW 矩阵必须是方阵")
        if self.dtw.shape[0] != self.num_vars:
            raise ValueError(f"DTW 矩阵大小 {self.dtw.shape} 与变量数 {self.num_vars} 不匹配")

    def _build_similarity(self):
        # d 为 DTW 距离矩阵
        d = self.dtw.copy()
        np.fill_diagonal(d, 0.0)  # 对角线置 0
        # 获取非对角元素用于尺度计算
        mask = ~np.eye(self.num_vars, dtype=bool)
        all_d = d[mask].reshape(-1) # 根据mask提取非对角元素

        if self.sigma_method == 'global':
            # 全局 RBF 核尺度
            if self.global_sigma is None:
                sigma = max(np.median(all_d), 1e-8)
            else:
                sigma = float(self.global_sigma)
            denom = 2 * (sigma ** 2) + 1e-12
            self.sim = np.exp(- (d ** 2) / denom)
        elif self.sigma_method == 'self_tuning':
            # 自适应尺度：sigma_i = 第 k 个最近邻的距离
            kth = max(1, min(self.self_tuning_k, self.num_vars - 1)) # k近邻数量，最少为1，最大不超过变量数
            sorted_rows = np.sort(d + np.eye(self.num_vars) * 1e12, axis=1)  # 通过给对角元素加上1e12的大值，确保自身不被当作最邻近，然后每行进行排序
            sigma_i = sorted_rows[:, kth-1] # 获取第k个最近邻的距离, 达到自适应的效果，稠密区的sigma_i会小，相似度衰减更快，稀疏区的sigma_i会大，相似度衰减慢
            sigma_i = np.maximum(sigma_i, 1e-12) # 给sigma_i加下界，防止为0或过小
            sigprod = np.outer(sigma_i, sigma_i) + 1e-12 # outer是外积，用于计算相似度的分母
            self.sim = np.exp(- (d ** 2) / sigprod) # 使用公式计算相似度
        else:
            raise ValueError("sigma_method 必须是 'global' 或 'self_tuning'")

        # 对角线置为 1（自己与自己相似度最大）
        np.fill_diagonal(self.sim, 1.0)

    def _build_knn_graph(self):
        """
        构建邻居列表（优先 mutual kNN，但保证每个节点至少有候选邻居）
        改进点：
        - 先取 mutual neighbors（互为近邻）作为高置信候选
        - 若候选数 < min_required（默认取 P），则从 top-k 中补充非 mutual 邻居
        - 如果 top-k 都不可用（极少见），直接取最近若干个作为兜底
        """
        d = self.dtw
        idx_sorted = np.argsort(d, axis=1)
        topk = idx_sorted[:, 1:self.k+1]  # 排除自身，取前 k
        self.adj_lists = [[] for _ in range(self.num_vars)]

        # 期望每个节点至少有的候选数（这里取与 P 的最大值，保证后续能采到 P 个）
        min_required = max(1, self.P)

        for i in range(self.num_vars):
            # 找 mutual neighbors（双方都把对方当 top-k）
            mutual_neighbors = []
            for j in topk[i]:
                if i in topk[j]:
                    mutual_neighbors.append(int(j))

            neighbors = mutual_neighbors.copy()

            # 如果使用 mutual 优先但数目不足，补充 top-k 中非 mutual 的邻居
            if self.use_mutual:
                if len(neighbors) < min_required:
                    for j in topk[i]:
                        if int(j) not in neighbors:
                            neighbors.append(int(j))
                        if len(neighbors) >= min_required:
                            break
            else:
                # 不使用 mutual 时直接用 top-k
                neighbors = [int(j) for j in topk[i]]

            # 再次兜底：如果仍为空（理论上很少发生），直接取全局最近的几个（排除自身）
            if len(neighbors) == 0:
                sorted_idx = np.argsort(d[i])
                fallback = [int(j) for j in sorted_idx if j != i][:min_required]
                neighbors = fallback

            self.adj_lists[i] = neighbors

    def __len__(self):
        return self.num_vars

    def __getitem__(self, idx):
        """
        返回：
            anchor: 张量形状 (1, T)
            pos: 张量形状 (P, T)
            neg: 张量形状 (N, T)
        注意：
        - pos/neg 的长度固定为 self.P / self.N （通过补充或有放回采样保证）
        - 不会把 anchor 自己当作正样本（除非极端兜底）
        """
        anchor = self.data[:, idx][np.newaxis, :].astype(np.float32) # 原始形状是(T, ), 变为(1, T)

        # 正样本候选：先取 adj_lists（已经处理了补充逻辑）
        pos_cands = np.array(self.adj_lists[idx], dtype=int)
        if pos_cands.size == 0:
            raise RuntimeError("正样本没有足够的数据！请检查 _build_knn_graph 的输出。")

        # 负样本候选：除正样本和自身外的所有索引
        all_idx = np.arange(self.num_vars)
        mask = np.ones(self.num_vars, dtype=bool)
        mask[pos_cands] = False
        mask[idx] = False
        neg_cands = all_idx[mask]
        if neg_cands.size == 0:
            raise RuntimeError("负样本没有足够的数据！请检查 _build_knn_graph 的输出。")

        # --------- 采样正样本（按相似度加权） ----------
        P = self.P
        # 计算权重（相似度）
        weights = self.sim[idx, pos_cands]
        if weights.sum() <= 0 or np.allclose(weights, 0):
            probs = None
        else:
            probs = weights / weights.sum()
        # 如果候选数不足 P，则允许有放回采样；否则不放回
        replace = len(pos_cands) < P
        pos_choice = self.rng.choice(pos_cands, size=P, replace=replace, p=probs)
        pos_sample_idx = np.array(pos_choice, dtype=int)

        # --------- 采样负样本 ----------
        N = self.N
        if self.neg_sampling == 'uniform':
            replace = len(neg_cands) < N
            neg_choice = self.rng.choice(neg_cands, size=N, replace=replace)
        else:  # 'hard'：优先采相对较难的负样本（1 - sim 权重）
            weights = 1.0 - self.sim[idx, neg_cands]
            weights = np.clip(weights, 1e-6, None) # clip会限制数组元素的范围，这里设置了下限为1e-6，避免除0错误
            probs = weights / weights.sum()
            replace = len(neg_cands) < N
            neg_choice = self.rng.choice(neg_cands, size=N, replace=replace, p=probs)
        neg_sample_idx = np.array(neg_choice, dtype=int)

        # 收集时序片段并转为 (P, T) / (N, T)
        pos_data = self.data[:, pos_sample_idx].T.astype(np.float32)  # (P, T)
        neg_data = self.data[:, neg_sample_idx].T.astype(np.float32)  # (N, T)

        return torch.from_numpy(anchor), torch.from_numpy(pos_data), torch.from_numpy(neg_data)
    
    

    @staticmethod # 装饰器，标示为静态方法，不会传入self参数，只处理batch
    def collate_fn(batch):
        # 将 batch 中的 anchor、pos、neg 合并成批次张量
        anchors = torch.stack([b[0] for b in batch], dim=0)  # (B, 1, T)
        pos = torch.stack([b[1] for b in batch], dim=0)      # (B, P, T)
        neg = torch.stack([b[2] for b in batch], dim=0)      # (B, N, T)
        return anchors, pos, neg

    def get_sim_matrix(self):
        return self.sim

    def get_adj_lists(self):
        return self.adj_lists


# 假设：
# - electricity.csv: 行=time, 列=variable1,...,variable321
# - dtw.csv: 第一列/行是变量索引或名称，内容是 DTW 距离矩阵
if __name__ == "__main__":
    dataset = GraphContrastDataset(
        data_path="./dataset/electricity.csv",
        dtw_matrix="./DTW_matrix/electricity_dtw_analysis.csv",
        k=10,
        P=5,
        N=20,
        sigma_method='self_tuning',
        self_tuning_k=10,
        use_mutual=True,
        neg_sampling='hard'
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn)

    # for anchors, pos, neg in dataloader:
    #     # anchors: (B, 1, T)
    #     # pos: (B, P, T)
    #     # neg: (B, N, T)
    #     # 送入你的 PatchTST / encoder
    #     print("Batch anchors shape:", anchors.shape)
    #     print("Dataset data length:", dataset.data_length)
    #     print("Batch pos shape:", pos.shape)
    #     print("Batch neg shape:", neg.shape)
    #     break  # 只处理一个批次用于演示

    # 绘制相似度矩阵热力图
    sim_matrix = dataset.get_sim_matrix()
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, cmap='viridis', cbar=True)
    plt.title('Similarity Matrix Heatmap (DTW-based)')
    plt.xlabel('Variable Index')
    plt.ylabel('Variable Index')
    
    # 保存热力图
    heatmap_path = './similarity_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {heatmap_path}")
    
    # 如果在非交互式环境（如脚本）中运行，可能需要关闭图形
    # plt.close()
    
    # 如果在交互式环境（如Jupyter）中运行，可以显示图形
    plt.show()
