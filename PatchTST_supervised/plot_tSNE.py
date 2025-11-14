"""
绘制 PatchTST 和 PCLE embeddings 的 t-SNE 可视化
从 data_outputs 目录读取保存的 embeddings 进行对比可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入 t-SNE
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def plot_tsne_comparison(embeddings, title='t-SNE Visualization', save_path=None, 
                        labels=None, max_samples=10000, perplexity=30):
    """
    使用 t-SNE 对 embeddings 进行降维可视化
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title: str, 图表标题
        save_path: str, 保存路径（如果为None则显示图片）
        labels: np.ndarray, 用于着色的标签，shape [B*C*N]
        max_samples: int, 最大采样点数（t-SNE计算很慢，建议限制样本数）
        perplexity: int, t-SNE perplexity参数（建议5-50之间）
    
    Returns:
        embedding_tsne: np.ndarray, t-SNE降维后的2D坐标 [n_samples, 2]
    """
    if not TSNE_AVAILABLE:
        print("❌ 未安装 scikit-learn 库")
        print("请运行: pip install scikit-learn")
        return None
    
    # 转换为 numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # 获取形状信息
    B, C, N, P = embeddings.shape
    print(f"原始形状: B={B}, C={C}, N={N}, P={P}")
    
    # Reshape 为 2D: [B*C*N, P]
    embeddings_2d = embeddings.reshape(-1, P)
    total_samples = embeddings_2d.shape[0]
    print(f"Reshape后: {embeddings_2d.shape}")
    
    # 随机采样（如果样本太多）
    if total_samples > max_samples:
        np.random.seed(42)  # 固定随机种子保证可重复性
        indices = np.random.choice(total_samples, max_samples, replace=False)
        embeddings_2d = embeddings_2d[indices]
        if labels is not None:
            labels = labels[indices]
        print(f"采样到 {max_samples}/{total_samples} 个点")
    else:
        indices = None
    
    # 调整 perplexity（不能大于样本数-1）
    actual_perplexity = min(perplexity, embeddings_2d.shape[0] - 1)
    if actual_perplexity < perplexity:
        print(f"⚠️  Perplexity 调整为 {actual_perplexity}（样本数限制）")
    
    # t-SNE 降维
    print(f"正在进行 t-SNE 降维 (perplexity={actual_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        learning_rate='auto',
        init='pca',
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    
    embedding_tsne = tsne.fit_transform(embeddings_2d)
    print(f"✓ t-SNE 完成: {embedding_tsne.shape}")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None and len(labels) == embeddings_2d.shape[0]:
        # 使用提供的标签着色
        scatter = ax.scatter(
            embedding_tsne[:, 0], 
            embedding_tsne[:, 1], 
            c=labels, 
            cmap='tab10', 
            s=5, 
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        # 为每个 (channel, patch) 组合分配唯一ID
        if indices is not None:
            # 生成组合标签: channel_id * N + patch_id
            channel_ids = np.repeat(np.arange(C), N)
            patch_ids = np.tile(np.arange(N), C)
            var_labels = np.tile(channel_ids * N + patch_ids, B)
            var_labels = var_labels[indices]
        else:
            channel_ids = np.repeat(np.arange(C), N)
            patch_ids = np.tile(np.arange(N), C)
            var_labels = np.tile(channel_ids * N + patch_ids, B)
        
        scatter = ax.scatter(
            embedding_tsne[:, 0], 
            embedding_tsne[:, 1], 
            c=var_labels[:len(embedding_tsne)], 
            cmap='tab10', 
            s=5, 
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Variable Index')
    
    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图片已保存: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return embedding_tsne


def plot_side_by_side_comparison(patchtst_emb, pcle_emb, save_dir='./data_outputs', 
                                 max_samples=5000, perplexity=30):
    """
    并排对比 PatchTST 和 PCLE 的 t-SNE 可视化
    
    Args:
        patchtst_emb: torch.Tensor, PatchTST embeddings [B, C, N, P]
        pcle_emb: torch.Tensor, PCLE embeddings [B, C, N, P']
        save_dir: str, 保存目录
        max_samples: int, 每个模型最大采样点数
        perplexity: int, t-SNE perplexity参数
    """
    if not TSNE_AVAILABLE:
        print("❌ 未安装 scikit-learn 库")
        print("请运行: pip install scikit-learn")
        return
    
    # 转换为 numpy
    if isinstance(patchtst_emb, torch.Tensor):
        patchtst_emb = patchtst_emb.detach().cpu().numpy()
    if isinstance(pcle_emb, torch.Tensor):
        pcle_emb = pcle_emb.detach().cpu().numpy()
    
    B1, C1, N1, P1 = patchtst_emb.shape
    B2, C2, N2, P2 = pcle_emb.shape
    
    print(f"PatchTST embeddings: B={B1}, C={C1}, N={N1}, P={P1}")
    print(f"PCLE embeddings: B={B2}, C={C2}, N={N2}, P={P2}")
    
    # Reshape
    patchtst_2d = patchtst_emb.reshape(-1, P1)
    pcle_2d = pcle_emb.reshape(-1, P2)
    
    # 采样
    np.random.seed(42)
    if patchtst_2d.shape[0] > max_samples:
        indices1 = np.random.choice(patchtst_2d.shape[0], max_samples, replace=False)
        patchtst_2d = patchtst_2d[indices1]
        print(f"PatchTST 采样: {max_samples}/{patchtst_emb.reshape(-1, P1).shape[0]}")
    else:
        indices1 = None
    
    if pcle_2d.shape[0] > max_samples:
        indices2 = np.random.choice(pcle_2d.shape[0], max_samples, replace=False)
        pcle_2d = pcle_2d[indices2]
        print(f"PCLE 采样: {max_samples}/{pcle_emb.reshape(-1, P2).shape[0]}")
    else:
        indices2 = None
    
    # 创建标签（按 channel * N + patch 组合着色）
    channel_ids1 = np.repeat(np.arange(C1), N1)
    patch_ids1 = np.tile(np.arange(N1), C1)
    patchtst_labels = np.tile(channel_ids1 * N1 + patch_ids1, B1)
    if indices1 is not None:
        patchtst_labels = patchtst_labels[indices1]
    else:
        patchtst_labels = patchtst_labels[:len(patchtst_2d)]
    
    channel_ids2 = np.repeat(np.arange(C2), N2)
    patch_ids2 = np.tile(np.arange(N2), C2)
    pcle_labels = np.tile(channel_ids2 * N2 + patch_ids2, B2)
    if indices2 is not None:
        pcle_labels = pcle_labels[indices2]
    else:
        pcle_labels = pcle_labels[:len(pcle_2d)]
    
    # 调整 perplexity
    perplexity1 = min(perplexity, patchtst_2d.shape[0] - 1)
    perplexity2 = min(perplexity, pcle_2d.shape[0] - 1)
    
    # t-SNE 降维
    print(f"\n正在对 PatchTST embeddings 进行 t-SNE 降维 (perplexity={perplexity1})...")
    tsne1 = TSNE(
        n_components=2,
        perplexity=perplexity1,
        learning_rate='auto',
        init='pca',
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    patchtst_tsne = tsne1.fit_transform(patchtst_2d)
    
    print(f"\n正在对 PCLE embeddings 进行 t-SNE 降维 (perplexity={perplexity2})...")
    tsne2 = TSNE(
        n_components=2,
        perplexity=perplexity2,
        learning_rate='auto',
        init='pca',
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    pcle_tsne = tsne2.fit_transform(pcle_2d)
    
    # 创建并排对比图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # PatchTST
    scatter1 = axes[0].scatter(
        patchtst_tsne[:, 0], 
        patchtst_tsne[:, 1], 
        c=patchtst_labels, 
        cmap='tab10', 
        s=5, 
        alpha=0.6
    )
    axes[0].set_xlabel('t-SNE 1', fontsize=14)
    axes[0].set_ylabel('t-SNE 2', fontsize=14)
    axes[0].set_title(f'PatchTST Embeddings (dim={P1})', fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Variable Index')
    
    # PCLE
    scatter2 = axes[1].scatter(
        pcle_tsne[:, 0], 
        pcle_tsne[:, 1], 
        c=pcle_labels, 
        cmap='tab10', 
        s=5, 
        alpha=0.6
    )
    axes[1].set_xlabel('t-SNE 1', fontsize=14)
    axes[1].set_ylabel('t-SNE 2', fontsize=14)
    axes[1].set_title(f'PCLE Embeddings (dim={P2})', fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Variable Index')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'tsne_comparison_patchtst_vs_pcle.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {save_path}")
    plt.close()


def main():
    """
    主函数：读取 data_outputs 中的 embeddings 并绘制 t-SNE
    """
    print("=" * 80)
    print("PatchTST vs PCLE Embeddings t-SNE 可视化")
    print("=" * 80)
    
    # 设置路径
    data_dir = './data_outputs'
    patchtst_path = os.path.join(data_dir, 'patchtst_embeddings_02.pt')
    pcle_path = os.path.join(data_dir, 'pcle_embeddings_02.pt')
    
    # 检查文件是否存在
    if not os.path.exists(patchtst_path):
        print(f"❌ 文件不存在: {patchtst_path}")
        return
    if not os.path.exists(pcle_path):
        print(f"❌ 文件不存在: {pcle_path}")
        return
    
    print(f"\n读取文件:")
    print(f"  - {patchtst_path}")
    print(f"  - {pcle_path}")
    
    # 读取 embeddings
    print("\n" + "-" * 80)
    print("加载 PatchTST embeddings...")
    patchtst_data = torch.load(patchtst_path, map_location='cpu')
    
    # 处理可能的数据格式
    if isinstance(patchtst_data, dict):
        patchtst_emb = patchtst_data.get('embeddings', patchtst_data)
        print(f"  数据格式: dict, keys={list(patchtst_data.keys())}")
    else:
        patchtst_emb = patchtst_data
        print(f"  数据格式: tensor")
    
    print(f"  形状: {patchtst_emb.shape}")
    
    print("\n加载 PCLE embeddings...")
    pcle_data = torch.load(pcle_path, map_location='cpu')
    
    if isinstance(pcle_data, dict):
        pcle_emb = pcle_data.get('embeddings', pcle_data)
        print(f"  数据格式: dict, keys={list(pcle_data.keys())}")
    else:
        pcle_emb = pcle_data
        print(f"  数据格式: tensor")
    
    print(f"  形状: {pcle_emb.shape}")
    
    # 验证形状
    if len(patchtst_emb.shape) != 4:
        print(f"⚠️  PatchTST embeddings 形状不是 4D: {patchtst_emb.shape}")
    if len(pcle_emb.shape) != 4:
        print(f"⚠️  PCLE embeddings 形状不是 4D: {pcle_emb.shape}")
    
    # 绘制单独的 t-SNE 图
    print("\n" + "-" * 80)
    print("绘制单独的 t-SNE 图...")
    
    # PatchTST
    print("\n处理 PatchTST embeddings...")
    plot_tsne_comparison(
        patchtst_emb,
        title='PatchTST Embeddings t-SNE',
        save_path=os.path.join(data_dir, 'tsne_patchtst.png'),
        max_samples=10000,  # t-SNE 建议样本数不要太大
        perplexity=30
    )
    
    # PCLE
    print("\n处理 PCLE embeddings...")
    plot_tsne_comparison(
        pcle_emb,
        title='PCLE Embeddings t-SNE',
        save_path=os.path.join(data_dir, 'tsne_pcle.png'),
        max_samples=10000,
        perplexity=30
    )
    
    # 绘制并排对比图
    print("\n" + "-" * 80)
    print("绘制并排对比图...")
    plot_side_by_side_comparison(
        patchtst_emb,
        pcle_emb,
        save_dir=data_dir,
        max_samples=10000,
        perplexity=30
    )
    
    print("\n" + "=" * 80)
    print("✓ 所有 t-SNE 可视化完成！")
    print(f"输出目录: {data_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
