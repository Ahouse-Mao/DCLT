"""
绘制 PatchTST 和 PCLE embeddings 的 UMAP 可视化
按 embedding 的每个维度值进行着色，分析不同维度的语义信息
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 正确导入 UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap.umap_ as umap
        UMAP = umap.UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False


def plot_umap_by_dimensions(embeddings, title_prefix='UMAP', save_dir='./data_outputs', 
                            max_samples=10000, selected_dims=None):
    """
    使用 UMAP 对 embeddings 进行降维，并按每个维度的值进行着色
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title_prefix: str, 图表标题前缀
        save_dir: str, 保存目录
        max_samples: int, 最大采样点数
        selected_dims: list, 选择要可视化的维度索引，如果为None则全部可视化
    
    Returns:
        embedding_umap: np.ndarray, UMAP降维后的2D坐标 [n_samples, 2]
        embeddings_sampled: np.ndarray, 采样后的原始embeddings [n_samples, P]
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        print("请运行: pip install umap-learn")
        return None, None
    
    # 转换为 numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # 获取形状信息
    B, C, N, P = embeddings.shape
    print(f"原始形状: B={B}, C={C}, N={N}, P={P} (特征维度)")
    
    # Reshape 为 2D: [B*C*N, P]
    embeddings_2d = embeddings.reshape(-1, P)
    total_samples = embeddings_2d.shape[0]
    print(f"Reshape后: {embeddings_2d.shape}")
    
    # 随机采样（如果样本太多）
    if total_samples > max_samples:
        np.random.seed(42)
        indices = np.random.choice(total_samples, max_samples, replace=False)
        embeddings_sampled = embeddings_2d[indices]
        print(f"采样到 {max_samples}/{total_samples} 个点")
    else:
        embeddings_sampled = embeddings_2d
        indices = None
    
    # UMAP 降维（只做一次）
    print("正在进行 UMAP 降维...")
    reducer = UMAP(
        n_neighbors=30, 
        min_dist=0.4,
        n_components=2, 
        random_state=42,
        verbose=True
    )
    
    embedding_umap = reducer.fit_transform(embeddings_sampled)
    print(f"✓ UMAP 完成: {embedding_umap.shape}")
    
    # 确定要可视化的维度
    if selected_dims is None:
        selected_dims = list(range(P))
    else:
        selected_dims = [d for d in selected_dims if 0 <= d < P]
    
    print(f"\n将为 {len(selected_dims)} 个维度生成可视化图...")
    
    # 为每个维度生成单独的图
    for dim_idx in selected_dims:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用该维度的值作为颜色
        dim_values = embeddings_sampled[:, dim_idx]
        
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            c=dim_values, 
            cmap='viridis',  # 使用连续色彩映射
            s=5, 
            alpha=0.6
        )
        
        cbar = plt.colorbar(scatter, ax=ax, label=f'Dimension {dim_idx} Value')
        cbar.ax.tick_params(labelsize=10)
        
        ax.set_xlabel('UMAP 1', fontsize=14)
        ax.set_ylabel('UMAP 2', fontsize=14)
        ax.set_title(f'{title_prefix} - Dimension {dim_idx}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        dim_min, dim_max = dim_values.min(), dim_values.max()
        dim_mean, dim_std = dim_values.mean(), dim_values.std()
        stats_text = f'Min: {dim_min:.3f}\nMax: {dim_max:.3f}\nMean: {dim_mean:.3f}\nStd: {dim_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(save_dir, f'umap_dim_{dim_idx:03d}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 维度 {dim_idx:3d}: {save_path}")
        plt.close()
    
    return embedding_umap, embeddings_sampled


def plot_umap_grid_comparison(embeddings, title_prefix='UMAP', save_path=None, 
                              max_samples=10000, num_dims=16):
    """
    在一个网格图中展示多个维度的 UMAP 可视化
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title_prefix: str, 图表标题前缀
        save_path: str, 保存路径
        max_samples: int, 最大采样点数
        num_dims: int, 要展示的维度数量（必须是完全平方数，如 4, 9, 16, 25）
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        return None
    
    # 转换为 numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    B, C, N, P = embeddings.shape
    print(f"\n原始形状: B={B}, C={C}, N={N}, P={P}")
    
    # Reshape 和采样
    embeddings_2d = embeddings.reshape(-1, P)
    if embeddings_2d.shape[0] > max_samples:
        np.random.seed(42)
        indices = np.random.choice(embeddings_2d.shape[0], max_samples, replace=False)
        embeddings_sampled = embeddings_2d[indices]
        print(f"采样到 {max_samples}/{embeddings_2d.shape[0]} 个点")
    else:
        embeddings_sampled = embeddings_2d
    
    # UMAP 降维
    print("正在进行 UMAP 降维...")
    reducer = UMAP(n_neighbors=30, min_dist=0.4, n_components=2, random_state=42, verbose=True)
    embedding_umap = reducer.fit_transform(embeddings_sampled)
    print(f"✓ UMAP 完成: {embedding_umap.shape}")
    
    # 选择要展示的维度（均匀分布）
    selected_dims = np.linspace(0, P-1, num_dims, dtype=int)
    
    # 计算网格大小
    grid_size = int(np.sqrt(num_dims))
    if grid_size * grid_size != num_dims:
        print(f"⚠️  num_dims={num_dims} 不是完全平方数，将调整为 {grid_size**2}")
        num_dims = grid_size * grid_size
        selected_dims = selected_dims[:num_dims]
    
    # 创建网格图
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4*grid_size, 3.5*grid_size))
    axes = axes.flatten()
    
    print(f"\n生成 {grid_size}x{grid_size} 网格图，展示 {num_dims} 个维度...")
    
    for i, dim_idx in enumerate(selected_dims):
        ax = axes[i]
        dim_values = embeddings_sampled[:, dim_idx]
        
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            c=dim_values, 
            cmap='viridis', 
            s=3, 
            alpha=0.5
        )
        
        ax.set_title(f'Dim {dim_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.ax.tick_params(labelsize=7)
        
        print(f"  ✓ 维度 {dim_idx} 完成")
    
    plt.suptitle(f'{title_prefix} - Dimension-wise Visualization', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 网格图已保存: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    """
    主函数：读取 embeddings 并按维度可视化
    """
    print("=" * 80)
    print("Embeddings 维度语义分析 - UMAP 可视化")
    print("=" * 80)
    
    # 设置路径
    data_dir = './data_outputs'
    patchtst_path = os.path.join(data_dir, 'patchtst_embeddings_02.pt')
    pcle_path = os.path.join(data_dir, 'pcle_embeddings_02.pt')
    
    # 创建子目录保存单个维度图
    patchtst_dim_dir = os.path.join(data_dir, 'patchtst_dimensions')
    pcle_dim_dir = os.path.join(data_dir, 'pcle_dimensions')
    os.makedirs(patchtst_dim_dir, exist_ok=True)
    os.makedirs(pcle_dim_dir, exist_ok=True)
    
    # 读取 PatchTST embeddings
    if os.path.exists(patchtst_path):
        print("\n" + "-" * 80)
        print("加载 PatchTST embeddings...")
        patchtst_data = torch.load(patchtst_path, map_location='cpu')
        
        if isinstance(patchtst_data, dict):
            patchtst_emb = patchtst_data.get('embeddings', patchtst_data)
        else:
            patchtst_emb = patchtst_data
        
        print(f"  形状: {patchtst_emb.shape}")
        B, C, N, P = patchtst_emb.shape
        
        # 方式1: 生成网格对比图（快速查看）
        print("\n生成 PatchTST 网格对比图...")
        plot_umap_grid_comparison(
            patchtst_emb,
            title_prefix='PatchTST Embeddings',
            save_path=os.path.join(data_dir, 'umap_patchtst_grid_dims.png'),
            max_samples=30000,
            num_dims=16  # 4x4 网格
        )
        
        # 方式2: 为每个维度生成单独的图（可选，如果维度不多）
        # 如果维度太多（如128维），建议只可视化部分维度
        if P <= 32:
            print(f"\n为 PatchTST 的所有 {P} 个维度生成单独图...")
            plot_umap_by_dimensions(
                patchtst_emb,
                title_prefix='PatchTST',
                save_dir=patchtst_dim_dir,
                max_samples=30000,
                selected_dims=None  # 全部维度
            )
        else:
            # 只可视化部分维度（如前16个）
            print(f"\n为 PatchTST 的前 16 个维度生成单独图...")
            plot_umap_by_dimensions(
                patchtst_emb,
                title_prefix='PatchTST',
                save_dir=patchtst_dim_dir,
                max_samples=30000,
                selected_dims=list(range(16))
            )
    else:
        print(f"\n⚠️  文件不存在: {patchtst_path}")
    
    # 读取 PCLE embeddings
    if os.path.exists(pcle_path):
        print("\n" + "-" * 80)
        print("加载 PCLE embeddings...")
        pcle_data = torch.load(pcle_path, map_location='cpu')
        
        if isinstance(pcle_data, dict):
            pcle_emb = pcle_data.get('embeddings', pcle_data)
        else:
            pcle_emb = pcle_data
        
        print(f"  形状: {pcle_emb.shape}")
        B, C, N, P = pcle_emb.shape
        
        # 方式1: 生成网格对比图
        print("\n生成 PCLE 网格对比图...")
        plot_umap_grid_comparison(
            pcle_emb,
            title_prefix='PCLE Embeddings',
            save_path=os.path.join(data_dir, 'umap_pcle_grid_dims.png'),
            max_samples=30000,
            num_dims=16
        )
        
        # 方式2: 单独图
        if P <= 32:
            print(f"\n为 PCLE 的所有 {P} 个维度生成单独图...")
            plot_umap_by_dimensions(
                pcle_emb,
                title_prefix='PCLE',
                save_dir=pcle_dim_dir,
                max_samples=30000,
                selected_dims=None
            )
        else:
            print(f"\n为 PCLE 的前 16 个维度生成单独图...")
            plot_umap_by_dimensions(
                pcle_emb,
                title_prefix='PCLE',
                save_dir=pcle_dim_dir,
                max_samples=30000,
                selected_dims=list(range(16))
            )
    else:
        print(f"\n⚠️  文件不存在: {pcle_path}")
    
    print("\n" + "=" * 80)
    print("✓ 所有维度语义分析完成！")
    print(f"输出目录: {data_dir}")
    print(f"  - 网格图: umap_*_grid_dims.png")
    print(f"  - 单独图: patchtst_dimensions/ 和 pcle_dimensions/")
    print("=" * 80)


if __name__ == '__main__':
    main()
