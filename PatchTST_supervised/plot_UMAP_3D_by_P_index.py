"""
绘制 PatchTST 和 PCLE embeddings 的 3D UMAP 可视化
按照 P 维度的索引位置进行着色（使用 argmax 方法）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def plot_umap_3d_by_p_position(embeddings, title='3D UMAP Visualization', save_path=None, 
                                max_samples=10000):
    """
    使用 3D UMAP 对 embeddings 进行降维，按 P 维度的主导索引着色
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title: str, 图表标题
        save_path: str, 保存路径
        max_samples: int, 最大采样点数
    
    Returns:
        embedding_umap: np.ndarray, UMAP降维后的3D坐标 [n_samples, 3]
        p_labels: np.ndarray, P索引标签
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        return None, None
    
    # 转换为 numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    B, C, N, P = embeddings.shape
    print(f"原始形状: B={B}, C={C}, N={N}, P={P}")
    
    # Reshape 为 [B*C*N, P]
    embeddings_2d = embeddings.reshape(-1, P)
    total_samples = embeddings_2d.shape[0]
    print(f"Reshape后: {embeddings_2d.shape}")
    
    # 为每个样本分配P索引标签（使用argmax：哪个维度的值最大）
    p_labels = np.argmax(embeddings_2d, axis=1)  # [B*C*N]
    print(f"P标签分布: {np.bincount(p_labels)}")
    
    # 随机采样
    if total_samples > max_samples:
        np.random.seed(42)
        indices = np.random.choice(total_samples, max_samples, replace=False)
        embeddings_sampled = embeddings_2d[indices]
        p_labels_sampled = p_labels[indices]
        print(f"采样到 {max_samples}/{total_samples} 个点")
    else:
        embeddings_sampled = embeddings_2d
        p_labels_sampled = p_labels
    
    # 3D UMAP 降维
    print("正在进行 3D UMAP 降维...")
    reducer = UMAP(
        n_neighbors=30, 
        min_dist=0.4,
        n_components=3,  # 3D
        random_state=42,
        verbose=True
    )
    
    embedding_umap = reducer.fit_transform(embeddings_sampled)
    print(f"✓ UMAP 完成: {embedding_umap.shape}")
    
    # 创建 3D 图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        embedding_umap[:, 0], 
        embedding_umap[:, 1], 
        embedding_umap[:, 2],
        c=p_labels_sampled, 
        cmap='tab20' if P <= 20 else 'viridis',
        s=5, 
        alpha=0.6
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Dominant P Index (argmax)', 
                        shrink=0.5, aspect=5, pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加说明
    info_text = f'Total dims: {P}\nColored by argmax(P)\nSamples: {len(p_labels_sampled)}'
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图片已保存: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return embedding_umap, p_labels_sampled


def save_multiple_views(embedding_umap, p_labels, P, save_dir, prefix, title_base):
    """
    保存多个角度的 3D 视图
    
    Args:
        embedding_umap: np.ndarray, UMAP 3D坐标 [n_samples, 3]
        p_labels: np.ndarray, P索引标签
        P: int, 总维度数
        save_dir: str, 保存目录
        prefix: str, 文件名前缀
        title_base: str, 标题基础文本
    """
    views = [
        (20, 45, 'view1'),    # 默认视角
        (20, 135, 'view2'),   # 旋转90度
        (20, 225, 'view3'),   # 旋转180度
        (20, 315, 'view4'),   # 旋转270度
        (60, 45, 'top'),      # 俯视
        (-20, 45, 'bottom'),  # 仰视
    ]
    
    print(f"\n保存 {len(views)} 个不同角度的视图...")
    
    for elev, azim, view_name in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            embedding_umap[:, 2],
            c=p_labels, 
            cmap='tab20' if P <= 20 else 'viridis',
            s=5, 
            alpha=0.6
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
        ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
        ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
        ax.set_title(f'{title_base} - {view_name}', fontsize=16, fontweight='bold', pad=20)
        ax.view_init(elev=elev, azim=azim)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Dominant P Index', 
                           shrink=0.5, aspect=5, pad=0.1)
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{prefix}_{view_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存视角 {view_name}: {save_path}")


def plot_3d_side_by_side(patchtst_emb, pcle_emb, save_dir='./data_outputs', 
                         max_samples=10000):
    """
    并排对比 PatchTST 和 PCLE 的 3D UMAP 可视化
    
    Args:
        patchtst_emb: torch.Tensor, PatchTST embeddings [B, C, N, P]
        pcle_emb: torch.Tensor, PCLE embeddings [B, C, N, P']
        save_dir: str, 保存目录
        max_samples: int, 最大采样点数
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
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
    
    # 计算 P 标签
    patchtst_p_labels = np.argmax(patchtst_2d, axis=1)
    pcle_p_labels = np.argmax(pcle_2d, axis=1)
    
    # 采样
    np.random.seed(42)
    if patchtst_2d.shape[0] > max_samples:
        indices1 = np.random.choice(patchtst_2d.shape[0], max_samples, replace=False)
        patchtst_2d = patchtst_2d[indices1]
        patchtst_p_labels = patchtst_p_labels[indices1]
        print(f"PatchTST 采样: {max_samples}/{B1*C1*N1}")
    
    if pcle_2d.shape[0] > max_samples:
        indices2 = np.random.choice(pcle_2d.shape[0], max_samples, replace=False)
        pcle_2d = pcle_2d[indices2]
        pcle_p_labels = pcle_p_labels[indices2]
        print(f"PCLE 采样: {max_samples}/{B2*C2*N2}")
    
    # 3D UMAP 降维
    print("\n正在对 PatchTST embeddings 进行 3D UMAP 降维...")
    reducer1 = UMAP(n_neighbors=30, min_dist=0.4, n_components=3, random_state=42, verbose=True)
    patchtst_umap = reducer1.fit_transform(patchtst_2d)
    
    print("\n正在对 PCLE embeddings 进行 3D UMAP 降维...")
    reducer2 = UMAP(n_neighbors=30, min_dist=0.4, n_components=3, random_state=42, verbose=True)
    pcle_umap = reducer2.fit_transform(pcle_2d)
    
    # 创建并排对比图
    fig = plt.figure(figsize=(24, 10))
    
    # PatchTST
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(
        patchtst_umap[:, 0], 
        patchtst_umap[:, 1], 
        patchtst_umap[:, 2],
        c=patchtst_p_labels, 
        cmap='tab20' if P1 <= 20 else 'viridis',
        s=5, 
        alpha=0.6
    )
    ax1.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax1.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax1.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax1.set_title(f'PatchTST (dim={P1})', fontsize=16, fontweight='bold', pad=20)
    ax1.view_init(elev=20, azim=45)
    plt.colorbar(scatter1, ax=ax1, label='Dominant P Index', shrink=0.5, aspect=5, pad=0.1)
    
    # PCLE
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(
        pcle_umap[:, 0], 
        pcle_umap[:, 1], 
        pcle_umap[:, 2],
        c=pcle_p_labels, 
        cmap='tab20' if P2 <= 20 else 'viridis',
        s=5, 
        alpha=0.6
    )
    ax2.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax2.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax2.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax2.set_title(f'PCLE (dim={P2})', fontsize=16, fontweight='bold', pad=20)
    ax2.view_init(elev=20, azim=45)
    plt.colorbar(scatter2, ax=ax2, label='Dominant P Index', shrink=0.5, aspect=5, pad=0.1)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'umap_3d_by_p_index_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 3D对比图已保存: {save_path}")
    plt.close()


def main():
    """
    主函数：读取 embeddings 并绘制 3D UMAP（按 P 索引着色）
    """
    print("=" * 80)
    print("Embeddings 按 P 维度索引着色 - 3D UMAP 可视化")
    print("=" * 80)
    
    # 设置路径
    data_dir = './data_outputs'
    patchtst_path = os.path.join(data_dir, 'patchtst_embeddings_02.pt')
    pcle_path = os.path.join(data_dir, 'pcle_embeddings_02.pt')
    
    # 创建子目录
    views_dir = os.path.join(data_dir, 'p_index_3d_views')
    os.makedirs(views_dir, exist_ok=True)
    
    patchtst_umap = None
    patchtst_labels = None
    patchtst_P = None
    pcle_umap = None
    pcle_labels = None
    pcle_P = None
    
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
        patchtst_P = patchtst_emb.shape[3]
        
        # 生成 3D UMAP
        print("\n生成 PatchTST 3D UMAP...")
        patchtst_umap, patchtst_labels = plot_umap_3d_by_p_position(
            patchtst_emb,
            title='PatchTST - 3D UMAP by Dominant P Index',
            save_path=os.path.join(data_dir, 'umap_3d_patchtst_by_p_index.png'),
            max_samples=30000
        )
        
        # 保存多角度视图
        if patchtst_umap is not None:
            save_multiple_views(
                patchtst_umap,
                patchtst_labels,
                patchtst_P,
                views_dir,
                'patchtst_3d_p_index',
                'PatchTST'
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
        pcle_P = pcle_emb.shape[3]
        
        # 生成 3D UMAP
        print("\n生成 PCLE 3D UMAP...")
        pcle_umap, pcle_labels = plot_umap_3d_by_p_position(
            pcle_emb,
            title='PCLE - 3D UMAP by Dominant P Index',
            save_path=os.path.join(data_dir, 'umap_3d_pcle_by_p_index.png'),
            max_samples=30000
        )
        
        # 保存多角度视图
        if pcle_umap is not None:
            save_multiple_views(
                pcle_umap,
                pcle_labels,
                pcle_P,
                views_dir,
                'pcle_3d_p_index',
                'PCLE'
            )
    else:
        print(f"\n⚠️  文件不存在: {pcle_path}")
    
    # 并排对比图
    if os.path.exists(patchtst_path) and os.path.exists(pcle_path):
        print("\n" + "-" * 80)
        print("生成 3D 并排对比图...")
        patchtst_data = torch.load(patchtst_path, map_location='cpu')
        pcle_data = torch.load(pcle_path, map_location='cpu')
        
        if isinstance(patchtst_data, dict):
            patchtst_emb = patchtst_data.get('embeddings', patchtst_data)
        else:
            patchtst_emb = patchtst_data
            
        if isinstance(pcle_data, dict):
            pcle_emb = pcle_data.get('embeddings', pcle_data)
        else:
            pcle_emb = pcle_data
        
        plot_3d_side_by_side(
            patchtst_emb,
            pcle_emb,
            save_dir=data_dir,
            max_samples=30000
        )
    
    print("\n" + "=" * 80)
    print("✓ 所有 3D UMAP 可视化完成！")
    print(f"输出目录: {data_dir}")
    print("  主图:")
    print("    - umap_3d_patchtst_by_p_index.png")
    print("    - umap_3d_pcle_by_p_index.png")
    print("    - umap_3d_by_p_index_comparison.png (并排对比)")
    print(f"  多角度视图: {views_dir}/")
    print("\n说明：")
    print("  每个样本按其在P维度上的最大值位置（argmax）进行着色")
    print("  3D可视化提供更丰富的空间信息")
    print("=" * 80)


if __name__ == '__main__':
    main()
