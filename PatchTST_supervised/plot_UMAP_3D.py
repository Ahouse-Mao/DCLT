"""
绘制 PatchTST 和 PCLE embeddings 的 3D UMAP 可视化
从 data_outputs 目录读取保存的 embeddings 进行对比可视化
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


def plot_umap_3d(embeddings, title='3D UMAP Visualization', save_path=None, 
                 labels=None, max_samples=10000):
    """
    使用 UMAP 对 embeddings 进行 3D 降维可视化
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title: str, 图表标题
        save_path: str, 保存路径（如果为None则显示图片）
        labels: np.ndarray, 用于着色的标签，shape [B*C*N]
        max_samples: int, 最大采样点数（UMAP计算很慢，建议限制样本数）
    
    Returns:
        embedding_umap: np.ndarray, UMAP降维后的3D坐标 [n_samples, 3]
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        print("请运行: pip install umap-learn")
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
        indices = np.random.choice(total_samples, max_samples, replace=False)
        embeddings_2d = embeddings_2d[indices]
        if labels is not None:
            labels = labels[indices]
        print(f"采样到 {max_samples}/{total_samples} 个点")
    else:
        indices = None
    
    # UMAP 降维到 3D
    print("正在进行 3D UMAP 降维...")
    reducer = UMAP(
        n_neighbors=30, 
        min_dist=0.4,
        n_components=3,  # 3D 降维
        random_state=42,
        verbose=True
    )
    
    embedding_umap = reducer.fit_transform(embeddings_2d)
    print(f"✓ UMAP 完成: {embedding_umap.shape}")
    
    # 创建 3D 图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None and len(labels) == embeddings_2d.shape[0]:
        # 使用提供的标签着色
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            embedding_umap[:, 2],
            c=labels, 
            cmap='tab10', 
            s=5, 
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Label', shrink=0.5, aspect=5)
    else:
        # 默认按通道（变量）着色
        if indices is not None:
            # 采样情况下重新生成标签
            var_labels = np.tile(np.repeat(np.arange(C), N), B)[:len(indices)]
            var_labels = var_labels[np.argsort(indices)]
        else:
            var_labels = np.tile(np.repeat(np.arange(C), N), B)
        
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            embedding_umap[:, 2],
            c=var_labels[:len(embedding_umap)], 
            cmap='tab10', 
            s=5, 
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Variable Index', shrink=0.5, aspect=5)
    
    ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图片已保存: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return embedding_umap


def plot_3d_side_by_side_comparison(patchtst_emb, pcle_emb, save_dir='./data_outputs', 
                                    max_samples=5000):
    """
    并排对比 PatchTST 和 PCLE 的 3D UMAP 可视化
    
    Args:
        patchtst_emb: torch.Tensor, PatchTST embeddings [B, C, N, P]
        pcle_emb: torch.Tensor, PCLE embeddings [B, C, N, P']
        save_dir: str, 保存目录
        max_samples: int, 每个模型最大采样点数
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        print("请运行: pip install umap-learn")
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
    if patchtst_2d.shape[0] > max_samples:
        indices = np.random.choice(patchtst_2d.shape[0], max_samples, replace=False)
        patchtst_2d = patchtst_2d[indices]
        print(f"PatchTST 采样: {max_samples}/{patchtst_emb.reshape(-1, P1).shape[0]}")
    
    if pcle_2d.shape[0] > max_samples:
        indices = np.random.choice(pcle_2d.shape[0], max_samples, replace=False)
        pcle_2d = pcle_2d[indices]
        print(f"PCLE 采样: {max_samples}/{pcle_emb.reshape(-1, P2).shape[0]}")
    
    # 创建标签（按变量着色）
    patchtst_labels = np.tile(np.repeat(np.arange(C1), N1), B1)[:len(patchtst_2d)]
    pcle_labels = np.tile(np.repeat(np.arange(C2), N2), B2)[:len(pcle_2d)]
    
    # UMAP 降维到 3D
    print("\n正在对 PatchTST embeddings 进行 3D UMAP 降维...")
    reducer1 = UMAP(n_neighbors=30, min_dist=0.4, n_components=3, random_state=42, verbose=True)
    patchtst_umap = reducer1.fit_transform(patchtst_2d)
    
    print("正在对 PCLE embeddings 进行 3D UMAP 降维...")
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
        c=patchtst_labels, 
        cmap='tab10', 
        s=5, 
        alpha=0.6
    )
    ax1.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax1.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax1.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax1.set_title(f'PatchTST Embeddings (dim={P1})', fontsize=16, fontweight='bold', pad=20)
    ax1.view_init(elev=20, azim=45)
    plt.colorbar(scatter1, ax=ax1, label='Variable Index', shrink=0.5, aspect=5)
    
    # PCLE
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(
        pcle_umap[:, 0], 
        pcle_umap[:, 1], 
        pcle_umap[:, 2],
        c=pcle_labels, 
        cmap='tab10', 
        s=5, 
        alpha=0.6
    )
    ax2.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
    ax2.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
    ax2.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
    ax2.set_title(f'PCLE Embeddings (dim={P2})', fontsize=16, fontweight='bold', pad=20)
    ax2.view_init(elev=20, azim=45)
    plt.colorbar(scatter2, ax=ax2, label='Variable Index', shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'umap_3d_comparison_patchtst_vs_pcle.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 3D对比图已保存: {save_path}")
    plt.close()


def save_interactive_3d_views(embedding_umap, labels, save_dir, prefix, title):
    """
    保存多个角度的 3D 视图
    
    Args:
        embedding_umap: np.ndarray, UMAP 3D坐标 [n_samples, 3]
        labels: np.ndarray, 标签
        save_dir: str, 保存目录
        prefix: str, 文件名前缀
        title: str, 图表标题
    """
    views = [
        (20, 45, 'view1'),    # 默认视角
        (20, 135, 'view2'),   # 旋转90度
        (20, 225, 'view3'),   # 旋转180度
        (20, 315, 'view4'),   # 旋转270度
        (60, 45, 'top'),      # 俯视
        (-20, 45, 'bottom'),  # 仰视
    ]
    
    for elev, azim, view_name in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embedding_umap[:, 0], 
            embedding_umap[:, 1], 
            embedding_umap[:, 2],
            c=labels, 
            cmap='tab10', 
            s=5, 
            alpha=0.6
        )
        
        ax.set_xlabel('UMAP 1', fontsize=12, labelpad=10)
        ax.set_ylabel('UMAP 2', fontsize=12, labelpad=10)
        ax.set_zlabel('UMAP 3', fontsize=12, labelpad=10)
        ax.set_title(f'{title} - {view_name}', fontsize=16, fontweight='bold', pad=20)
        ax.view_init(elev=elev, azim=azim)
        plt.colorbar(scatter, ax=ax, label='Variable Index', shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{prefix}_{view_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存视角 {view_name}: {save_path}")


def main():
    """
    主函数：读取 data_outputs 中的 embeddings 并绘制 3D UMAP
    """
    print("=" * 80)
    print("PatchTST vs PCLE Embeddings 3D UMAP 可视化")
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
    
    # 绘制单独的 3D UMAP 图
    print("\n" + "-" * 80)
    print("绘制单独的 3D UMAP 图...")
    
    # PatchTST
    print("\n处理 PatchTST embeddings...")
    patchtst_umap = plot_umap_3d(
        patchtst_emb,
        title='PatchTST Embeddings 3D UMAP',
        save_path=os.path.join(data_dir, 'umap_3d_patchtst.png'),
        max_samples=30000
    )
    
    # PCLE
    print("\n处理 PCLE embeddings...")
    pcle_umap = plot_umap_3d(
        pcle_emb,
        title='PCLE Embeddings 3D UMAP',
        save_path=os.path.join(data_dir, 'umap_3d_pcle.png'),
        max_samples=30000
    )
    
    # 绘制并排对比图
    print("\n" + "-" * 80)
    print("绘制 3D 并排对比图...")
    plot_3d_side_by_side_comparison(
        patchtst_emb,
        pcle_emb,
        save_dir=data_dir,
        max_samples=30000
    )
    
    # 保存多角度视图（可选）
    print("\n" + "-" * 80)
    print("保存多角度视图...")
    
    if patchtst_umap is not None:
        B, C, N, P = patchtst_emb.shape
        labels = np.tile(np.repeat(np.arange(C), N), B)[:len(patchtst_umap)]
        print("\nPatchTST 多角度视图:")
        save_interactive_3d_views(
            patchtst_umap, 
            labels, 
            data_dir, 
            'patchtst_3d',
            'PatchTST'
        )
    
    if pcle_umap is not None:
        B, C, N, P = pcle_emb.shape
        labels = np.tile(np.repeat(np.arange(C), N), B)[:len(pcle_umap)]
        print("\nPCLE 多角度视图:")
        save_interactive_3d_views(
            pcle_umap, 
            labels, 
            data_dir, 
            'pcle_3d',
            'PCLE'
        )
    
    print("\n" + "=" * 80)
    print("✓ 所有 3D UMAP 可视化完成！")
    print(f"输出目录: {data_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
