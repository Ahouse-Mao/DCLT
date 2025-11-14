"""
绘制 PatchTST 和 PCLE embeddings 的 UMAP 可视化
按照 P 维度的索引位置进行着色（将每个样本按其在P维度上的位置分配颜色）
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


def plot_umap_by_p_index(embeddings, title='UMAP Visualization', save_path=None, 
                         max_samples=10000, p_indices=None):
    """
    使用 UMAP 对 embeddings 进行降维，按 P 维度的索引位置着色
    
    说明：
    对于形状 [B, C, N, P] 的 embeddings，可以看作有 P 个不同的"特征子空间"
    本函数将每个样本按其对应的 P 索引位置进行着色，看不同位置是否有不同分布
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title: str, 图表标题
        save_path: str, 保存路径（如果为None则显示图片）
        max_samples: int, 最大采样点数
        p_indices: list, 要可视化的P索引列表，如果为None则取所有
    
    Returns:
        embedding_umap: np.ndarray, UMAP降维后的2D坐标 [n_samples, 2]
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
    
    # 选择要可视化的P索引
    if p_indices is None:
        p_indices = list(range(P))
    else:
        p_indices = [p for p in p_indices if 0 <= p < P]
    
    print(f"将可视化 {len(p_indices)} 个P索引: {p_indices[:10]}{'...' if len(p_indices) > 10 else ''}")
    
    # 只保留选定的P索引
    embeddings_selected = embeddings[:, :, :, p_indices]  # [B, C, N, len(p_indices)]
    
    # Reshape: 将 [B, C, N, len(p_indices)] -> [B*C*N*len(p_indices), 1]
    # 但实际上我们需要的是每个位置对应的embedding，这里需要重新思考...
    # 
    # 实际上应该是: [B, C, N, P] -> 每个 (b, c, n, p) 位置的数据
    # 重新组织为: 取出所有 P 维度的数据，为每个位置分配 P 索引标签
    
    # 正确的做法：
    # 1. Reshape 为 [B*C*N, P]，这样每一行是一个样本的所有P维特征
    # 2. 但我们想按 P 的索引着色，这意味着什么？
    #    - 如果只选择某些P索引，相当于只看这些特征维度
    #    - 然后按这些索引值着色
    
    # 重新理解需求：应该是要为每个(b,c,n)样本的P个特征点分别着色
    # 即：将 [B, C, N, P] 看作 B*C*N*P 个点，每个点按其P索引着色
    
    # Reshape 为 [B*C*N*P, 1]，每个值是一个标量
    embeddings_flat = embeddings.reshape(-1, 1)  # [B*C*N*P, 1]
    total_samples = embeddings_flat.shape[0]
    
    # 生成 P 索引标签：每个样本有 P 个特征，按 P 索引重复
    p_labels = np.tile(np.arange(P), B * C * N)  # [0,1,2,...,P-1, 0,1,2,...,P-1, ...]
    
    print(f"展平后形状: {embeddings_flat.shape}, 标签数量: {len(p_labels)}")
    
    # 随机采样（如果样本太多）
    if total_samples > max_samples:
        np.random.seed(42)
        indices = np.random.choice(total_samples, max_samples, replace=False)
        embeddings_sampled = embeddings_flat[indices]
        p_labels_sampled = p_labels[indices]
        print(f"采样到 {max_samples}/{total_samples} 个点")
    else:
        embeddings_sampled = embeddings_flat
        p_labels_sampled = p_labels
        indices = None
    
    print(f"\n⚠️  注意：这种可视化方式将每个样本的 P 个特征点视为独立的点")
    print(f"   每个点只有1个维度，可能不适合UMAP降维")
    print(f"   建议使用 plot_umap_by_p_position() 函数代替\n")
    
    # UMAP 降维（注意：1维数据降维效果可能不好）
    print("正在进行 UMAP 降维...")
    reducer = UMAP(
        n_neighbors=min(30, len(embeddings_sampled) - 1), 
        min_dist=0.4,
        n_components=2, 
        random_state=42,
        verbose=True
    )
    
    embedding_umap = reducer.fit_transform(embeddings_sampled)
    print(f"✓ UMAP 完成: {embedding_umap.shape}")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        embedding_umap[:, 0], 
        embedding_umap[:, 1], 
        c=p_labels_sampled, 
        cmap='tab20' if P <= 20 else 'viridis',  # P少用离散色，P多用连续色
        s=5, 
        alpha=0.6
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='P Index')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加说明文本
    info_text = f'Total P dimensions: {P}\nColored by P index'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
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
    
    return embedding_umap


def plot_umap_by_p_position(embeddings, title='UMAP Visualization', save_path=None, 
                            max_samples=10000):
    """
    正确的实现：对 [B, C, N, P] 进行降维，但按 P 维度的"分组"着色
    
    将 embeddings reshape 为 [B*C*N, P]，然后为每个样本分配一个P索引标签
    这个标签可以是：
    - 样本的主要P索引（argmax）
    - 样本所属的P分组
    
    Args:
        embeddings: torch.Tensor or np.ndarray, shape [B, C, N, P]
        title: str, 图表标题
        save_path: str, 保存路径
        max_samples: int, 最大采样点数
    """
    if not UMAP_AVAILABLE:
        print("❌ 未安装 umap-learn 库")
        return None
    
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
    
    # UMAP 降维
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
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        embedding_umap[:, 0], 
        embedding_umap[:, 1], 
        c=p_labels_sampled, 
        cmap='tab20' if P <= 20 else 'viridis',
        s=5, 
        alpha=0.6
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Dominant P Index (argmax)')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加说明
    info_text = f'Total dims: {P}\nColored by argmax(P)'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
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
    
    return embedding_umap


def main():
    """
    主函数：读取 embeddings 并按 P 索引着色
    """
    print("=" * 80)
    print("Embeddings 按 P 维度索引着色 - UMAP 可视化")
    print("=" * 80)
    
    # 设置路径
    data_dir = './data_outputs'
    patchtst_path = os.path.join(data_dir, 'patchtst_embeddings_02.pt')
    pcle_path = os.path.join(data_dir, 'pcle_embeddings_02.pt')
    
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
        
        # 使用 argmax 方法（推荐）
        print("\n使用 argmax 方法：每个样本按其最大值所在的P索引着色")
        plot_umap_by_p_position(
            patchtst_emb,
            title='PatchTST Embeddings - Colored by Dominant P Index',
            save_path=os.path.join(data_dir, 'umap_patchtst_by_p_index.png'),
            max_samples=30000
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
        
        # 使用 argmax 方法
        print("\n使用 argmax 方法：每个样本按其最大值所在的P索引着色")
        plot_umap_by_p_position(
            pcle_emb,
            title='PCLE Embeddings - Colored by Dominant P Index',
            save_path=os.path.join(data_dir, 'umap_pcle_by_p_index.png'),
            max_samples=30000
        )
    else:
        print(f"\n⚠️  文件不存在: {pcle_path}")
    
    print("\n" + "=" * 80)
    print("✓ 按 P 索引着色的 UMAP 可视化完成！")
    print(f"输出目录: {data_dir}")
    print("  - umap_patchtst_by_p_index.png")
    print("  - umap_pcle_by_p_index.png")
    print("\n说明：")
    print("  每个样本按其在P维度上的最大值位置（argmax）进行着色")
    print("  如果不同颜色聚类分明，说明不同P维度捕获了不同的模式")
    print("=" * 80)


if __name__ == '__main__':
    main()
