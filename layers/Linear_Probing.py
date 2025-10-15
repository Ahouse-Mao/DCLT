import time
from typing import Iterable, List, Tuple, Dict, Optional
import os
import sys
import numpy as np
import torch

# 确保项目根目录在 sys.path 中，便于从子目录直接运行本文件
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)
	
from layers.linear_probe_metrics import calc_forecast_metrics
from layers.Eval_Model import fit_ridge
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data_provider.DCLT_data_loader_v3 import DCLT_data_loader_v3
from cl_models.DCLT_patchtst_pretrained_cl import LitModel as LitModelPatchTST
from cl_models.DCLT_pretrained_cl_v3_1 import LitModel as LitModelV3_1


@torch.no_grad()
def _extract_features_and_labels(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	device: Optional[torch.device] = None,
	feature_mode: str = "flatten",  # 'flatten' | 'mean' | 'max'
	extract_max_batches: Optional[int] = None,
	extract_sample_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	从 (x, y) DataLoader 中抽取特征与标签，用于线性预测。

	假设：
	- dataloader 返回 (x, y)，x:(B, C, L), y:(B, C, pred_len)
	- model(x) -> (B, C, N, D) 的 token 表示
	- 特征构造：
		- 'flatten': 将 token 维与通道维展平成 (B, C, N*D) -> (B*C, N*D)
		- 'mean'/'max': 在 token 维上做池化得到 (B, C, D) -> (B*C, D)

	返回：
	- X: (num_samples, D) = 所有 batch 聚合后的特征
	- Y: (num_samples, pred_len) = 对应的未来标签（标准化空间）
	- var_idx: (num_samples,) = 每条样本对应的变量列索引 [0..C-1]
	"""
	model.eval()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	feat_list: List[np.ndarray] = []
	label_list: List[np.ndarray] = []
	varid_list: List[np.ndarray] = []

	# 若设置比例抽样，确定最多处理的批次数
	max_batches = None
	if extract_sample_ratio is not None:
		try:
			total_batches = len(dataloader)
			if total_batches > 0:
				max_batches = max(1, int(total_batches * float(extract_sample_ratio)))
		except Exception:
			pass
	if extract_max_batches is not None:
		max_batches = extract_max_batches if max_batches is None else min(max_batches, extract_max_batches)

	for b_idx, (x, y) in enumerate(dataloader):
		if max_batches is not None and b_idx >= max_batches:
			break
		# x:(B,C,L), y:(B,C,pred_len)
		B, C, _ = x.shape
		_, _, pred_len = y.shape
		x = x.to(device)

		z = model(x)  # (B, C, N, D)
		if z.dim() != 4:
			raise ValueError(f"模型 forward 期望返回 (B, C, N, D)，却得到 {tuple(z.shape)}")
		# 特征构造
		if feature_mode == "flatten":
			f = z.reshape(B, C, -1)   # (B, C, N*D)
		elif feature_mode == "mean":
			f = z.mean(dim=2)         # (B, C, D)
		elif feature_mode == "max":
			f, _ = z.max(dim=2)       # (B, C, D)
		else:
			raise ValueError("feature_mode 仅支持 'flatten' | 'mean' | 'max'")

		f = f.reshape(B * C, -1)  # (B*C, N*D or D)
		y_np = y.permute(0, 1, 2).contiguous().view(B * C, pred_len).cpu().numpy()

		feat_list.append(f.detach().cpu().numpy())
		label_list.append(y_np)
		# 记录变量索引
		var_ids = np.tile(np.arange(C, dtype=np.int64), B)
		varid_list.append(var_ids)

	X = np.concatenate(feat_list, axis=0)
	Y = np.concatenate(label_list, axis=0)
	var_idx = np.concatenate(varid_list, axis=0)
	return X, Y, var_idx


def _seq_split_indices(n: int, splits: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""按照时间顺序划分索引为 train/valid/test。"""
	a, b, c = splits
	assert abs(a + b + c - 1.0) < 1e-6, "splits 必须相加为 1.0"
	n_train = int(n * a)
	n_valid = int(n * b)
	n_test = n - n_train - n_valid
	idx = np.arange(n)
	return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def _inverse_transform_by_var(
	arr_norm: np.ndarray,  # (N, pred_len)
	var_idx: np.ndarray,   # (N,)
	mean: np.ndarray,      # (C,)
	scale: np.ndarray,     # (C,)
) -> np.ndarray:
	"""按样本所属变量列进行逐列反标准化。"""
	mean_sel = mean[var_idx][:, None]
	scale_sel = scale[var_idx][:, None]
	return arr_norm * scale_sel + mean_sel


def eval_forecasting(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	scaler,
	pred_lens: Iterable[int],
	pool: str = "flatten",
	method: str = "ridge",
	device: Optional[torch.device] = None,
	split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
	extract_max_batches: Optional[int] = None,
	extract_sample_ratio: Optional[float] = None,
	ridge_max_samples: int = 100000,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Dict[int, float]]]:
	"""
	参考 TS2Vec 的 eval_forecasting，将线性探测类改为一个评估方法：
	- 使用当前预训练模型的 forward(x)->(B,C,N,D) 抽取特征
	- 特征构造：默认展平 token 与通道维，得到 (B*C, N*D)
	- 用线性回归（Ridge）从特征预测未来 pred_len 个步长
	- 返回预测日志与评估指标（标准化空间与反标准化后的原始空间）

	参数：
	  model: 预训练模型（其 forward 符合上述契约）
	  dataloader: 由 DCLT_data_loader_v3 构建，返回 (x,y)
	  scaler: dataset.scaler（StandardScaler）用于反标准化
	  pred_lens: 需要评估的多个预测步长（需 <= dataset.pred_len）
	  pool: 特征模式 'flatten' | 'mean' | 'max'（默认 flatten 展平 N*D）
	  method: 线性模型方法（当前实现 ridge）
	  device: 计算设备
	  split: 按时间顺序的训练/验证/测试划分比例

	返回：
	  out_log: {pred_len: {norm_pred, raw_pred, norm_gt, raw_gt}}
	  eval_res: { 'lr_train_time': {pred_len: t}, 'lr_infer_time': {pred_len: t}, 'emb_extract_time': seconds,
				  'metrics_norm': {pred_len: (mae,mse,rmse,mape,mspe,rse,corr)},
				  'metrics_raw':  {pred_len: (mae,mse,rmse,mape,mspe,rse,corr)} }
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 1) 抽取所有样本的特征与标签（标准化空间）
	t0 = time.time()
	# 兼容旧参数名：pool 用于传递特征模式
	feature_mode = pool
	X_all, Y_all, var_idx_all = _extract_features_and_labels(
		model, dataloader, device=device, feature_mode=feature_mode,
		extract_max_batches=extract_max_batches, extract_sample_ratio=extract_sample_ratio,
	)
	emb_extract_time = time.time() - t0

	n_samples, out_len_ds = Y_all.shape
	pred_lens = list(pred_lens)
	for L in pred_lens:
		if L > out_len_ds:
			raise ValueError(f"pred_len={L} 超过数据集标签长度 {out_len_ds}，请在 cfg.model.pred_len >= {L}")

	# 2) 顺序划分 Train/Valid/Test
	idx_tr, idx_va, idx_te = _seq_split_indices(n_samples, split)

	out_log: Dict[int, Dict[str, np.ndarray]] = {}
	lr_train_time: Dict[int, float] = {}
	lr_infer_time: Dict[int, float] = {}
	metrics_norm: Dict[int, Tuple[float, ...]] = {}
	metrics_raw: Dict[int, Tuple[float, ...]] = {}

	mean = getattr(scaler, 'mean_', None)
	scale = getattr(scaler, 'scale_', None)
	if mean is None or scale is None:
		raise ValueError("scaler 必须为 StandardScaler 并已 fit（需含 mean_ 与 scale_ 属性）")

	for L in pred_lens:
		X_tr, Y_tr = X_all[idx_tr], Y_all[idx_tr, :L]
		X_va, Y_va = X_all[idx_va], Y_all[idx_va, :L]
		X_te, Y_te = X_all[idx_te], Y_all[idx_te, :L]
		var_te = var_idx_all[idx_te]

		# 3) 训练线性模型（Ridge）
		if method.lower() != "ridge":
			raise NotImplementedError("当前仅实现 ridge 方法，其他方法可按需扩展")
		t1 = time.time()
		lr = fit_ridge(X_tr, Y_tr, X_va, Y_va, MAX_SAMPLES=ridge_max_samples)
		lr_train_time[L] = time.time() - t1

		# 4) 测试集推理
		t2 = time.time()
		Y_pred_te = lr.predict(X_te)  # (Nte, L)
		lr_infer_time[L] = time.time() - t2

		# 5) 反标准化
		Y_pred_te_raw = _inverse_transform_by_var(Y_pred_te, var_te, mean, scale)
		Y_te_raw = _inverse_transform_by_var(Y_te, var_te, mean, scale)

		# 6) 计算指标（展平后整体评估）
		metrics_norm[L] = calc_forecast_metrics(Y_pred_te, Y_te)
		metrics_raw[L] = calc_forecast_metrics(Y_pred_te_raw, Y_te_raw)

		out_log[L] = {
			'norm_pred': Y_pred_te,
			'norm_gt': Y_te,
			'raw_pred': Y_pred_te_raw,
			'raw_gt': Y_te_raw,
		}

	eval_res = {
		'emb_extract_time': emb_extract_time,
		'lr_train_time': lr_train_time,
		'lr_infer_time': lr_infer_time,
		'metrics_norm': metrics_norm,
		'metrics_raw': metrics_raw,
	}

	return out_log, eval_res


if __name__ == "__main__":
	# 简单测试：按照主训练代码中的线性探测流程跑一次评估
	PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	if PROJECT_ROOT not in sys.path:
		sys.path.insert(0, PROJECT_ROOT)

	# 根据 cfg 选择模型（与主脚本 select_cl_model 一致的逻辑）
	def _select_cl_model(cfg):
		if cfg.model_name == 'DCLT_pretrained_cl_v3':
			return LitModelV3_1(cfg)
		else:
			raise ValueError(f"Unknown model: {cfg.model_name}")

	# 读取配置
	cfg_path = os.path.join(PROJECT_ROOT, 'cl_conf', 'pretrain_cfg_v3.yaml')
	if not os.path.exists(cfg_path):
		print(f"找不到配置文件: {cfg_path}")
		sys.exit(1)
	cfg = OmegaConf.load(cfg_path)

	# 构建数据与模型
	dataset = DCLT_data_loader_v3(cfg=cfg)
	loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)
	model = _select_cl_model(cfg)

	# 预测步长与标准化器
	pred_lens = [getattr(dataset, 'pred_len', None) or (cfg.model.seq_len // 2)]
	scaler = dataset.scaler

	# 评估
	print("[Linear_Probing Test] 开始评估...")
	out_log, eval_res = eval_forecasting(
		model, loader, scaler,
		pred_lens=pred_lens,
		pool="flatten",           # 展平特征 (N*D)
		method="ridge",
		extract_sample_ratio=0.2,  # 仅用 20% 的 batch 做 probing 以降低耗时
		ridge_max_samples=50000    # Ridge 内部最大样本裁剪
	)

	for L, vals in eval_res['metrics_raw'].items():
		mae, mse, rmse, mape, mspe, rse, corr = vals
		print(f"[Linear_Probing Test] pred_len={L} | raw: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, MSPE={mspe:.4f}, RSE={rse:.4f}, CORR={corr:.4f}")
	for L, vals in eval_res['metrics_norm'].items():
		mae, mse, rmse, mape, mspe, rse, corr = vals
		print(f"[Linear_Probing Test] pred_len={L} | norm: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, MSPE={mspe:.4f}, RSE={rse:.4f}, CORR={corr:.4f}")

	print("[Linear_Probing Test] 结束。")

