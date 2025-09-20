import os
import datetime
import logging
import torch
import pytorch_lightning as L

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from data_provider.DCLT_data_loader_v2 import GraphContrastDataset
from data_provider.DCLT_pred_data_loader import DCLT_pred_dataset
from torch.utils.data import DataLoader
from utils.utils import print_cfg, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping  # callback version
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary as ModelSummaryUtil  # utility for manual string summary


torch.set_printoptions(sci_mode=False)

# 模块级 logger
logger = logging.getLogger(__name__)


def _setup_logging(log_dir: str, time_tag: str):
    """初始化当前模块(logger = logging.getLogger(__name__))的日志输出。

    参数:
            log_dir   : 日志文件保存目录（由调用方决定根路径）。
            time_tag  : 一次运行的时间戳，用于区分多个实验/运行实例。

    设计原则:
        1. 只在该模块第一次调用时添加 handler，防止重复添加导致日志成倍输出。
        2. 同时输出到控制台(便于实时观察) 与 文件(便于持久化/排查历史)。
        3. 使用统一格式，包含时间、级别、模块名、消息，方便筛选。
        4. 关闭向父 logger 继续冒泡 (propagate=False)，避免根 logger 再次打印重复内容。

    可扩展点:
        - 想添加 JSON 日志，可新增一个自定义 Formatter。
        - 想分级别写不同文件，可再建 WARNING 以上的 FileHandler。
        - 想在多进程/分布式环境中区分进程，可在格式里加入 %(process)d / %(rank)s。
    """
    if not logger.handlers:  # 确保只配置一次（否则多次 import / hydra 多阶段初始化会重复）
        logger.setLevel(logging.INFO)  # 这里设置模块级别；全局细化可在入口再覆盖

        # 日志格式说明：
        # %(asctime)s  时间戳   %(levelname)s 级别
        # %(name)s     logger 名（= 模块路径）
        # %(message)s  实际日志内容
        fmt = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s',
                                                        datefmt='%Y-%m-%d %H:%M:%S')

        # ============== 控制台 Handler ==============
        # 实时输出到标准输出，便于前台/终端直接观察训练进度
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)     # 若想调试详细信息改成 DEBUG
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # ============== 文件 Handler ==============
        # 每次运行生成独立日志文件: train_<time_tag>.log
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'train_{time_tag}.log')
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # 关闭向上冒泡，避免根 logger 再打印一遍（常见重复输出问题）
        logger.propagate = False


def select_cl_model(cfg: DictConfig, T):
    if cfg.model_name == 'DCLT_patchtst_pretrained_cl':
        from cl_models.DCLT_patchtst_pretrained_cl import LitModel
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")

    return LitModel(cfg, T)

def init_path(cfg: DictConfig):
    root_path = os.getcwd()
    cfg.dataset.path = os.path.join(root_path, "dataset", f"{cfg.dataset.name}.csv")
    cfg.dataset.dtw_path = os.path.join(root_path, "DTW_matrix", f"{cfg.dataset.name}.csv")
    return cfg 

@hydra.main(version_base=None, config_path="cl_conf", config_name="pretrain_cfg")
def main(cfg: DictConfig) -> None:
    # =============================
    # 0. 基础环境与配置打印
    # =============================
    # 先初始化日志（否则 INFO 级别不会显示）
    time_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_root = get_original_cwd()
    log_root = getattr(cfg.light_model.trainer, 'log_root', 'logs')
    # 日志分区到数据集：logs/<dataset_name>/...
    dataset_name = getattr(cfg.dataset, 'name', 'default')
    dataset_log_base = os.path.join(project_root, log_root, dataset_name)
    _setup_logging(dataset_log_base, time_tag)

    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Using GPU ID: {cfg.gpu_id}")
    print_cfg(cfg, logger)         # 仍沿用原打印函数（内部可再改 logger）
    seed_everything(cfg.seed)      # 统一随机种子，增强复现性

    cfg = init_path(cfg)

    # =============================
    # 1. 构建数据 (仅训练集; 暂不使用验证集)
    # =============================
    dataset = GraphContrastDataset(
        data_path=cfg.dataset.path,
        dtw_matrix=cfg.dataset.dtw_path,
        k=cfg.dataset.k,
        P=cfg.dataset.P,
        N=cfg.dataset.N,
        sigma_method=cfg.dataset.sigma_method,
        self_tuning_k=cfg.dataset.self_tuning_k,
        use_mutual=cfg.dataset.use_mutual,
        neg_sampling=cfg.dataset.neg_sampling
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.light_model.trainer.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    T = dataset.data_length
    model = select_cl_model(cfg, T)

    # ===== 将模型结构摘要写入日志文件 =====
    try:
        summary_txt = str(ModelSummaryUtil(model, max_depth=1))
        logger.info("\nMODEL SUMMARY (logged):\n" + summary_txt)
    except Exception as e:
        logger.warning(f"记录模型结构摘要失败: {e}")

    # =============================
    # 2. 准备 checkpoints 与日志路径
    # =============================
    # time_tag / project_root / log_base 已在开头生成，可复用
    ckpt_root = getattr(cfg.light_model.trainer, 'ckpt_root', 'checkpoints')
    # checkpoints/<dataset_name>/pretrain_<time_tag>
    dataset_ckpt_base = os.path.join(project_root, ckpt_root, dataset_name)
    os.makedirs(dataset_ckpt_base, exist_ok=True)
    ckpt_dir = os.path.join(dataset_ckpt_base, f"pretrain_{time_tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # TensorBoardLogger（tb_logger） / CSVLogger（csv_logger）：用来自动收集 训练过程指标，方便 可视化和分析。
    # TensorBoardLogger存储二进制文件，而CSVLogger存储纯文本csv文件，二者可任选其一或同时使用。
    tb_logger = TensorBoardLogger(save_dir=dataset_log_base, name='tb', version=time_tag, default_hp_metric=False)
    csv_logger = CSVLogger(save_dir=dataset_log_base, name='csv', version=time_tag)

    # =============================
    # 3. 定义回调 (仅保存最优模型; 暂不启用 EarlyStopping)
    # =============================
    monitor_metric = getattr(cfg.light_model.trainer, 'monitor', None)  # 若无验证集，默认监控训练损失
    if monitor_metric is None:
        monitor_metric = 'train_loss'
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="model-{epoch:02d}",
        save_top_k=1 if monitor_metric else -1,  # 若无监控指标则保存最后一次 (-1 表示全部; 这里保持 1 只保留最新)
        monitor=monitor_metric,
        mode=getattr(cfg.light_model.trainer, 'monitor_mode', 'min') if monitor_metric else 'min'
    )
    summary_cb = ModelSummary(max_depth=1)

    # =============================
    # 3.1 EarlyStopping (可选)
    # =============================
    early_stop_cfg = cfg.light_model.early_stop
    early_stop_cb = None
    if early_stop_cfg.enable:
        # 监控指标：使用配置里的 monitor；默认仍然可使用上面的 train_loss
        es_monitor = early_stop_cfg.monitor
        early_stop_cb = EarlyStopping(
            monitor=es_monitor,
            mode=early_stop_cfg.mode,
            patience=early_stop_cfg.patience,
            min_delta=early_stop_cfg.min_delta,
            check_on_train_epoch_end=getattr(early_stop_cfg, 'check_on_train_epoch_end', True),
            verbose=True
        )
        logger.info(f"Enable EarlyStopping on '{es_monitor}' (mode={early_stop_cfg.mode}, patience={early_stop_cfg.patience})")
    else:
        logger.info("EarlyStopping disabled")

    # =============================
    # 4. 构建 Trainer （仅训练，无验证）
    # =============================
    max_epochs = cfg.light_model.trainer.max_epochs
    callbacks = [checkpoint_cb, summary_cb]
    if early_stop_cb:
        callbacks.append(early_stop_cb)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=cfg.devices,
        benchmark=cfg.benchmark,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger]
    )

    # =============================
    # 5. 启动训练
    # =============================
    trainer.fit(model, train_dataloaders=train_loader)  # 不传 val_dataloaders -> 纯训练

    # =============================
    # 6. 保存模型权重 (LightningModule state_dict)
    # =============================
    final_ckpt_path = os.path.join(ckpt_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_ckpt_path)
    logger.info(f"Saved final weights to: {final_ckpt_path}")

    # 可选：同时保存完整模型（含结构）
    full_model_path = os.path.join(ckpt_dir, 'full_model.pth')
    torch.save(model, full_model_path)
    logger.info(f"Saved full model to: {full_model_path}")

    # =============================
    # 7. 输出 checkpoint 信息
    # =============================
    if checkpoint_cb.best_model_path:
        logger.info(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    else:
        logger.info("No monitored metric; only final weights saved.")

    logger.info(f"TensorBoard logs dir: {tb_logger.log_dir}")
    logger.info(f"CSV logs dir: {csv_logger.log_dir}")

    # # =============================
    # # 8. 调整为推理模式
    # # =============================
    # dataset_pred = DCLT_pred_dataset(data_path=cfg.dataset.path)
    # dataloader_pred = DataLoader(
    #     dataset_pred,
    #     batch_size=cfg.light_model.trainer.batch_size,
    #     shuffle=False,
    # )
    # model.eval()
    # model.freeze()
    # x_list = trainer.predict(model, dataloaders=dataloader_pred)
    # x = torch.cat(x_list, dim=0)
    # x = x.squeeze(1)  # (num_vars, final_out_dim)
    # print(x)

if __name__ == "__main__":
    # 允许直接 python DCLT_main_pretrain.py 运行；Hydra 会接管参数与工作目录
    main()


