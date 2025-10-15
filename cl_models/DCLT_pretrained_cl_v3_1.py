import pytorch_lightning as L
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rotary_embedding_torch as rotary
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中，以便 'layers' 等顶层模块可被导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.Mypydebug import show_shape
show_shape()

from layers.RevIN import RevIN
from omegaconf import OmegaConf, DictConfig
from layers.CL_layers_v3 import LearnableDecompose, Encoder, FusionModule, TokenProjectionHead, TokenDecoder
from layers.CL_layers_v3 import spectral_loss_from_raw, token_ortho_loss

from layers.Patch_Soft_CL_V3 import Patch_Soft_CL
from layers.DTW_Sim_Cal_Load_v3 import DTW_Sim

from loss.infonce_loss import InfoNCE

from data_provider.DCLT_data_loader_v3 import DCLT_data_loader_v3
from torch.utils.data import DataLoader

EPS = 1e-12
NEG_INF = -1e9


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # 训练参数
        self.lr = cfg.train.lr

        # soft_cl_ctrl参数
        self.cl_shift_type = cfg.model.soft_cl_ctrl.shift_type # 'shift_trend', 'shift_season', 'shift_both'
        self.cl_weight = cfg.model.soft_cl_ctrl.cl_weight
        self.use_instance_cl = cfg.model.soft_cl_ctrl.use_instance_cl
        self.use_temporal_cl = cfg.model.soft_cl_ctrl.use_temporal_cl
        self.weight_mode = cfg.model.soft_cl_ctrl.weight_mode # patch_sim(仅使用patch_sim, 没有的部分用dist_sim补足) | dist_sim | patch_and_dist_sim(混合, 使用不同权重)

        # soft_cl_params参数
        self.temperature = cfg.model.soft_cl_params.temperature
        self.patch_sim_weight = cfg.model.soft_cl_params.patch_sim_weight
        self.tau_temporal = cfg.model.soft_cl_params.tau_temporal

        # other_loss参数
        self.use_other_loss = cfg.model.other_loss.use_other_loss
        self.spectral_weight = cfg.model.other_loss.spectral_weight # 频域损失权重
        self.ortho_weight = cfg.model.other_loss.ortho_weight # 正交损失权重
        self.rec_weight = cfg.model.other_loss.rec_weight # 重构损失权重

        self.use_instance_cl = cfg.model.soft_cl_ctrl.use_instance_cl
        self.use_temporal_cl = cfg.model.soft_cl_ctrl.use_temporal_cl

        # revin参数
        self.revin = cfg.model.revin.use_revin
        self.n_vars = cfg.dataset.n_vars
        if self.revin:
            self.revin_layer = RevIN(
                self.n_vars,
                affine=cfg.model.revin.affine,
                subtract_last=cfg.model.revin.subtract_last,
            )

        # patch的参数
        self.seq_len = cfg.model.seq_len
        self.patch_len = cfg.model.patch_len
        self.stride = cfg.model.stride
        self.padding_patch = cfg.model.padding_patch

        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        # decomposition
        self.decompose_layer = LearnableDecompose(
            cfg.dataset.n_vars
        )

        # encoder
        # out_dim因为维度对不齐的问题暂时不考虑
        self.trend_encoder = Encoder(
            d_model=self.patch_len,
            n_heads=cfg.model.encoder.n_heads,
            out_dim=cfg.model.encoder.out_dim,
            dropout=cfg.model.encoder.dropout,
            d_ff=cfg.model.encoder.d_ff,
            activation="gelu",
        )
        self.season_encoder = Encoder(
            d_model=self.patch_len,
            n_heads=cfg.model.encoder.n_heads,
            out_dim=cfg.model.encoder.out_dim,
            dropout=cfg.model.encoder.dropout,
            d_ff=cfg.model.encoder.d_ff,
            activation="gelu",
        )

        # fusion module
        self.fusion = FusionModule(
            d_model=self.patch_len, fusion_type=cfg.model.fusion.fusion_type
        )

        self.proj_dim = 2 * self.patch_len
        # projection head
        self.proj_head = TokenProjectionHead(
            in_dim=self.patch_len,
            hidden_dim=3 * self.patch_len,
            proj_dim=2 * self.patch_len,
        )

        # reconstruction module
        self.decoder = TokenDecoder(
            proj_dim=2 * self.patch_len,
            num_patches=self.patch_num,
            seq_len=self.seq_len,
            out_channels=self.n_vars,
            weight_mse=cfg.model.decoder.weight_mse,
            weight_cos=cfg.model.decoder.weight_cos,
        )
        
        # 对比学习计算类初始化
        if self.use_temporal_cl:
            self.soft_cl = Patch_Soft_CL(cfg, self.patch_num)

        if self.use_instance_cl:
            # DTW 计算与缓存类初始化
            self.dtw_cal_and_cache = DTW_Sim(
                dataset_name=cfg.dataset.name,
                patch_len=cfg.model.patch_len
            )

        self.save_hyperparameters()

    def configure_optimizers(self):
        """配置优化器(AdamW, 可按需扩展 scheduler)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # norm
        # x(B, C, L)
        x_remain, y = batch
        x = x_remain

        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)
        B, C, P, N = x.shape
        x = x.reshape(B * N, C, P)

        # calculate seq_level_dtw_sim_matrix
        # 暂时不实现，太麻烦了，还要考虑batch的适配性
        if self.use_instance_cl:
            dtw_mat = self.dtw_cal_and_cache.compute_and_save(x, batch_idx)
        

        # decompose
        x_trend, x_season = self.decompose_layer(x)

        x_trend = x_trend.reshape(B, N, C, P)
        x_season = x_season.reshape(B, N, C, P)
        x_trend = x_trend.reshape(B * C, N, P)
        x_season = x_season.reshape(B * C, N, P)

        # encoder
        x_trend_emb = self.trend_encoder(x_trend)
        x_season_emb = self.season_encoder(x_season)

        # generate samples by using fusion module
        self_samples, enhance_samples = self.gennerate_cl_samples(
            x_trend_emb, x_season_emb
        )

        # projection head
        x_proj = self.proj_head(self_samples)
        x_proj_enhance = self.proj_head(enhance_samples)

        # reconstruction
        x_rec = self.decoder(x_proj)
        if self.use_other_loss:
            other_loss, other_components = self.other_loss_calculator(
                x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain
            )
        x_proj = x_proj.reshape(B, C, N, 1, -1)
        K = x_proj_enhance.shape[2]
        x_proj_enhance = x_proj_enhance.reshape(B, C, N, K, -1)

        x_proj = x_proj.reshape(B*C, N, 1, -1).squeeze(2) # (B*C, N, d)
        x_proj_enhance = x_proj_enhance.reshape(B*C, N, K, -1) # (B*C, N, K, d)
        cl_loss = self.soft_cl(x_proj, x_proj_enhance)
        if self.use_other_loss:
            total_loss = self.cl_weight + other_loss
        else:
            total_loss = cl_loss

        # 增加监测指标
        batch_size = x_remain.size(0)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("cl_loss", cl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        if self.use_other_loss:
            self.log("spectral_loss", other_components["spectral"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            self.log("ortho_loss", other_components["ortho"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
            self.log("rec_loss", other_components["recon"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

        return total_loss
    
    def forward(self, x):
        # do patching
        if self.padding_patch == "end":
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)
        B, C, P, N = x.shape
        x = x.reshape(B * N, C, P)

        # decompose
        x_trend, x_season = self.decompose_layer(x)

        x_trend = x_trend.reshape(B, N, C, P)
        x_season = x_season.reshape(B, N, C, P)
        x_trend = x_trend.reshape(B * C, N, P)
        x_season = x_season.reshape(B * C, N, P)

        # encoder
        x_trend_emb = self.trend_encoder(x_trend)
        x_season_emb = self.season_encoder(x_season)

        # fusion
        self_samples = self.fusion(x_trend_emb, x_season_emb)
        self_samples = self_samples.unsqueeze(2)

        # projection head
        x_proj = self.proj_head(self_samples)
        x_proj = x_proj.squeeze(2)
        x_proj = x_proj.reshape(B, C, N, -1)

        return x_proj
    def gennerate_cl_samples(self, x_trend_emb, x_season_emb):
        """
        生成对比学习样本
        input:
        x_trend_emb: (B*C, N, patch_len)
        x_season_emb: (B*C, N, patch_len)
        output:
        self_samples: (B*C, N, 1 or 2, out_dim)
        enhance_samples: (B*C, N, 1 or 2, out_dim)
        """

        B_C, N, P = x_trend_emb.shape
        self_samples = self.fusion(x_trend_emb, x_season_emb)
        self_samples = self_samples.unsqueeze(2)

        if self.cl_shift_type == "shift_trend":
            x_trend_emb_enhance = self.change_order(x_trend_emb)
            enhance_samples = self.fusion(x_trend_emb_enhance, x_season_emb)
            enhance_samples = enhance_samples.unsqueeze(2)
        elif self.cl_shift_type == "shift_season":
            x_season_emb_enhance = self.change_order(x_season_emb)
            enhance_samples = self.fusion(x_trend_emb, x_season_emb_enhance)
            enhance_samples = enhance_samples.unsqueeze(2)
        elif self.cl_shift_type == "shift_both":
            x_trend_emb_enhance = self.change_order(x_trend_emb)
            x_season_emb_enhance = self.change_order(x_season_emb)
            enhance_samples_part1 = self.fusion(x_trend_emb, x_season_emb_enhance)
            enhance_samples_part2 = self.fusion(x_trend_emb_enhance, x_season_emb)
            enhance_samples_part1 = enhance_samples_part1.unsqueeze(2)
            enhance_samples_part2 = enhance_samples_part2.unsqueeze(2)
            enhance_samples = torch.cat(
                [enhance_samples_part1, enhance_samples_part2], dim=2
            )

        return self_samples, enhance_samples

    def change_order(self, x):
        """
        输入：[patch1, patch2, ..., patchN-1, patchN]
        输出：[patch2, patch3, ..., patchN, patchN-1]
        """
        x_shift = x[:, 1:, :]
        x_last_patch = x[:, -2, :].unsqueeze(1)
        out = torch.cat([x_shift, x_last_patch], dim=1)

        return out

    def other_loss_calculator(
        self, x_trend, x_season, x_trend_emb, x_season_emb, x_rec, x_remain
    ):
        # spectral loss
        spectral_loss = spectral_loss_from_raw(
            x_trend, x_season, cutoff_ratio=self.cfg.model.spectral_loss.cutoff_ratio
        )
        # token orthogonality loss
        ortho_loss = token_ortho_loss(x_trend_emb, x_season_emb)
        # rec_loss
        if self.revin:
            x_rec = x_rec.permute(0, 2, 1)
            x_rec = self.revin_layer(x_rec, "denorm")
            x_rec = x_rec.permute(0, 2, 1)

        rec_loss = self.decoder.rec_loss(x_rec, x_remain)

        if getattr(self.cfg.train, "normalize_loss", False):
            denom = rec_loss.detach() + spectral_loss.detach() + ortho_loss.detach() + EPS
            rec_loss_bal = rec_loss / denom
            spectral_loss_bal = spectral_loss / denom
            ortho_loss_bal = ortho_loss / denom
            loss = (
                rec_loss_bal * self.rec_weight
                + spectral_loss_bal * self.spectral_weight
                + ortho_loss_bal * self.ortho_weight
            )
        else:
            loss = (
                rec_loss * self.rec_weight
                + spectral_loss * self.spectral_weight
                + ortho_loss * self.ortho_weight
            )

        return loss, {"recon": rec_loss.detach(), "spectral": spectral_loss.detach(), "ortho": ortho_loss.detach()}

if __name__ == "__main__":

    cfg = OmegaConf.load("./cl_conf/pretrain_cfg_v3.yaml")

    dataset = DCLT_data_loader_v3(cfg=cfg)

    model = LitModel(cfg)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    train_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        devices="auto",
        log_every_n_steps=1,
        accelerator="auto",
    )
    from pytorch_lightning.utilities.model_summary import summarize

    print(summarize(model, max_depth=2))
    trainer.fit(model, train_loader, val_dataloaders=None)
