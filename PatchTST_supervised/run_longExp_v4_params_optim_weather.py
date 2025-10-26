#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正后的 Optuna + 训练脚本
说明：
- 所有训练/模型相关的 argparse 参数都在同一个 parser 中定义，避免多处 parser/重复注册的问题。
- 当 --trials > 0 时只运行 Optuna 搜索；若要在搜索后再用最优超参跑一次最终训练，请添加 --do-final。
- 每个 trial 使用 deep copy 的 base_args（即 CLI 解析得到的 args）作为模板，以避免污染。
- 需要在 exp_main.py 的 train() 中加入：
    self.optuna_trial = getattr(self.args, 'optuna_trial', None)
  并在每个 epoch 上报 vali_loss 并调用 trial.should_prune()（见之前补丁说明）。
"""

import argparse
import copy
import json
import os
import random
import time
import sys
import gc
from argparse import Namespace

import numpy as np
import torch
import optuna
import logging

from exp.exp_main import Exp_Main  # 你的 Exp 主类
# 如果你的 Exp 在别处，按需修改导入路径

# -------------------------
# Objective 函数（每个 trial 调用）
# -------------------------
def objective(trial, base_args: Namespace):
    # 拷贝 base_args，保证每个 trial 的 args 互相隔离
    args = copy.deepcopy(base_args)

    # ========== 搜索空间（按需修改） ==========
    # learning_rate: 在对数尺度上搜索连续区间（旧版也可用 suggest_loguniform）
    args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-3)
    
    args.d_model = trial.suggest_categorical('d_model', [128, 256])
    args.pcle_outdims = trial.suggest_categorical('pcle_outdims', [256, 384, 512])
    args.enable_cross_attn = trial.suggest_categorical('enable_cross_attn', [True, False])
    args.d_ff = trial.suggest_categorical('d_ff', [256, 384])
    # dropout: 离散候选值
    args.dropout = trial.suggest_categorical('dropout', [0.0, 0.2, 0.5])

    args.batch_size = trial.suggest_categorical('batch_size', [64, 128])
    args.seq_len = trial.suggest_categorical('seq_len', [96])
    args.pred_len = trial.suggest_categorical('pred_len', [96, 336])
    args.patch_len = trial.suggest_categorical('patch_len', [8, 16, 24, 32])
    args.stride = trial.suggest_categorical('stride', [4, 8, 12, 16])
    args.pcle_depth = trial.suggest_categorical('pcle_depth', [4, 8, 10])
    args.pcle_temporal_unit = trial.suggest_categorical('pcle_temporal_unit', [0, 1, 2])

    # cl_weight: 连续搜索区间
    args.cl_weight = trial.suggest_float('cl_weight', 0.05, 0.5)
    args.pcle_tau_temp = trial.suggest_float('pcle_tau_temp', 0.5, 1.5)


    # ==========================================

    # 每个 trial 使用不同但可复现的 seed
    seed = int(base_args.random_seed) + trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(seed)

    # 简单合法性校验，避免无效配置浪费时间
    if args.d_model % args.n_heads != 0:
        # 直接剪枝这种不兼容的配置
        raise optuna.exceptions.TrialPruned()

    # 创建实验实例并把 trial 对象传入
    exp = Exp_Main(args)
    exp.args.optuna_trial = trial  # 要求你在 exp.train() 中读取这个属性并 report/prune

    setting = f"{args.model_id}_{args.model}_trial{trial.number}"

    try:
        exp.train(setting)
    except optuna.exceptions.TrialPruned:
        # 记为剪枝
        del exp
        gc.collect()
        torch.cuda.empty_cache()
        raise
    except Exception as e:
        # 训练崩溃当作坏的 trial
        print(f"[Trial {trial.number}] Training crashed: {e}", file=sys.stderr)
        del exp
        gc.collect()
        torch.cuda.empty_cache()
        return float('inf')

    # 训练后在验证集上评估
    try:
        criterion = exp._select_criterion()
        vali_data, vali_loader = exp._get_data(flag='val')
        vali_ret = exp.vali(vali_data, vali_loader, criterion)
        if isinstance(vali_ret, (tuple, list)):
            val_loss = float(vali_ret[0])
        else:
            val_loss = float(vali_ret)
    except Exception as e:
        print(f"[Trial {trial.number}] Validation failed: {e}", file=sys.stderr)
        val_loss = float('inf')

    # 清理资源
    del exp
    gc.collect()
    torch.cuda.empty_cache()

    return val_loss


# -------------------------
# 主逻辑：解析参数、运行 Optuna、可选最终训练
# -------------------------
def main():
    # 统一在同一个 parser 中定义所有训练相关的参数（避免重复注册/解析问题）
    parser = argparse.ArgumentParser(description="Optuna tuning + training script")

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PCLE_v4',
                        help='model name, options: [Autoformer, Informer, Transformer, PatchTST, PatchTST_pretrained_v3, PatchTST_pretrained_v4, PCLE_v4]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./PatchTST_supervised/checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=21, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='params_optim', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


    # 改进模块的参数放在这里

    # # cl_model_gene表明使用的是第几代cl模型，1表示cl模型是变量级的，2表示cl模型是patch级的，不同模型要对应不同的数据加载方式
    # parser.add_argument('--cl_model_gene', type=int, default=3, help='generation of cl model')

    # # cross-attention模块
    # parser.add_argument('--use_cross_attention', type=bool, default=True, help='whether to use cross-attention; True 1 False 0')
    # parser.add_argument('--add_pos', type=str, default='emb_x_backbone', help='whether to add positional encoding in cross-attention, emb_x_backbone: between emb and backbone, backbone_x_head: between backbone and head')
    # parser.add_argument('--cross_attention_type', type=str, default='full', help='full or patch')

    # v3版本的cl模型
    parser.add_argument('--use_pretrained_cl', type=bool, default=False, help='whether to use pretrained cl model; True 1 False 0')
    parser.add_argument('--enable_cross_attn', type=bool, default=False, help='whether to use cross-attention; True 1 False 0')

    # v4版本没有加新的参数
    parser.add_argument('--load_mode', type=str, default='pkl', help='load mode, ckpt or onnx')
    parser.add_argument('--model_state', type=str, default='reason', help='model state, reason or finetune')
    parser.add_argument('--pretrain_folder', type=str, default='24_05_20_16_1_336_256')
    parser.add_argument('--pretrain_d_model', type=int, default=128, help='d_model used in pretraining')

    # PCLE模型新增参数
    parser.add_argument('--use_PCLE', type=bool, default=True, help='whether to use Patch Contrastive Learning Embedding; True 1 False 0')
    parser.add_argument('--pcle_feature_extract_net', type=str, default='dilated_conv', help='feature extract net for PCLE module, dilated_conv, ')
    parser.add_argument('--pcle_temporal_unit', type=int, default=0, help='temporal unit for PCLE module, patch or timepoint')
    parser.add_argument('--pcle_outdims', type=int, default=128, help='output dims of PCLE module')
    parser.add_argument('--pcle_hidden_dims', type=int, default=64, help='hidden dims of PCLE module')

    parser.add_argument('--pcle_depth', type=int, default=8, help='depth of PCLE module')
    parser.add_argument('--lambda_', type=float, default=0.5, help='weight of instance contrastive loss and temporal contrastive loss')
    parser.add_argument('--pcle_soft_instance', type=bool, default=False, help='whether to use soft instance for PCLE module; True 1 False 0')
    parser.add_argument('--tau_temp', type=float, default=0.5, help='temperature parameter for temporal contrastive loss in PCLE module')
    parser.add_argument('--pcle_soft_temporal', type=bool, default=True, help='whether to use soft temporal for PCLE module; True 1 False 0')
    parser.add_argument('--tau_inst', type=float, default=0.5, help='temperature parameter for instance contrastive loss in PCLE module')
    parser.add_argument('--cl_weight', type=float, default=0.01, help='weight for contrastive learning loss')

    # ========== Optuna / script control 参数 ==========
    parser.add_argument('--trials', type=int, default=30, help='number of optuna trials; 0 means skip optuna')
    parser.add_argument('--study-name', type=str, default='exp_optuna_study')
    parser.add_argument('--storage', type=str, default=None, help='optuna storage, e.g. sqlite:///optuna.db')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--debug', action='store_true', help='debug prints')
    parser.add_argument('--do-final', action='store_true', help='after optuna, run final training with best params')
    parser.add_argument('--logging-level', type=str, default='INFO', help='logging level')

    args = parser.parse_args()

    # 配置 logging
    logging.basicConfig(
        level=getattr(logging, args.logging_level.upper(), logging.INFO),
        format='[%(asctime)s][%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # ========== GPU 处理（修正并保持兼容） ==========
    args.use_gpu = True if torch.cuda.is_available() and bool(args.use_gpu) else False
    if args.use_gpu and getattr(args, 'use_multi_gpu', False):
        args.devices = args.devices.replace(' ', '')
        device_ids = [d for d in args.devices.split(',') if d != '']
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0] if len(args.device_ids) > 0 else 0
        # 可选：设置 CUDA_VISIBLE_DEVICES（根据你的集群策略决定是否打开）
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.device_ids))
        # torch.cuda.set_device(0)
    elif args.use_gpu:
        # 当只用单卡时，确保使用提供的 args.gpu（不修改 CUDA_VISIBLE_DEVICES，除非你需要）
        try:
            torch.cuda.set_device(int(args.gpu))
        except Exception:
            pass

    # ========== 全局固定随机种子（可保留） ==========
    fix_seed = int(args.random_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(fix_seed)

    # base_args 就用 CLI 解析得到的 args 的深拷贝作为模板
    base_args = copy.deepcopy(args)

    # ========== 如果开启 optuna 搜索 ==========
    if args.trials and args.trials > 0:
        sampler = optuna.samplers.TPESampler(seed=base_args.random_seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        study = optuna.create_study(direction='minimize',
                                    study_name=args.study_name,
                                    sampler=sampler,
                                    pruner=pruner,
                                    storage=args.storage,
                                    load_if_exists=True)
        print(f"Starting Optuna study '{args.study_name}' with {args.trials} trials (n_jobs={args.n_jobs})")
        try:
            # 关键：把 base_args 传入 objective（每个 trial 内会 deep copy）
            study.optimize(lambda t: objective(t, base_args), n_trials=args.trials, n_jobs=args.n_jobs)
        except KeyboardInterrupt:
            print("Optuna interrupted by user.")

        # 输出 & 保存最好的超参
        print("Study finished. Best trial:")
        if study.best_trial is not None:
            print(f"  Trial number: {study.best_trial.number}")
            print(f"  Value (val loss): {study.best_trial.value:.6f}")
            print("  Params:")
            for k, v in study.best_trial.params.items():
                print(f"    {k}: {v}")
            outfn = f"optuna_best_{args.study_name}.json"
            with open(outfn, "w") as f:
                json.dump(study.best_trial.params, f, indent=2)
            print(f"Saved best params to {outfn}")

            # 如果用户指定 --do-final，使用 best params 覆盖 args 并做一次最终训练/测试
            if args.do_final:
                print("Running final training with best params...")
                best_params = study.best_trial.params
                # 覆盖 base_args 中对应的字段（类型转换请视需要扩展）
                for k, v in best_params.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                    else:
                        print(f"Warning: args has no attribute {k}, skipping assignment")
                # 运行最终训练
                Exp = Exp_Main
                for ii in range(args.itr):
                    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
                        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                        args.factor, args.embed, args.distil, args.des, ii)
                    exp = Exp(args)
                    print('>>>>>>>start final training : {}>>>>>>>>'.format(setting))
                    exp.train(setting)
                    print('>>>>>>>final testing : {}<<<<<<<<'.format(setting))
                    exp.test(setting)
                    if args.do_predict:
                        exp.predict(setting, True)
                    torch.cuda.empty_cache()
        else:
            print("No finished trials found.")
    else:
        print("Optuna trials == 0 -> skip optuna search.")


if __name__ == "__main__":
    main()
