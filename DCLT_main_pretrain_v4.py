import torch
import numpy as np
import argparse
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import sys
import time
import random
import datetime
from typing import Any, Dict
try:
    import yaml  # 用于保存训练配置到 YAML
except Exception:
    yaml = None

import tasks_v4
from utils_v4 import datautils
from utils_v4.utils import init_dl_program, pkl_save, data_dropout

from utils_v4.utils_distance_matrix import * 
import sys

print(os.getcwd())

from utils.Mypydebug import show_shape
show_shape()

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

class _CheckpointManager:
    """简单的 checkpoint 管理器：
    - 始终更新 latest.pkl
    - 当 loss 改善时更新 best.pkl
    """
    def __init__(self, ckpt_dir: str, save_every: int = 1):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_every = max(1, int(save_every)) if save_every is not None else 1
        self.best_loss = float('inf')
        self.best_epoch = -1

    def __call__(self, model, loss: float):
        epoch = getattr(model, 'n_epochs', None)
        if epoch is None:
            # 兜底：没有 epoch 计数则不保存
            return

        # 最新
        latest_path = os.path.join(self.ckpt_dir, 'latest.pkl')
        model.save(latest_path)

        # 最优
        if loss is not None and loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            best_path = os.path.join(self.ckpt_dir, 'best.pkl')
            model.save(best_path)

def _to_builtin(obj: Any) -> Any:
    """将 argparse.Namespace / numpy 类型 转为可 YAML/JSON 序列化的基础类型。"""
    if isinstance(obj, argparse.Namespace):
        return {k: _to_builtin(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_builtin(v) for v in obj ]
    # numpy -> python 标量
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

def _save_yaml_safe(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = _to_builtin(data)
    if yaml is not None:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    else:
        # 无 PyYAML 时，退化为 JSON 风格的文本保存，扩展名仍为 .yaml
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

def fix_seed(expid):
    SEED = 2000 + expid
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_n_vars(args):
    if args.dataset == 'ETTh1':
        args.n_vars = 7
    elif args.dataset == 'ETTh2':
        args.n_vars = 7
    elif args.dataset == 'ETTm1':
        args.n_vars = 7
    elif args.dataset == 'ETTm2':
        args.n_vars = 7
    elif args.dataset == 'weather':
        args.n_vars = 21
    elif args.dataset == 'electricity':
        args.n_vars = 321
    elif args.dataset == 'Solar':
        args.n_vars = 137
    elif args.dataset == 'Traffic':
        args.n_vars = 862
    elif args.dataset == 'ILI':
        args.n_vars = 7
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="ETTh1", help='The dataset name')
    parser.add_argument('--loader', type=str, default="forecast_csv", help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--dist_type', type=str, default='DTW')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=256, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=336, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=300, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--eval', type=bool, default=False, help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--tau_inst', type=float, default=0.5)
    parser.add_argument('--tau_temp', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--expid', type=int, default=131)
    parser.add_argument('--separate_reg', action="store_true", help='Whether to perform weighting in temporal loss')
    parser.add_argument('--test_encode', type=bool, default=False, help='Whether to test the encoder after training')
    # 添加的额外参数
    parser.add_argument('--use_patch_cl', type=bool, default=True, help='Whether to use patch-level contrastive learning')
    parser.add_argument('--patch_cl_ver', type=int, default=1, help='The version of patch-level contrastive learning model to use (1 or 2)')
    parser.add_argument('--patch_len', type=int, default=16, help='The length of each patch when using patch-level contrastive learning')
    parser.add_argument('--patch_stride', type=int, default=2, help='The stride between patches when using patch-level contrastive learning')
    parser.add_argument('--depth', type=int, default=8, help='The depth of the model when using patch-level contrastive learning')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--TS', type=str, default=None, help='Timestamp')
    args = parser.parse_args()

    args = init_n_vars(args) # 根据数据集名称初始化 n_vars 参数
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    args.soft_instance = (args.tau_inst > 0)
    args.soft_temporal = (args.tau_temp > 0)
    args.soft_cl = args.soft_instance + args.soft_temporal
    if 'forecast' in args.loader:
        args.soft_instance = False
    
    if args.loader in ['UCR','UEA']:
        result_name = f'results_classification_{args.loader}/'
    else:
        result_name = f'results_{args.loader}/'
    
    run_dir = result_name + f'TEMP{int(args.soft_temporal)}_INST{int(args.soft_instance)}'
    if args.soft_cl:
        run_dir += f'_tau_temp{float(args.tau_temp)}_tau_inst{float(args.tau_inst)}' 
    if args.lambda_==0:
        run_dir += '_no_instCL'
        
    run_dir = os.path.join(run_dir, args.dataset, f'bs{args.batch_size}', f'run{args.expid}')
    os.makedirs(run_dir, exist_ok=True)

    # =============== Checkpoints 目录（根据用户要求） ===============
    # ./checkpoints/{dataset}/{patch_len}-{repr-dims}v4{时间戳}/
    timestamp = args.TS
    ckpt_sub = f"{timestamp}_{args.epochs}_{args.patch_len}_{args.patch_stride}_{args.max_train_length}_{args.repr_dims}"
    ckpt_dir = os.path.join('.', 'checkpoints', args.dataset, ckpt_sub)
    os.makedirs(ckpt_dir, exist_ok=True)
    # 中断输出同时保存到checkpoints下的run.log文件
    import atexit
    
    log_file_path = os.path.join(ckpt_dir, 'run.log')
    try:
        _log_f = open(log_file_path, 'a', encoding='utf-8')
    except Exception:
        _log_f = None

    class Tee(object):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    if _log_f is not None:
        sys.stdout = Tee(sys.stdout, _log_f)
        sys.stderr = Tee(sys.stderr, _log_f)
        atexit.register(lambda: _log_f.close())
        # 立即把关键信息写入日志文件（有些早期打印可能发生在 Tee 安装之前）
        try:
            _log_f.write('----- RUN HEADER -----\n')
            _log_f.write(f'Timestamp: {timestamp}\n')
            try:
                _log_f.write(f'Dataset: {args.dataset}\n')
                _log_f.write(f'Arguments: {str(args)}\n')
            except Exception:
                # args 可能尚未完全可用，忽略写入错误
                pass
            _log_f.write('----------------------\n')
            _log_f.flush()
        except Exception as e:
            print(f"[warn] failed to write header into run.log: {e}")
    
    # file_exists = os.path.join(run_dir,f'eval_res_{args.dist_type}.pkl')
        
    # if os.path.isfile(file_exists):
    #     print('You alreay have the results. Bye Bye~')
    #     sys.exit(0)
    
    fix_seed(args.expid)
    device = init_dl_program(args.gpu, seed=args.seed)#, max_threads=args.max_threads)

    print('Loading data... ', end='')
    test_data = None
    valid_data = None
    if args.use_patch_cl:
        if args.loader == 'forecast_csv':
            task_type = 'forecasting'
            UNI = False
            data, train_slice, valid_slice, test_slice, scaler, pred_lens = datautils.load_forecast_csv_patch(args.dataset, args.max_train_length, univar=UNI)
            train_data = data[:, train_slice]
            valid_data = data[:, valid_slice]
            test_data = data[:, test_slice]
            n_covariate_cols = 0
    else:
        if args.loader == 'UCR':
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels, sim_mat = datautils.load_UCR(args.dataset, args.dist_type)
            
        elif args.loader == 'UEA':
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels, sim_mat = datautils.load_UEA(args.dataset, args.max_train_length)
        
        elif args.loader == 'semi':
            task_type = 'semi-classification'
            train_data, train_labels, train1_data, train1_labels, train5_data, train5_labels, test_data, test_labels, sim_mat = datautils.load_semi_SSL(args.dataset)
        
        elif args.loader == 'forecast_csv':
            task_type = 'forecasting'
            UNI = False
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, args.max_train_length, univar=UNI)
            train_data = data[:, train_slice]
            valid_data = data[:, valid_slice]
            test_data = data[:, test_slice]
            
        elif args.loader == 'forecast_csv_univar':
            task_type = 'forecasting'
            UNI = True
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, args.max_train_length, univar=UNI)
            train_data = data[:, train_slice]
            test_data = data[:, test_slice]
            
        elif args.loader == 'anomaly':
            task_type = 'anomaly_detection'
            train_data, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, args.max_train_length, cold=False)
            
        elif args.loader == 'anomaly_coldstart':
            task_type = 'anomaly_detection_coldstart'
            _, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, args.max_train_length, cold=True)
            train_data, _, _, _, _ = datautils.load_UCR('FordA')
            
        else:
            raise ValueError(f"Unknown loader {args.loader}.")
            
        if args.irregular > 0:
            if task_type == 'classification':
                train_data = data_dropout(train_data, args.irregular)
                test_data = data_dropout(test_data, args.irregular)
            else:
                raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    if args.use_patch_cl:
        config = dict(
            n_vars=args.n_vars,
            batch_size=args.batch_size,
            lr=args.lr,
            tau_temp=args.tau_temp,
            lambda_ = args.lambda_,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length,
            soft_instance = args.soft_instance,
            soft_temporal = args.soft_temporal,
            # extra params
            patch_len = args.patch_len,
            patch_stride = args.patch_stride,
            depth = args.depth,
            revin = args.revin
        )
    else:
        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            tau_temp=args.tau_temp,
            lambda_ = args.lambda_,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length,
            soft_instance = args.soft_instance,
            soft_temporal = args.soft_temporal
        )

    # 将本次训练的初始化参数，按 TS2Vec __init__ 的顺序写入 config.yaml（仅保留初始化需要的参数）
    # 适配不同版本的 TS2Vec
    # 计算 input_dims
    try:
        device_str = str(device)
    except Exception:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.use_patch_cl:
        input_dims_for_yaml = args.patch_len
        init_param_names = [
            'n_vars','patch_cl_ver','input_dims','output_dims','hidden_dims','depth','device','lr','batch_size',
            'lambda_','tau_temp','max_train_length','temporal_unit','soft_instance','soft_temporal',
            'patch_len','patch_stride','padding_patch'
        ]
        init_values = [
            args.n_vars,
            args.patch_cl_ver,
            input_dims_for_yaml,
            args.repr_dims,
            64,
            args.depth,
            device_str,
            args.lr,
            args.batch_size,
            args.lambda_,
            args.tau_temp,
            args.max_train_length,
            0,
            args.soft_instance,
            args.soft_temporal,
            args.patch_len,
            args.patch_stride,
            'end'
        ]

    # 按顺序构造有序字典
    init_params_ordered: Dict[str, Any] = {}
    for k, v in zip(init_param_names, init_values):
        init_params_ordered[k] = v

    _save_yaml_safe(os.path.join(ckpt_dir, 'config.yaml'), init_params_ordered)
    
    # 始终以 epoch 为单位进行 checkpoint 保存（满足用户需求）
    ckpt_manager = _CheckpointManager(ckpt_dir, save_every=args.save_every if args.save_every is not None else 1)
    config['after_epoch_callback'] = ckpt_manager
    
    t = time.time()
    if args.use_patch_cl:
        if args.patch_cl_ver == 1:
            from cl_models_v4.PCLE_Model_ver1 import TS2Vec
        elif args.patch_cl_ver == 2:
            from cl_models_v4.PCLE_Model_ver2 import TS2Vec
            
    else:
        from cl_models_v4.soft_ts2vec import TS2Vec

    if args.use_patch_cl:
        if args.patch_cl_ver == 1:
            model = TS2Vec(
                input_dims=args.patch_len,
                device=device,
                patch_cl_ver=args.patch_cl_ver,
                **config
        )
        elif args.patch_cl_ver == 2:
            pass
    else:
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            **config
        )

    if args.soft_instance:
        dist_mat = 1 - sim_mat
        del sim_mat
        sim_mat = densify(-dist_mat, args.tau_inst, args.alpha)
        print('Soft Assignment Matrix', sim_mat.shape)
    
    if task_type in ['forecasting','anomaly_detection','anomaly_detection_coldstart']:
        sim_mat = None
        
    if task_type == 'classification':
        loss_log = model.fit(
            train_data, train_labels, test_data, test_labels,
            sim_mat, run_dir, n_epochs=args.epochs, n_iters=args.iters,
            verbose=True)
    else:
        loss_log = model.fit(
            train_data, 0, 0, 0,
            sim_mat, run_dir, n_epochs=args.epochs, n_iters=args.iters,
            verbose=True,
            valid_data=valid_data,
            test_data_eval=test_data)
    
    # 训练结束：仅保留 latest.pkl 与 best.pkl（latest 在最后一个 epoch 回调中已更新）
    model.save(f'{run_dir}/model.pkl')
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.test_encode:
        encode_inputs = None
        encode_kwargs = {}

        if test_data is not None:
            encode_inputs = test_data
        elif task_type == 'forecasting':
            encode_inputs = data[:, test_slice]

        if encode_inputs is None:
            print('Skip encoder test: no evaluation split available for encode().')
        else:
            if task_type in ('classification', 'semi-classification'):
                encode_kwargs['encoding_window'] = 'full_series'

            if args.max_train_length is not None and encode_inputs.shape[1] > args.max_train_length:
                encode_kwargs['sliding_length'] = args.max_train_length
                encode_kwargs['sliding_padding'] = 0 if task_type == 'forecasting' else 0

            encode_repr = model.encode(encode_inputs, **encode_kwargs)
            print(f'Encoded representations shape: {encode_repr.shape}')

    if args.eval:
        if args.use_patch_cl:
            if args.patch_cl_ver == 1:
                out, eval_res = tasks_v4.eval_forecasting_patch(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        else:
            if task_type == 'classification':
                out, eval_res = tasks_v4.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
            elif task_type == 'forecasting':
                if args.dataset == 'electricity':
                    if args.separate_reg:
                        out, eval_res = tasks_v4.eval_forecasting_separate(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
                    else:
                        out, eval_res = tasks_v4.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
                else:
                    if args.separate_reg:
                        out, eval_res = tasks_v4.eval_forecasting_separate(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
                    else:
                        out, eval_res = tasks_v4.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=UNI)
            elif task_type == 'anomaly_detection':
                out, eval_res = tasks_v4.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
            elif task_type == 'anomaly_detection_coldstart':
                out, eval_res = tasks_v4.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
            elif task_type == 'semi-classification':
                out, eval_res = tasks_v4.eval_semi_classification(model, 
                                                            train1_data, train1_labels, 
                                                            train5_data, train5_labels, 
                                                            train_labels,
                                                            test_data, test_labels, eval_protocol='svm')
            else:
                assert False
            
        #pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res_{args.dist_type}.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
