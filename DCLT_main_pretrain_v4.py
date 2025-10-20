import torch
import numpy as np
import argparse
import os
import sys
import time
import random
import datetime
from cl_models_v4.CLE_Model import TS2Vec
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="weather", help='The dataset name')
    parser.add_argument('--loader', type=str, default="forecast_csv", help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--dist_type', type=str, default='DTW')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--eval', type=bool, default=True, help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--tau_inst', type=float, default=0.5)
    parser.add_argument('--tau_temp', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--expid', type=int, default=2)
    parser.add_argument('--separate_reg', action="store_true", help='Whether to perform weighting in temporal loss')
    parser.add_argument('--test_encode', type=bool, default=True, help='Whether to test the encoder after training')
    args = parser.parse_args()
    
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
    
    file_exists = os.path.join(run_dir,f'eval_res_{args.dist_type}.pkl')
        
    if os.path.isfile(file_exists):
        print('You alreay have the results. Bye Bye~')
        sys.exit(0)
    
    fix_seed(args.expid)
    device = init_dl_program(args.gpu, seed=args.seed)#, max_threads=args.max_threads)

    print('Loading data... ', end='')
    test_data = None
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
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    t = time.time()

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
            sim_mat,run_dir, n_epochs=args.epochs, n_iters=args.iters,
            verbose=True)
    
    #model.save(f'{run_dir}/model.pkl')
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
