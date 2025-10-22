import numpy as np
import time
from . import _eval_protocols as eval_protocols
import torch
# from torch.utils.data import DataLoader, TensorDataset
# from layers.RevIN import RevIN
# from PatchTST_supervised.data_provider.data_loader import Dataset_Custom

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])
            
def generate_pred_samples_norm(features, data, pred_len, drop=0):
    n = data.shape[1]

    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i : 1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    last_val = data[:,:labels.shape[1],:] #last_val = data[0,:labels.shape[1],0]
    features = features[:, drop:]
    labels = labels[:, drop:]
    last_val = last_val[:,drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3]), last_val[0,:,:]            

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
        
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        
        
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        #fadfsd
        t = time.time()

        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = temp_2d.T
                test_pred_inv = temp_2d.reshape(test_pred.shape) 

                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = temp_2d.T
                test_labels_inv = temp_2d.reshape(test_labels.shape)

            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d)
                test_pred_inv = temp_2d.reshape(test_pred.shape)
                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d)
                test_labels_inv = temp_2d.reshape(test_labels.shape)
            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                
            #test_pred_inv = scaler.inverse_transform(test_pred)
            #test_labels_inv = scaler.inverse_transform(test_labels)
        
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res

def eval_forecasting_patch(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols):
    # 校验模型的patch配置是否完整，缺失时直接报错提醒使用者补全参数
    patch_len = getattr(model, 'patch_len', None)
    patch_stride = getattr(model, 'patch_stride', None)
    if patch_len is None or patch_stride is None or patch_stride <= 0:
        raise ValueError('Patch configuration is required for patch forecasting.')

    # 推理阶段统一转到模型所在的设备上，同时依据样本数自适应批大小
    device = getattr(model, 'device', torch.device('cpu'))
    batch_size = min(256, data.shape[0]) if data.shape[0] > 0 else 1
    dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 逐批抽取patch并通过encoder编码，随后还原为时间步级别的表示
    infer_start = time.time()
    repr_chunks = []
    org_training = model.net.training
    model.net.eval()
    with torch.no_grad():
        for (batch,) in loader:
            # batch形状：(B, T, F) —— B为样本数、T为原始时间步长度、F为变量数
            batch = batch.to(device)
            B, T, F = batch.shape
            # permute后张量变为(C通道, 时间)的布局，方便调用unfold切patch
            patch_input = batch.permute(0, 2, 1)
            # 若训练时做过末尾复制填充，这里保持一致，以避免最后一个patch长度不足
            if getattr(model, 'padding_patch', None) == 'end' and hasattr(model, 'padding_patch_layer'):
                patch_input = model.padding_patch_layer(patch_input)
            # unfold得到(B, C, N, P)：N为patch个数，P为patch长度
            patches = patch_input.unfold(dimension=-1, size=patch_len, step=patch_stride)
            if patches.numel() == 0:
                raise ValueError('Unfold produced an empty tensor; check patch settings.')
            Bp, C, N, P = patches.shape
            # 将通道展平成批维度，交给encoder逐patch时间步处理
            patches = patches.reshape(Bp * C, N, P)
            patch_repr = model.net(patches)
            repr_dim = patch_repr.shape[-1]
            # 编码后再还原出(B, C, N, D)，并对不同通道求平均得到patch级表示
            patch_repr = patch_repr.reshape(B, C, N, repr_dim).mean(dim=1)

            # 将patch级特征均匀复制回原始时间轴，保持与forecasting评估函数一致
            time_repr = torch.empty(B, T, repr_dim, device=patch_repr.device)
            filled = 0
            for idx in range(N):
                start = idx * patch_stride
                if start >= T:
                    break
                end = min(start + patch_stride, T) if idx < N - 1 else T
                if end <= start:
                    continue
                # 对应时间段复制同一个patch嵌入，近似视为滑动窗口的平铺
                time_repr[:, start:end] = patch_repr[:, idx:idx+1, :].expand(-1, end - start, -1)
                filled = end
            # 针对可能尾部未覆盖的时间步，使用最后一个patch的表示进行填充
            if filled < T:
                time_repr[:, filled:] = patch_repr[:, -1:, :].expand(-1, T - filled, -1)
            repr_chunks.append(time_repr.cpu())
    model.net.train(org_training)
    ts2vec_infer_time = time.time() - infer_start

    if not repr_chunks:
        raise ValueError('No representations were produced for forecasting.')
    all_repr = torch.cat(repr_chunks, dim=0).numpy()

    # 划分训练、验证、测试时间片，与原始forecasting评估逻辑保持一致
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    base_padding = patch_len
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        max_drop = max(0, train_repr.shape[1] - pred_len - 1)
        drop = min(base_padding, max_drop)

        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=drop)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        t = time.time()
        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t

        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            a, b, c, d = test_pred.shape
            # 多变量多样本场景：转置+reshape，便于整体送入scaler做逆标准化
            test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))
            test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))
            test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
            test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
            test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
            test_pred_inv = test_pred_reshaped.reshape(a, b, c, d)

            test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))
            test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))
            test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
            test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
            test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
            test_labels_inv = test_labels_reshaped.reshape(a, b, c, d)
        else:
            a, b, c, d = test_pred.shape
            # 单样本时的处理逻辑与多样本一致，只是维度更小
            test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))
            test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))
            test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
            test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
            test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
            test_pred_inv = test_pred_reshaped.reshape(a, b, c, d)

            test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))
            test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))
            test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
            test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
            test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
            test_labels_inv = test_labels_reshaped.reshape(a, b, c, d)

        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }

    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res


def eval_forecasting_norm(model, data, data_not_scaled, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data_not_scaled[:, train_slice, n_covariate_cols:]
    valid_data = data_not_scaled[:, valid_slice, n_covariate_cols:]
    test_data = data_not_scaled[:, test_slice, n_covariate_cols:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:

        train_features, train_labels, train_last_val = generate_pred_samples_norm(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels, valid_last_val = generate_pred_samples_norm(valid_repr, valid_data, pred_len)
        test_features, test_labels, test_last_val = generate_pred_samples_norm(test_repr, test_data, pred_len)
        t = time.time()

        lr = eval_protocols.fit_ridge_norm(train_features, train_labels, train_last_val, valid_features, valid_labels, valid_last_val)
        lr_train_time[pred_len] = time.time() - t
        t = time.time()
        test_pred = lr.predict(test_features)
        try:
            test_pred = test_pred+test_last_val
        except:
            test_pred = test_pred+np.repeat(test_last_val,pred_len,axis=1)
        
        lr_infer_time[pred_len] = time.time() - t
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                
                ######## temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_pred_inv = temp_2d.reshape(test_pred.shape) 

                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                
                ######## temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_labels_inv = temp_2d.reshape(test_labels.shape)

            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                
                ######## test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = scaler.transform(test_pred_reshaped.T)
                test_pred_reshaped = test_pred_reshaped.T
                
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                
                ######## test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = scaler.transform(test_labels_reshaped.T)
                test_labels_reshaped = test_labels_reshaped.T
                
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1)

                ######## temp_2d = scaler.inverse_transform(temp_2d)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_pred_inv = temp_2d.reshape(test_pred.shape)
                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                
                ######## temp_2d = scaler.inverse_transform(temp_2d)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_labels_inv = temp_2d.reshape(test_labels.shape)
            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                
                ######## test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = scaler.transform(test_pred_reshaped)
                
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                
                ######## test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = scaler.transform(test_labels_reshaped)
                
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                
            #test_pred_inv = scaler.inverse_transform(test_pred)
            #test_labels_inv = scaler.inverse_transform(test_labels)
        
        out_log[pred_len] = {
            'norm': test_pred_inv,
            'raw': test_pred,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred_inv, test_labels_inv),
            'raw': cal_metrics(test_pred, test_labels)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res


def eval_forecasting2(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
        
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        test_pred_list = []
        test_pred_inv_list = []
        test_labels_list = []
        test_labels_inv_list = []
        norm_metric_list = []
        raw_metric_list = []
        
        D = train_data.shape[2]

        
        for d in range(D):            

            train_features, train_labels = generate_pred_samples(train_repr, np.expand_dims(train_data[:,:,d],-1), pred_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples(valid_repr, np.expand_dims(valid_data[:,:,d],-1), pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, np.expand_dims(test_data[:,:,d],-1), pred_len)
            t = time.time()
            
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
            lr_train_time[pred_len] = time.time() - t
            t = time.time()
            test_pred = lr.predict(test_features)
            lr_infer_time[pred_len] = time.time() - t
            ori_shape = test_data.shape[0], -1, pred_len, 1
   
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)

            if test_data.shape[0] > 1:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_pred_inv = temp_2d.reshape(test_pred.shape) 

                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_labels_inv = temp_2d.reshape(test_labels.shape)

                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                    #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    test_pred_inv = temp_2d.reshape(test_pred.shape)
                    
                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d)
                    test_labels_inv = temp_2d.reshape(test_labels.shape)
                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    
                #test_pred_inv = scaler.inverse_transform(test_pred)
                #test_labels_inv = scaler.inverse_transform(test_labels)
            
            test_pred_list.append(test_pred)
            test_pred_inv_list.append(test_pred_inv)
            test_labels_list.append(test_labels)
            test_labels_inv_list.append(test_labels_inv)
            
            norm_metric_list.append(cal_metrics(test_pred, test_labels))
            raw_metric_list.append(cal_metrics(test_pred_inv, test_labels_inv))
            
        out_log[pred_len] = {
            'norm': test_pred_list,
            'raw': test_pred_inv_list,
            'norm_gt': test_labels_list,
            'raw_gt': test_labels_inv_list
        }
        
        ours_result[pred_len] = {
            'norm': norm_metric_list,
            'raw': raw_metric_list
        }

    
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return None , eval_res

# =========================
# Linear forecast probe
# =========================

# class LinearForecastProbe(torch.nn.Module):
#     """线性预测探测器
#     - 提取 TS2Vec 表征 (B*C, N, D)，展平为 (B*C, N*D)
#     - 通过线性层映射到不同预测步长 pred_len
#     - 使用 RevIN 归一化输入并在输出端反归一化
#     - 仅训练线性头，不更新 TS2Vec 编码器
#     """

#     def __init__(self, ts2vec_model, patch_len, patch_stride, padding_patch='end', use_revin=True, device=None):
#         super().__init__()
#         self.encoder = ts2vec_model
#         self.device = device if device is not None else getattr(ts2vec_model, 'device', torch.device('cpu'))
#         self.patch_len = int(patch_len)
#         self.patch_stride = int(patch_stride)
#         self.padding_patch = padding_patch
#         self.use_revin = bool(use_revin)

#         self.padding_patch_layer = None
#         if self.padding_patch == 'end':
#             self.padding_patch_layer = torch.nn.ReplicationPad1d((0, self.patch_stride))

#         self.revin_layer = None  # lazy init with C
#         self._flat_dim = None    # N*D
#         self.heads = torch.nn.ModuleDict()
#         self.to(self.device)

#     def _maybe_init_revin(self, C: int):
#         if not self.use_revin:
#             return
#         if self.revin_layer is None:
#             self.revin_layer = RevIN(C, affine=False, subtract_last=0).to(self.device)

#     def _forward_repr(self, x_btc: torch.Tensor):
#         # x_btc: (B, T, C)
#         B, T, C = x_btc.shape
#         self._maybe_init_revin(C)
#         x_norm = self.revin_layer(x_btc, 'norm') if self.use_revin else x_btc
#         # to (B, C, T)
#         x = x_norm.permute(0, 2, 1)
#         if self.padding_patch_layer is not None:
#             x = self.padding_patch_layer(x)
#         # (B, C, N, P)
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
#         Bp, Cc, N, P = x.shape
#         x = x.reshape(Bp * Cc, N, P)

#         # TS2Vec 编码，不更新参数
#         self.encoder.net.eval()
#         with torch.no_grad():
#             repr_bcn = self.encoder.encode(x)  # (B*C, N, D)
#         BmulC, N, D = repr_bcn.shape
#         flat = repr_bcn.reshape(BmulC, N * D)
#         if self._flat_dim is None:
#             self._flat_dim = flat.shape[1]
#         return flat, (B, C)

#     def _maybe_init_head(self, pred_len: int):
#         key = str(int(pred_len))
#         if key not in self.heads:
#             if self._flat_dim is None:
#                 raise RuntimeError('flat dim unknown before first forward')
#             self.heads[key] = torch.nn.Linear(self._flat_dim, int(pred_len), bias=True).to(self.device)

#     def forward(self, x_btc: torch.Tensor, pred_len: int):
#         flat, (B, C) = self._forward_repr(x_btc)
#         self._maybe_init_head(pred_len)
#         y = self.heads[str(int(pred_len))](flat)  # (B*C, pred_len)
#         y = y.reshape(B, C, int(pred_len)).permute(0, 2, 1).contiguous()  # (B, pred_len, C)
#         if self.use_revin:
#             y = self.revin_layer(y, 'denorm')
#         return y

#     def train_one_epoch(self, train_loader, optimizer, pred_lens):
#         self.train()
#         # 冻结编码器
#         for p in self.encoder.parameters():
#             p.requires_grad_(False)
#         mse = torch.nn.MSELoss()
#         epoch_loss = {int(pl): 0.0 for pl in pred_lens}
#         count = {int(pl): 0 for pl in pred_lens}

#         for seq_x, seq_y, _, _ in train_loader:
#             seq_x = seq_x.float().to(self.device)
#             seq_y = seq_y.float().to(self.device)
#             optimizer.zero_grad()
#             total = 0.0
#             loss_dict = {}
#             for pl in pred_lens:
#                 y_pred = self.forward(seq_x, pl)
#                 y_true = seq_y[:, -pl:, :]
#                 l = mse(y_pred, y_true)
#                 loss_dict[int(pl)] = l
#                 total = total + l
#             total.backward()
#             optimizer.step()
#             for pl, l in loss_dict.items():
#                 epoch_loss[int(pl)] += float(l.detach().item())
#                 count[int(pl)] += 1
#         for pl in epoch_loss:
#             if count[pl] > 0:
#                 epoch_loss[pl] /= count[pl]
#         return epoch_loss

#     @torch.no_grad()
#     def eval_loop(self, loader, pred_lens):
#         self.eval()
#         mse = torch.nn.MSELoss()
#         out = {int(pl): 0.0 for pl in pred_lens}
#         cnt = {int(pl): 0 for pl in pred_lens}
#         for seq_x, seq_y, _, _ in loader:
#             seq_x = seq_x.float().to(self.device)
#             seq_y = seq_y.float().to(self.device)
#             for pl in pred_lens:
#                 y_pred = self.forward(seq_x, pl)
#                 y_true = seq_y[:, -pl:, :]
#                 l = mse(y_pred, y_true)
#                 out[int(pl)] += float(l.item())
#                 cnt[int(pl)] += 1
#         for pl in out:
#             if cnt[pl] > 0:
#                 out[pl] /= cnt[pl]
#         return out


# def _build_patchtst_loaders(root_path, data_path, seq_len, label_len, pred_len,
#                             batch_size=128, num_workers=4, features='M', target='OT', freq='h', scale=True):
#     train_ds = Dataset_Custom(root_path=root_path, flag='train', size=(seq_len, label_len, pred_len),
#                               features=features, data_path=data_path, target=target, scale=scale, timeenc=0, freq=freq)
#     val_ds = Dataset_Custom(root_path=root_path, flag='val', size=(seq_len, label_len, pred_len),
#                             features=features, data_path=data_path, target=target, scale=scale, timeenc=0, freq=freq)
#     test_ds = Dataset_Custom(root_path=root_path, flag='test', size=(seq_len, label_len, pred_len),
#                              features=features, data_path=data_path, target=target, scale=scale, timeenc=0, freq=freq)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
#     return train_loader, val_loader, test_loader


# def linear_forecast_probe(ts2vec_model, dataset_name, pred_lens, seq_len=96, label_len=48,
#                           patch_len=16, patch_stride=2, padding_patch='end', use_revin=True,
#                           device=None, root_path='./dataset/', batch_size=128, epochs=1,
#                           num_workers=4, features='M', target='OT', freq='h'):
#     """基于 TS2Vec 表征的线性探测训练与评估。

#     返回：{pred_len: {'train': x, 'val': y, 'test': z}}
#     """
#     data_path = f'{dataset_name}.csv'
#     # 以较小的 pred_len 构建 Loader 的 size（Dataset_Custom 需要固定 size）
#     base_pl = int(min(pred_lens))
#     train_loader, val_loader, test_loader = _build_patchtst_loaders(
#         root_path, data_path, seq_len, label_len, base_pl,
#         batch_size=batch_size, num_workers=num_workers,
#         features=features, target=target, freq=freq, scale=True
#     )

#     probe = LinearForecastProbe(ts2vec_model, patch_len, patch_stride, padding_patch, use_revin, device)
#     optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

#     logs = {int(pl): {'train': None, 'val': None, 'test': None} for pl in pred_lens}
#     for ep in range(epochs):
#         train_loss = probe.train_one_epoch(train_loader, optimizer, pred_lens)
#         val_loss = probe.eval_loop(val_loader, pred_lens)
#         test_loss = probe.eval_loop(test_loader, pred_lens)
#         summary = [f'[Epoch {ep+1}]']
#         for pl in pred_lens:
#             logs[int(pl)]['train'] = train_loss[int(pl)]
#             logs[int(pl)]['val'] = val_loss[int(pl)]
#             logs[int(pl)]['test'] = test_loss[int(pl)]
#             summary.append(f'P{pl}: train={train_loss[int(pl)]:.6f} val={val_loss[int(pl)]:.6f} test={test_loss[int(pl)]:.6f}')
#         print(' '.join(summary))

#     return logs

def eval_forecasting2_norm(model, data, data_not_scaled, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data_not_scaled[:, train_slice, n_covariate_cols:]
    valid_data = data_not_scaled[:, valid_slice, n_covariate_cols:]
    test_data = data_not_scaled[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        test_pred_list = []
        test_pred_inv_list = []
        test_labels_list = []
        test_labels_inv_list = []
        norm_metric_list = []
        raw_metric_list = []
        
        D = train_data.shape[2]
 
        
        for d in range(D):            

            train_features, train_labels, train_last_val = generate_pred_samples_norm(train_repr, np.expand_dims(train_data[:,:,d],-1), pred_len, drop=padding)
            valid_features, valid_labels, valid_last_val = generate_pred_samples_norm(valid_repr, np.expand_dims(valid_data[:,:,d],-1), pred_len)
            test_features, test_labels, test_last_val = generate_pred_samples_norm(test_repr, np.expand_dims(test_data[:,:,d],-1), pred_len)
        

            t = time.time()
            lr = eval_protocols.fit_ridge_norm(train_features, train_labels, train_last_val, valid_features, valid_labels, valid_last_val)
            
            lr_train_time[pred_len] = time.time() - t
            t = time.time()
            test_pred = lr.predict(test_features)
            test_pred = test_pred+test_last_val
            lr_infer_time[pred_len] = time.time() - t
            ori_shape = test_data.shape[0], -1, pred_len, 1
  
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)

            if test_data.shape[0] > 1:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_pred_inv = temp_2d.reshape(test_pred.shape) 

                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_labels_inv = temp_2d.reshape(test_labels.shape)

                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    
                    ########test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = scaler.transform(test_pred_reshaped)

                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    
                    ########test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = scaler.transform(test_labels_reshaped)
                    
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                    #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    test_pred_inv = temp_2d.reshape(test_pred.shape)
                    
                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d)
                    test_labels_inv = temp_2d.reshape(test_labels.shape)
                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    
                    ########test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = scaler.transform(test_pred_reshaped)
                    
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    
                    ########test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = scaler.transform(test_labels_reshaped)
                    
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    
                #test_pred_inv = scaler.inverse_transform(test_pred)
                #test_labels_inv = scaler.inverse_transform(test_labels)
            
            test_pred_list.append(test_pred)
            test_pred_inv_list.append(test_pred_inv)
            test_labels_list.append(test_labels)
            test_labels_inv_list.append(test_labels_inv)
            
            norm_metric_list.append(cal_metrics(test_pred, test_labels))
            raw_metric_list.append(cal_metrics(test_pred_inv, test_labels_inv))
            
        out_log[pred_len] = {
            'norm': test_pred_inv_list,
            'raw': test_pred_list,
            'norm_gt': test_labels_inv_list,
            'raw_gt': test_labels_list
        }
        
        ours_result[pred_len] = {
            'norm': raw_metric_list,
            'raw': norm_metric_list
        }

    
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return None , eval_res
