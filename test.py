#%%
import _config
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.data_utils import get_libsvm_data, get_fit_dataloaders
from collections import Counter
from modules.np_modules import construct_np_model, train_np_model, eval_np_model
from utils.modules_utils import cal_fx, cal_km
import numpy as np
import torch
import time
import pytorch_lightning as pl
from modules.conv_modules import Basic_Block, CNN_Res18
from modules.data_modules import GeneralDataModule
from modules.lightning_modules import CompressionNet

datasets = _config.datasets
n_classes = _config.n_classes
labels = _config.labels

n_epochs = 3
model_type = 'svc'
kernel = 'rbf'
traverse_list = ['a9a']

map_size = 128
blocks = [2, 2, 2, 2]
rate_list = [0.4, 0.5, 0.6]

lr = 0.01

early_stop_callback = EarlyStopping(monitor='val_mae', min_delta=0.00, patience=3, verbose=False, mode='max')

for name in traverse_list:
    n_features = datasets[name]['n_features']
    is_multi = datasets[name]['is_multi']
    has_test = datasets[name]['has_test']

    # -------------------
    # Load Data
    # -------------------

    print('...Load data')
    X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test)

    # -------------------
    # Train np model
    # -------------------

    print('\n... Train np model')
    np_model = construct_np_model(model_type=model_type, kernel=kernel)
    np_model = train_np_model(np_model, X_train, y_train)
    eval_np_model(np_model, X_train, y_train, X_test, y_test)

    # 获取参数并计算fx
    X_fit = X_train[np_model.support_]
    y_fit = y_train[np_model.support_]
    params = np_model.get_params()
    if 'gamma' in params:
        params['gamma'] = 1 / (n_features * X_train.var()) if params['gamma'] == 'scale' else params['gamma']
    coef = np_model.dual_coef_
    intercept = np_model.intercept_

    km = cal_km(params, X_fit, X_test[:10], type='realize')
    fx = cal_fx(km, coef, intercept)
    print('\nfx-realize:')
    print(fx)
    print('\ndecision_function:')
    print(np_model.decision_function(X_test[:10]))
    print('\ny_pred')
    print(np_model.predict(X_test[:10]))

    # -------------------
    # Pre-processing
    # -------------------

    print('\n...Pre-processing')
    # 根据coef对X_fit等升序排序
    sorted_idx = np.argsort(coef[0])
    sorted_coef = coef[0][sorted_idx].reshape(-1, 1)
    sorted_X_fit = X_fit[sorted_idx]
    sorted_y_fit = y_fit[sorted_idx]

    # 舍弃低权重的instance
    fit_dataloaders, n_fit = get_fit_dataloaders(sorted_X_fit, sorted_coef, sorted_y_fit, _config.labels, map_size)

    # 展示原分布
    print('\ndata distribution compare:')
    print('train data - num:{}, distribution:{}'.format(X_train.shape[0], sorted(Counter(y_train).items())))
    print('fit data - num:{}, distribution:{}'.format(X_fit.shape[0], sorted(Counter(y_fit).items())))

    # -------------------
    # Compressing && Inference
    # -------------------

    for rate in rate_list:
        start = time.time()

        # 计算每个map压缩的数量
        n_compressed = X_fit.shape[0] * (1 - rate)
        rate = n_compressed / n_fit
        n_generate = int(rate * map_size)

        # 初始化压缩表示
        X_compressed = None
        y_compressed = None
        coef_compressed = None
        is_none = True

        # 对于所有label
        for label in labels:
            fit_dataloader = fit_dataloaders[label]
            # 对于每个batch
            for data in fit_dataloader:
                X_map, coef_map = data
                coef_map = coef_map.view(1, -1)

                #初始化网络
                conv_module = CNN_Res18(Basic_Block, blocks, n_generate)
                compression_net = CompressionNet(conv_module, X_map, label, lr, n_generate, params)

                #     for name, param in compression_net.named_parameters():
                #         print(name)
                #     break
                # break

                # 初始化数据
                gdm = GeneralDataModule(n_features, n_classes, labels, batch_size=64)
                gdm.prepare_data(X_train, X_test, params, X_map[0][0].numpy(), coef_map.numpy(), label)

                #训练
                trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=n_epochs)
                trainer.fit(compression_net, gdm)  # 单轮测试

                #测试
                trainer.test(datamodule=gdm)

                # 保存压缩向量
                X_compressed_partial = compression_net.forward().cpu().detach().numpy()
                y_compressed_partial = np.full((X_compressed_partial.shape[0], ), label)
                alpha_i_compressed_partial = compression_net.get_alpha_i().cpu().detach().numpy()
                label_i = 1 if label > 0 else -1
                coef_compressed_partial = alpha_i_compressed_partial * label_i

                if is_none:
                    X_compressed = X_compressed_partial
                    coef_compressed = coef_compressed_partial
                    y_compressed = y_compressed_partial
                    is_none = False
                else:
                    X_compressed = np.concatenate((X_compressed, X_compressed_partial), axis=0)
                    coef_compressed = np.concatenate((coef_compressed, coef_compressed_partial), axis=0)
                    y_compressed = np.concatenate((y_compressed, y_compressed_partial), axis=0)

        end = time.time()
        print('\nTraining time used:{:.2f}'.format(end - start))
        print('-------------------')

        # 使用coef_compressed 和 X_compressed 来计算fx
        start = time.time()
        print('\n eval compressed ins')
        n_true = 0
        for i in range(X_test.shape[0]):
            km_compressed = cal_km(params, X_compressed, X_test[i].reshape(1, -1), type='interface')
            fx_compressed = cal_fx(km_compressed, coef_compressed, intercept=0)
            pred_compressed = 1 if fx_compressed > 0 else 0
            y_pred = np_model.predict(X_test[i].reshape(1, -1))
            if pred_compressed == y_pred:
                n_true = n_true + 1

            if i < 10:
                print('\nThe {}-th ins:'.format(i))
                print('fx_compressed:{}'.format(fx_compressed))
                print('fx_np_model:{}'.format(np_model.decision_function(X_test[i].reshape(1, -1))))
                print('pred_comressed:{}'.format(pred_compressed))
                print('pred_np_model:{}'.format(y_pred))

        end = time.time()
        print('\nTraining time used:{:.2f}'.format(end - start))
        print('On compression rate {}, Acc:{:.2f}'.format(rate, n_true / X_test.shape[0]))
        print('-------------------------')

#%%
print(torch.cuda.is_available())
# %%
