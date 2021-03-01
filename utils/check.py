import sys
sys.path.append("..")
import _config

from utils.data_utils import get_libsvm_data, get_fit_dataloaders
from torch.autograd import Variable
from collections import Counter
from modules.np_modules import construct_np_model, train_np_model, eval_np_model
from utils.modules_utils import cal_fx, cal_km
import numpy as np
import torch
from modules.conv_modules import Basic_Block, CNN_Res18
from modules.data_modules import GeneralDataModule


def check_datasets():
    # Datasets Check
    traverse_list = ['a9a', 'ijcnn1', 'madelon', 'mushrooms', 'phishing', 'splice', 'w8a', 'dna.scale', 'mnist', 'pendigits', 'Sensorless', 'usps']
    print('{:^8}\t{:^8}\t{:^8}\t{:^28}\t{:^28}'.format('name', 'n_features', 'is_multi', 'trainset', 'testset'))

    for name in traverse_list:

        n_features = _config.datasets[name]['n_features']
        is_multi = _config.datasets[name]['is_multi']
        has_test = _config.datasets[name]['has_test']

        try:
            X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test, show=False)
        except (Exception):
            continue
        else:
            print('{:^8}\t{:^8}\t{:^8}\t{:^28}\t{:^28}'.format(name, n_features, is_multi, '{}'.format(sorted(Counter(y_train).items())),
                                                               '{}'.format(sorted(Counter(y_test).items()))))


def check_np_traning():
    traverse_list = ['a9a']
    for name in traverse_list:
        n_features = _config.datasets[name]['n_features']
        is_multi = _config.datasets[name]['is_multi']
        has_test = _config.datasets[name]['has_test']

        try:
            X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test)
        except (Exception):
            continue

        # -------------------
        # Train np model
        # -------------------

        # svc
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        for kernel in kernels:
            print('---{} svc---'.format(kernel))
            svc = construct_np_model(model_type='svc', kernel=kernel)
            svc = train_np_model(svc, X_train, y_train)
            eval_np_model(svc, X_train, y_train, X_test, y_test)
            alpha_i = np.abs(svc.dual_coef_[0])
            print('max_alpha_i:{}'.format(np.max(alpha_i)))
            print('min_alpha_i:{}'.format(np.min(alpha_i)))

        # todo: svr

        # todo: krr

        # todo: klr


def check_kernel(kernel, dataset_name):

    name = dataset_name
    n_features = _config.datasets[name]['n_features']
    is_multi = _config.datasets[name]['is_multi']
    has_test = _config.datasets[name]['has_test']

    X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test, show=False)

    svc = construct_np_model(model_type='svc', kernel=kernel)
    svc = train_np_model(svc, X_train, y_train)

    X_fit = X_train[svc.support_]
    coef = svc.dual_coef_
    intercept = svc.intercept_
    params = svc.get_params()
    if 'gamma' in params:
        params['gamma'] = 1 / (n_features * X_train.var()) if params['gamma'] == 'scale' else params['gamma']

    print('----show kernel metrics---')
    km_interface = cal_km(params, X_fit, X_train[:10], type='interface')
    km_realize = cal_km(params, X_fit, X_train[:10], type='realize')

    print('\n sklearn:')
    print(km_interface)
    print(km_interface.shape)

    print('\n realize:')
    print(km_realize)
    print(km_realize.shape)
    print('')

    print('---show hand-10 fx ---')
    fx_interface = cal_fx(km_interface, coef, intercept)
    fx_realize = cal_fx(km_realize, coef, intercept)
    print('\n fx_interface:')
    print(fx_interface)
    print('\n fx_realize:')
    print(fx_realize)
    print('\n decsion function:')
    print(svc.decision_function(X_train[:10]))
    print('\n y_pred:')
    print(svc.predict(X_train[:10]))
    print('')

    print('---show composite fx---')
    km1 = cal_km(params, X_fit[:5000], X_train[:10], type='realize')
    fx1 = cal_fx(km1, coef[:, :5000], 0)
    km2 = cal_km(params, X_fit[5000:], X_train[:10], type='realize')
    fx2 = cal_fx(km2, coef[:, 5000:], 0)
    print('\n fx1-5000 ins')
    print(fx1)
    print('\n fx2-others')
    print(fx2)
    print('\n batch fx = fx1 + fx2 + intercept')
    print(fx1 + fx2 + intercept)
    print('')


def conv_check(kernel, dataset_name):
    # 训练svc
    name = dataset_name
    n_features = _config.datasets[name]['n_features']
    is_multi = _config.datasets[name]['is_multi']
    has_test = _config.datasets[name]['has_test']

    X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test, show=False)

    svc = construct_np_model(model_type='svc', kernel=kernel)
    svc = train_np_model(svc, X_train, y_train)

    X_fit = X_train[svc.support_]
    y_fit = y_train[svc.support_]
    coef = svc.dual_coef_
    intercept = svc.intercept_
    params = svc.get_params()
    if 'gamma' in params:
        params['gamma'] = 1 / (n_features * X_train.var()) if params['gamma'] == 'scale' else params['gamma']

    # 数据分类别分批次输入
    map_size = 128
    n_blocks = [2, 2, 2, 2]

    # 根据coef对X_fit等升序排序
    sorted_idx = np.argsort(coef[0])
    sorted_coef = coef[0][sorted_idx].reshape(-1, 1)
    sorted_X_fit = X_fit[sorted_idx]
    sorted_y_fit = y_fit[sorted_idx]

    # 舍弃低权重的instance
    fit_dataloaders, n_fit = get_fit_dataloaders(sorted_X_fit, sorted_coef, sorted_y_fit, _config.labels, map_size)

    # 展示原分布
    print('----data distribution----')
    print('train data - num:{}, distribution:{}'.format(X_train.shape[0], sorted(Counter(y_train))))
    print('fit data - num:{}, distribution:{}'.format(X_fit.shape[0], sorted(Counter(y_fit))))
    print(' ')
    # 确定压缩率(如果map_size过小，90%会报错)
    rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for rate in rate_list:
        # 计算每个map压缩的数量
        n_compressed = X_fit.shape[0] * (1 - rate)
        rate = n_compressed / n_fit
        n_generate = int(rate * map_size)
        conv_module = CNN_Res18(Basic_Block, n_blocks, n_generate)
        if torch.cuda.is_available():
            conv_module = conv_module.cuda()

        X_gen = None
        y_gen = None
        is_none = True
        for label in _config.labels:
            fit_dataloader = fit_dataloaders[label]
            for data in fit_dataloader:
                X_map, coef_map = data
                X_map = Variable(X_map)
                if torch.cuda.is_available():
                    X_map = X_map.cuda()
                X_gen_temp = conv_module(X_map)
                y_gen_temp = np.full((X_gen_temp.shape[0], ), label)
                if is_none:
                    print(X_gen_temp.size())
                    X_gen = X_gen_temp.cpu().detach().numpy()
                    y_gen = y_gen_temp
                    is_none = False
                else:
                    X_gen = np.concatenate((X_gen, X_gen_temp.cpu().detach().numpy()), axis=0)
                    y_gen = np.concatenate((y_gen, y_gen_temp), axis=0)
        #展示压缩后分布
        print('{:.2f}% data - num:{}, distribution:{}'.format(rate * 100, X_gen.shape[0], sorted(Counter(y_gen).items())))

    # for all comp-rate
    # for all map
    # res-18
    # res + transformer


def datamodule_check(kernel, dataset_name):
    # 训练svc
    name = dataset_name
    n_features = _config.datasets[name]['n_features']
    is_multi = _config.datasets[name]['is_multi']
    has_test = _config.datasets[name]['has_test']

    X_train, y_train, X_test, y_test = get_libsvm_data(name, n_features, is_multi, has_test, show=False)

    svc = construct_np_model(model_type='svc', kernel=kernel)
    svc = train_np_model(svc, X_train, y_train)

    X_fit = X_train[svc.support_]
    y_fit = y_train[svc.support_]
    coef = svc.dual_coef_
    intercept = svc.intercept_
    params = svc.get_params()
    if 'gamma' in params:
        params['gamma'] = 1 / (n_features * X_train.var()) if params['gamma'] == 'scale' else params['gamma']

    # 数据分类别分批次输入
    map_size = 128

    # 根据coef对X_fit等升序排序
    sorted_idx = np.argsort(coef[0])
    sorted_coef = coef[0][sorted_idx].reshape(-1, 1)
    sorted_X_fit = X_fit[sorted_idx]
    sorted_y_fit = y_fit[sorted_idx]

    # 舍弃低权重的instance
    fit_dataloaders, n_fit = get_fit_dataloaders(sorted_X_fit, sorted_coef, sorted_y_fit, _config.labels, map_size)

    for label in _config.labels:
        fit_dataloader = fit_dataloaders[label]
    for data in fit_dataloader:
        X_map, coef_map = data
        coef_map = coef_map.view(1, -1)
        gdm = GeneralDataModule(_config.datasets[name]['n_features'], _config.n_classes, _config.labels, batch_size=64)
        gdm.setup(X_train, X_test, params, X_map[0][0].numpy(), coef_map.numpy(), label)

        train_loader = gdm.train_dataloader()
        val_loader = gdm.val_dataloader()
        test_loader = gdm.test_dataloader()

        print('train data')
        for train_data in train_loader:
            X_batch, fx_batch = train_data
            print('X size:', X_batch.size())
            print('fx size:', fx_batch.size())
            print(fx_batch)
            break

        print('val data')
        for val_data in val_loader:
            X_batch, fx_batch = val_data
            print(X_batch.size())
            print('fx size:', fx_batch.size())
            print(fx_batch)
            break

        print('test data')
        for test_data in test_loader:
            X_batch, fx_batch = test_data
            print(X_batch.size())
            print('fx size:', fx_batch.size())
            print(fx_batch)
            break
        break
    # gdm
    # setup
