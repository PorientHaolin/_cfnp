from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file
import torch.utils.data as Data
import torch
import time


def multi_to_binary(y: np.ndarray) -> np.ndarray:
    label = sorted(Counter(y).items(), reverse=True)[0][0]
    y[y != label] = -1.0
    return y


def load_libsvm_data(train_path: str, n_features: int, test_path: str = None, test_size: float = 0.3, is_multi: bool = False):
    encoder = LabelEncoder()

    X_train, y_train = load_svmlight_file(train_path, n_features=n_features)
    X_train = X_train.toarray()
    if is_multi:
        y_train = multi_to_binary(y_train)
    y_train = encoder.fit_transform(y_train)

    if not test_path:
        # test_path 为空只有一个文件要读取，且要切分
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    else:
        X_test, y_test = load_svmlight_file(test_path, n_features=n_features)
        X_test = X_test.toarray()
        if is_multi:
            y_test = multi_to_binary(y_test)
        y_test = encoder.fit_transform(y_test)

    return X_train, y_train, X_test, y_test


def get_libsvm_data(name, n_features, is_multi, has_test, show=True):
    if is_multi:
        train_path = 'datasets/libsvm/multiple/' + name
        if has_test:
            test_path = 'datasets/libsvm/multiple/' + name + '.t'
        else:
            test_path = None
    else:
        train_path = 'datasets/libsvm/binary/' + name
        if has_test:
            test_path = 'datasets/libsvm/binary/' + name + '.t'
        else:
            test_path = None

    try:
        start = time.time()
        X_train, y_train, X_test, y_test = load_libsvm_data(train_path, n_features, test_path, is_multi=is_multi)
        end = time.time()
        load_time = end - start
    except (Exception):
        print('dataset {} load exception!'.format(name))
        print('Exception Info:')
        print(Exception)
        raise Exception
    else:
        if show:
            print('Loading Finish: Time used {:.2f}'.format(load_time))
            print('---Dataset Info---')
            print('dataset name:{}'.format(name))
            print('n_features:{}'.format(n_features))
            print('is_multi:{}'.format(is_multi))
            print('Trainset:{}'.format(sorted(Counter(y_train).items())))
            print('Testset:{}'.format(sorted(Counter(y_test).items())))

        return X_train, y_train, X_test, y_test


def get_fit_dataloaders(X_fit, coef, y_fit, labels, map_size, batch_size=1):
    # 按类别构建Dataloader
    dataloaders = []
    n_fit = 0
    for label in labels:
        print('----label {}----'.format(label))
        #按类筛选
        X_temp = X_fit[y_fit == label][:]
        coef_temp = coef[y_fit == label]
        print("before X_temp_shape:", X_temp.shape)
        print('before coef_temp_shape:', coef_temp.shape)
        #计算余数
        n_delete = X_temp.shape[0] % map_size
        #舍去余数部分以便均等分割
        X_temp = X_temp[n_delete:]
        coef_temp = coef_temp[n_delete:]
        n_fit += X_temp.shape[0]
        print("after X_temp_shape:", X_temp.shape)
        print('after coef_shape:', coef.shape)
        #计算分段数
        n_parts = X_temp.shape[0] / map_size
        print('n_parts:', n_parts)
        #均等分割
        X_temp_arr = np.split(X_temp, int(n_parts), axis=0)  # 数组拆分
        coef_temp_arr = np.split(coef_temp, int(n_parts), axis=0)
        print("{} map for each {} item,total {}".format(len(X_temp_arr), map_size, len(X_temp_arr) * map_size))
        for i in range(len(X_temp_arr)):
            X_temp_arr[i] = np.expand_dims(X_temp_arr[i], axis=0)

        X_temp = np.array(X_temp_arr)
        coef_temp = np.array(coef_temp_arr)
        print('batch X_temp_shape:', X_temp.shape)
        print('batch coef_temp_shape:', coef_temp.shape)

        tensor_X_temp = torch.Tensor(X_temp)
        tensor_coef_temp = torch.Tensor(coef_temp)
        dataset_temp = Data.TensorDataset(tensor_X_temp, tensor_coef_temp)
        dataloader_temp = Data.DataLoader(dataset=dataset_temp, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloaders.append(dataloader_temp)
    return dataloaders, n_fit
