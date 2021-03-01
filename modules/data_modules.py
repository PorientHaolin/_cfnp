from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import torch.utils.data as Data
import torch
from typing import Optional
import numpy as np


class GeneralDataModule(pl.LightningDataModule):
    def __init__(self, n_features, n_classes, labels, batch_size):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.labels = labels
        self.batch_size = batch_size

    def prepare_data(self, X_train, X_test, params, X_fit, coef, label, intercept=0) -> None:
        # prepare train data
        tensor_X = torch.Tensor(X_train)
        fx_train = self.cal_fx(params, X_fit, X_train, coef, intercept)
        tensor_fx = torch.Tensor(fx_train)
        train_dataset = Data.TensorDataset(tensor_X, tensor_fx)
        train_size = int(0.75 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.trainset, self.valset = random_split(train_dataset, [train_size, val_size])

        # prepare test data
        tensor_X = torch.Tensor(X_test)
        fx_test = self.cal_fx(params, X_fit, X_test, coef, intercept)
        tensor_fx = torch.Tensor(fx_test)
        self.testset = Data.TensorDataset(tensor_X, tensor_fx)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.testset, batch_size=self.batch_size)

    def cal_fx(self, params, X_fit, X, coef, intercept):
        if params['kernel'] == 'linear':
            kernel_metrics = linear_kernel(X_fit, X)
        elif params['kernel'] == 'rbf':
            kernel_metrics = rbf_kernel(X_fit, X, gamma=params['gamma'])
        elif params['kernel'] == 'poly':
            kernel_metrics = polynomial_kernel(X_fit, X, gamma=params['gamma'], coef0=0.0)
        elif params['kernel'] == 'sigmoid':
            kernel_metrics = sigmoid_kernel(X_fit, X, gamma=params['gamma'], coef0=0.0)
        else:
            print('Unknown kernel')

        fx = np.sum(coef * kernel_metrics.T, axis=1) + intercept
        return fx

