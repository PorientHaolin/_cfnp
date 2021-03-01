from pytorch_lightning.metrics import functional as FM
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook

class CompressionNet(pl.LightningModule):
    def __init__(self, conv_module, X_fit, label, lr, n_generate, params):
        super().__init__()
        self.lr = lr
        self.conv = conv_module
        # self.X_fit = X_fit
        self.label = 1 if label > 0 else -1
        self.params = params

        self.register_buffer('X_fit', X_fit)
        self.register_parameter('untreated_coef', nn.Parameter(torch.rand(1, n_generate)))
        
        self.grads = {}

    def forward(self):
        # 返回训练好的压缩表示
        return self.conv(self.X_fit)

    def training_step(self, batch, batch_idx):
        X, fx = batch
        X_compressed = self.conv(self.X_fit)
        X_compressed.register_hook(save_grad('X_compressed', self.grads))

        km = self.cal_km(self.params, X_compressed, X)
        km.retain_grad()
        km.register_hook(save_grad('km', self.grads))

        alpha_i = torch.abs(self.untreated_coef)
        alpha_i.register_hook(save_grad('alpha_i', self.grads))

        # softmax 缩放得到的alpha_i 的数量级1e-6
        # constrainted_alpha_i = F.softmax(alpha_i) 
        constrainted_alpha_i = (alpha_i - torch.min(alpha_i)) / (torch.max(alpha_i) - torch.min(alpha_i))
        constrainted_alpha_i.register_hook(save_grad('constrainted_alpha_i', self.grads))

        coef = constrainted_alpha_i * self.label
        coef.register_hook(save_grad('coef', self.grads))

        fx_hat = torch.sum(coef * km.t(), axis=1)
        fx_hat.register_hook(save_grad('fx_hat', self.grads))

        loss = F.smooth_l1_loss(fx_hat, fx)
        loss.register_hook(save_grad('loss', self.grads))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, fx = batch
        X_compressed = self.conv(self.X_fit)
        km = self.cal_km(self.params, X_compressed, X)
        alpha_i = torch.abs(self.untreated_coef)
        constrainted_alpha_i = (alpha_i - torch.min(alpha_i)) / (torch.max(alpha_i) - torch.min(alpha_i))
        coef = constrainted_alpha_i * self.label
        fx_hat = torch.sum(coef * km.t(), axis=1)
        loss = F.smooth_l1_loss(fx_hat, fx)
        var = FM.explained_variance(fx_hat, fx)
        mae = FM.mean_absolute_error(fx_hat, fx)
        mse = FM.mean_squared_error(fx_hat, fx)

        val_metrics = {'val_var': var, 'val_mae': mae, 'val_mse': mse, 'val_loss': loss}
        self.log_dict(val_metrics)
        return val_metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        test_metrics = {'test_var': metrics['val_var'], 'test_mae': metrics['val_mae'], 'test_mse': metrics['val_mse'], 'test_loss': metrics['val_loss']}
        self.log_dict(test_metrics)
        return test_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)

    def cal_km(self, params, X_fit, X):
        if params['kernel'] == 'linear':
            return torch.mm(X_fit, X.t())
        elif params['kernel'] == 'rbf':
            X_a = X.view(-1, 1, X.size(1))
            km = torch.exp(-params['gamma'] * torch.sum(torch.pow(X_fit - X_a, 2), axis=2))
            return km.t()
        elif params['kernel'] == 'poly':
            coef0 = 0.0
            degree = 3
            return torch.pow(params['gamma'] * torch.mm(X_fit, X.t()) + coef0, degree)
        elif params['kernel'] == 'sigmoid':
            coef0 = 0.0
            return torch.tanh(params['gamma'] * torch.mm(X_fit, X.t()) + coef0)
        else:
            print('Unknown kernel')
            return
    
    def get_alpha_i(self):
        # todo 扩展成使用多种约束函数的类型
        # abs 转换为 alpha_i
        alpha_i = torch.abs(self.untreated_coef)
        # 归一化限制范围
        constrainted_alpha_i = (alpha_i - torch.min(alpha_i)) / (torch.max(alpha_i) - torch.min(alpha_i))
        return constrainted_alpha_i

    def normalize(self, input_tensor):
        max = torch.max(input_tensor)
        min = torch.min(input_tensor)
        return (input_tensor - torch.min(input_tensor)) / (torch.max(input_tensor) - torch.min(input_tensor))

    def on_train_start(self):
        print('\nInitial coef:')
        print(self.untreated_coef)

        print('\nInitial alpha_i:')
        print(self.get_alpha_i())

        pass

    def on_train_end(self):
        print('\nAfter train coef:')
        print(self.untreated_coef)

        print('\nAfter train alpha_i:')
        print(self.get_alpha_i())

        for name, grad in self.grads.items():
            print('\n {} grads:'.format(name))
            print(grad)

