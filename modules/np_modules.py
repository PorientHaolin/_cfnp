from cuml.svm import SVC
# from typing import Tuple
import time
from typing import Tuple
from sklearn.metrics import classification_report


def construct_svc(**kwargs) -> SVC:

    kernel = kwargs['kernel'] if 'kernel' in kwargs else 'linear'

    if kernel == 'linear':
        return SVC(kernel=kernel)
    elif kernel in ['rbf', 'poly', 'sigmoid']:
        C = kwargs['C'] if 'C' in kwargs else 1
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 'scale'
        return SVC(kernel=kernel, C=C, gamma=gamma)
    else:
        print('Error: Unknown kernel')
        return


def construct_np_model(model_type: str, **kwargs):
    if model_type == 'svc':
        return construct_svc(**kwargs)
    elif model_type == 'svr':
        pass
    elif model_type == 'krr':
        # kernel ridge regression
        pass
    elif model_type == 'klr':
        # kernel logistic regression
        pass


def train_np_model(model, X_train, y_train) -> None:
    # train
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start

    print('Training Finish: Time used {:.2f}'.format(train_time))
    return model


def eval_avg_score(X, y_true, y_predict) -> Tuple[float, float, float, float]:
    # 评估各种数值指标
    report_dict = classification_report(y_true, y_predict, digits=4, output_dict=True)
    # 准确率：(TP+TN)/(TP+TN+FP+FN)
    accuracy = report_dict['accuracy']
    # 精确率：TP/(TP+FP)，不将负样本标记为正样本的能力
    macro_avg_precision = report_dict['macro avg']['precision']
    # 召回率：TP/(TP+FN)，找到所有正样本的能力
    recall = report_dict['macro avg']['recall']
    # F1 Score，越高模型越稳健
    f1_score = report_dict['macro avg']['f1-score']

    return accuracy, macro_avg_precision, recall, f1_score


def eval_np_model(model, X_train, y_train, X_test, y_test) -> None:
    # Inference
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Eval
    print('{0:<10}\t{1:<10}\t{2:<10}\t{3:<10}\t{4:<10}'.format('dataset', 'acc', 'precision', 'recall', 'f1-score'))
    acc, precision, recall, f1 = eval_avg_score(X_train, y_train, pred_train)
    print('{0:<10}\t{1:<10.2f}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format('trainset', acc, precision, recall, f1))
    acc, precision, recall, f1 = eval_avg_score(X_test, y_test, pred_test)
    print('{0:<10}\t{1:<10.2f}\t{2:<10.2f}\t{3:<10.2f}\t{4:<10.2f}'.format('testset', acc, precision, recall, f1))
    return
