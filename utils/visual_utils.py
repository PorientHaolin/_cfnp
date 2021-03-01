from typing import Tuple


def draw_binary_pr(model, X, y, title):
    # 绘制二分类的pr曲线
    y_score = model.decision_function(X)
    average_precision = average_precision_score(y, y_score)
    precision, recall, _ = precision_recall_curve(y, y_score)

    plt.figure()
    plt.plot(recall, precision, color='tomato', lw=2, label='PR curve(area={:0.2f})'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{0} P-R:AP={1:0.2f}'.format(title, average_precision))
    plt.show()
    # disp = plot_precision_recall_curve(model,X,y)
    # disp.ax_.set_title('{0} P-R:AP={1:0.2f}'.format(title,average_precision))
    return


def draw_binary_roc(model, X, y, title):
    # 绘制二分类的roc曲线，当正负样本大致均匀时，roc作为性能指标更鲁棒，在样本分布发生变化时roc曲线也能保持文档
    y_score = model.decision_function(X)
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve(area={:0.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC'.format(title))
    plt.legend(loc="lower right")
    plt.show()
    return


def showPlot(points, title):
    plt.figure()
    fig, ax = plt.subplots()
    #loc = ticker.MultipleLocator(base=1)
    # ax.xaxis.set_major_locator
    plt.plot(points)
    plt.title(title)
    plt.show()


def showComparePlot(points1, points2, title):
    plt.figure()
    #todo
