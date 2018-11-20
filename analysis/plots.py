from __future__ import absolute_import, division, print_function

import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import f1_score, auc

# set style
sns.set(style='whitegrid')
sns.set_context('notebook')


def plot_histogram(data, color='b', alpha=1.,
                   ax=ax, norm=False, kde=False,
                   rug=False, label=None):
    '''
        Plot the histogram of the provided data
        params:
            data : data to plot the histogram of
            color : hue of the plot
            alpha : transparency of the plot \in [0,1]
            ax : axis on which to plot
            norm : (True/False) whether to normalize the data
            kde : (True/False) whether kde-plot is required or not
            rug : (True/False) whether to show the rug-plot at hist-base
            label : label for the plot
    '''
    # 1. calculate bins
    bins = np.arange(np.min(data), np.max(data))
    # 2. plot
    sns.distplot(data,
                 color=color,
                 bins=bins,
                 norm_hist=norm,
                 kde=kde,
                 rug=rug,
                 hist_kws={'alpha':alpha},
                 label=label,
                 ax=ax)


def plot_confusion_matrix(y_test, y_pred,
                          cmap='coolwarm_r',
                          normalize=False,
                          report=False, ax=ax):
    '''
        Plot the confusion matrix and print reports
        params:
            y_test : test labels
            y_pred : predictions
            cmap : colour map to use
            normalize : whether to normalize the data
            report : whether to cerate report
            ax : axis on which to plot
    '''
    # 1. get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # 2. do we need normalization?
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    # 3. build the confusion matrix heatmap
    sns.heatmap(cm, cmap=cmap, annot=True, ax=ax)


def roc_pr_f1(y_test, y_pred):
    '''
        Get the PR, F1 and ROC-AUC data.
        params:
            y_test : test labels
            y_pred : predicted labels
    '''
    # 1. precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    # 2. roc curve
    tpr, fpr, _ = roc_curve(y_test, y_pred)
    auc = auc(fpr, tpr)
    # 3. f1 score
    f1_score = f1_score(y_test, y_pred)
    # 4. report
    class_report = classification_report(y_test, y_pred)

    return precision, recall, tpr, fpr, auc, f1_score, class_report
