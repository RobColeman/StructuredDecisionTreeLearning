import os, sys, json
import numpy as np
from random import randint
from copy import deepcopy
from numpy.linalg import det, inv
import scipy as sp
import scipy.stats as stats
from sklearn import metrics
from pandas import DataFrame, read_csv, Int64Index, Series
import pandas as pd
from StructuredDecisionTreeLearning.constants import *
#from StructuredDecisionTreeLearning.gen_helpers import *


def add_random_predicted_labels(df, prediction_column):
    df[prediction_column] = Series([randint(0,1) for x in range(len(df))], index = df.index)
    return df


def compute_net_gain(df, apsat_prediction_column, background_apsat_prediction_column, predicted_apsat_netgain_column):
    netgain = df[apsat_prediction_column] - df[background_apsat_prediction_column]
    df[predicted_apsat_netgain_column] = df[apsat_prediction_column] - df[background_apsat_prediction_column]
    return df


def unpack_cm(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    return ( float(tn), float(fp), float(tn), float(tp) )


def mcc(cm):
    tn, fp, fn, tp = unpack_cm(cm)
    num = (tp*tn)-(fp*fn)
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if denom == 0:
        denom = 1.0
    else:
        denom = float(np.sqrt(denom))

    return float(num / denom)


def precision(cm):
    tn, fp, fn, tp = unpack_cm(cm)
    if (tp + fp) == 0:
        return 0.0
    else:
        return float(tp / (tp + fp))


def recall(cm):
    tn, fp, fn, tp = unpack_cm(cm)
    if (tp + fn) == 0:
        return 0.0
    else:
        return float(tp / (tp + fn))


def f1_score(cm, beta = 1.0):
    tn, fp, fn, tp = unpack_cm(cm)
    p = precision(cm)
    r = recall(cm)
    b2 = beta**2
    if ((b2 * p) + r) == 0:
        return 0.0
    else:
        return float((1 + b2) * (p * r) / ((b2 * p) + r))


def accuracy(cm):
    tn, fp, fn, tp = unpack_cm(cm)
    if (tn + fp + fn + tp) == 0:
        return 0
    else:
        return float((tn + tp) / (tn + fp + fn + tp))


def avg_pos_netgain(df, prediction_column, weight_column, predicted_apsat_netgain_column):
    p_labels = df[prediction_column]
    pos_df = df.ix[p_labels[p_labels==1].index]
    predicted_pos_netgains = pos_df[predicted_apsat_netgain_column]

    if weight_column == None:
        weight_series = Series(list(np.ones(len(pos_df))), index = pos_df.index)
    else:
        weight_series = deepcopy(pos_df[weight_column])
        weight_series[weight_series[weight_series==0].index] = 1.0

    if len(predicted_pos_netgains) > 0:
        return float(predicted_pos_netgains.dot(weight_series) / float(len(predicted_pos_netgains)))
    else:
        return 0.0

def compute_gini(class_counts):
    n = float(sum(class_counts))
    if n > 0:
        fi2s = [(x/n)**2 for x in class_counts]
        return float(1 - sum(fi2s))
    else:
        return 0.0


def compute_avg_gini(class_counts_lt, class_counts_gte):
    n = float(sum(class_counts_lt) + sum(class_counts_gte))
    w_l = sum(class_counts_lt) / n
    w_r = sum(class_counts_gte) / n
    return (w_l * compute_gini(class_counts_lt)) + (w_r * compute_gini(class_counts_gte))


def compute_column_correlation(df, label_column, feature_column):
     return df[label_column].corr(df[feature_column])


def compute_all_feature_correlations(df, label_column, feature_columns):
    corrs = {}
    for col in feature_columns:
        corrs[col] = compute_column_correlation(df, label_column, col)
    return corrs


def print_feature_correlations(df, label_column, feature_columns):
    correlations = compute_all_feature_correlations(training_data_df, label_column, feature_columns)
    ordered = sorted([(corr, fname) for fname, corr in correlations.items()])
    ordered.reverse()

    for corr, fname in ordered:
        print("{0} : {1}".format(fname, corr))


def compute_label_counts(df, label_column):
    label_col_series = df[label_column]
    counts = [len(label_col_series[label_col_series==0]), len(label_col_series[label_col_series==1])]
    return counts


def compute_cm_metrics(conf_mat, beta = BETA):
    computed_metrics = {}
    tn, fp, fn, tp = unpack_cm(conf_mat)
    computed_metrics['confusion_matrix'] = { 'tn': tn, 'fp' : fp, 'fn' : fn, 'tp' : tp }
    computed_metrics['precision'] = precision(conf_mat)
    computed_metrics['recall'] = recall(conf_mat)
    computed_metrics['mcc'] = mcc(conf_mat)
    computed_metrics['accuracy'] = accuracy(conf_mat)
    computed_metrics['f1'] = f1_score(conf_mat, 1.0)
    computed_metrics['f1_beta']= f1_score(conf_mat, beta)
    return computed_metrics

def empty_cm_metrics():
    computed_metrics = {
        'confusion_matrix' :  { 'tn': 0, 'fp' : 0, 'fn' : 0, 'tp' : 0 },
        'precision' : 0.0,
        'recall' : 0.0,
        'mcc' : 0.0,
        'accuracy' : 0.0,
        'f1' : 0.0,
        'f1_beta' : 0.0,
        'avg_pos_net_gain' : 0.0
        }
    return computed_metrics


def safe_compute_comfusion_matrix(df, label_column, prediction_column, weight_column = None):
    if weight_column == None:
        weight_series = Series(list(np.ones(len(df[label_column]))), index = df[label_column].index)
    else:
        weight_series = deepcopy(df[weight_column])

    conf_mat = np.zeros((2,2))
    for l, p, w in zip(df[label_column], df[prediction_column], weight_series):
        if w == 0:
            w = 1
        if l == 0 and p == 0:
            conf_mat[0,0] += w
        elif l == 0 and p == 1:
            conf_mat[0,1] += w
        elif l == 1 and p == 0:
            conf_mat[1,0] += w
        elif l == 1 and p == 1:
            conf_mat[1,1] += w
        else:
            print("""Error in safe_compute_comfusion_matrix. l: {0}, p : {1}""".format(l, p))

    return conf_mat


def compute_darwin_perf_metrics(df, label_column, prediction_column, weight_column = None, netgain_column = None, beta = BETA):
    conf_mat = safe_compute_comfusion_matrix(df, label_column, prediction_column, weight_column)
    computed_metrics = compute_cm_metrics(conf_mat, beta)
    if netgain_column:
        computed_metrics['avg_pos_net_gain'] = avg_pos_netgain(df, prediction_column, weight_column, netgain_column)
    else:
        computed_metrics['avg_pos_net_gain'] = 0.0
    return computed_metrics


def compute_perf_metrics(df, label_column, prediction_column, weight_column = None, beta = BETA):
    conf_mat = safe_compute_comfusion_matrix(df, label_column, prediction_column, weight_column)
    computed_metrics = compute_cm_metrics(conf_mat, beta)
    return computed_metrics

def format_cm_metrics(computed_metrics):
    output = """confusion_matrix: {0}, precision : {1}, recall {2}, mcc : {3}, accuracy : {4}, f1 : {5}, f1_beta : {6}, avg_pos_net_gain : {7}""".format(
            computed_metrics['confusion_matrix'],
            computed_metrics['precision'],
            computed_metrics['recall'],
            computed_metrics['mcc'],
            computed_metrics['accuracy'],
            computed_metrics['f1'],
            computed_metrics['f1_beta'],
            computed_metrics['avg_pos_net_gain']
        )
    return output
