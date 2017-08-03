import os, sys, json
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from numpy.linalg import det, inv
import scipy as sp
import scipy.stats as stats
from sklearn import metrics
from pandas import DataFrame, read_csv, Int64Index
import pandas as pd
import random
from StructuredDecisionTreeLearning.constants import *
from StructuredDecisionTreeLearning.math_helpers import *



def sample_tsv_write_tsv(full_data_path, output_path, n = 5000):
    df = DataFrame.from_csv(full_data_path, sep='\t')
    sample = df.sample(n)
    sample.to_csv(output_path, sep='\t', encoding='utf-8')


def random_split_data_frame(df, right_perc):
    n_right = int(right_perc*len(df))
    right_idx = random.sample(list(df.index), n_right)
    left_df =  df.drop(right_idx)
    right_df = df.ix[right_idx]
    return (left_df, right_df)


def write_string_to_file(string_to_write, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    f = open(file_path, "w+")
    f.write(string_to_write)
    f.close()


def read_training_data_as_data_frame(file_path):
    df = DataFrame.from_csv(file_path, sep='\t')
    df["m:index"] = Int64Index(list(range(len(df))))
    return df.set_index("m:index")


def create_working_dir(working_dir):
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    return working_dir


def write_metric_results(results, output_path):
    output_path += "parameter_performance.json"
    f = open(output_path, 'w+')
    f.write(json.dumps(results, indent=4, sort_keys=True))
    f.close()


def write_data_frame_to_file(df, output_path):
    output_path += "predictions.tsv"
    df.to_csv(output_path, sep='\t', encoding='utf-8')

def sort_dictionary(d):
    return OrderedDict(sorted(d.items()))


def to_json_string(d, pretty = False):
    if pretty:
        s = json.dumps(d, indent=4, sort_keys=True)
    else:
        s = json.dumps(d)
    return s


def print_to_json(d, pretty = False):
    print(to_json_string(d, pretty))


def write_to_json(d, output_path, pretty = False):
    f = open(output_path, 'w')
    f.write(to_json_string(d, pretty))
    f.close()
