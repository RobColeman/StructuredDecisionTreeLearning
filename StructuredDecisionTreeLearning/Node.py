from collections import namedtuple
from copy import deepcopy
import numpy as np
import os, json
import scipy as sp
import scipy.stats as stats
from sklearn import metrics
from pandas import DataFrame, read_csv, Int64Index
import pandas as pd
from sklearn import tree
import numpy as np
import os, sys
from StructuredDecisionTreeLearning.constants import *
from StructuredDecisionTreeLearning.math_helpers import *
from StructuredDecisionTreeLearning.gen_helpers import *


class Node():
    """
    Node data structure, for defining tree structures
    """

    def __init__(
        self,
        key,        
        is_leaf, 
        training_label_column = UHRS_LABELS_COLUMN, 
        training_feature_column = None, 
        training_weight_column = None,
        prod_output_value = None, 
        prod_feature_name = None, 
        split_value = None, 
        split_metric = "gini",
        parent = None,
        left = None, 
        right = None
        ):

        self.key = key
        self.is_leaf = is_leaf
        self.parent = parent
        self.training_label_column = training_label_column
        self.training_feature_column = training_feature_column
        self.training_weight_column = training_weight_column
        self.prod_output_value = prod_output_value
        self.prod_feature_name = prod_feature_name
        self.split_value = split_value
        self.split_metric = split_metric
        self.left = left
        self.right = right


    def set_left(self, node):
        self.left = node
    
        
    def set_right(self, node):
        self.right = node


def predefined_decision_node_constructor(
    key, 
    prod_output_value, 
    prod_feature_name = 'Darwin_Decision', 
    training_label_column = UHRS_LABELS_COLUMN, 
    training_weight_column = None, 
    parent = None):
    node = Node(key = key, 
                parent = parent,
                is_leaf = False, 
                training_label_column = training_label_column,
                training_feature_column = None,
                training_weight_column = training_weight_column, 
                prod_feature_name = prod_feature_name,
                prod_output_value = prod_output_value,
                split_value = None,
                split_metric = "gini",
                left = None,
                right = None)                
    return node


def internal_node_constructor(
        key, 
        prod_feature_name, 
        training_feature_column, 
        training_label_column = UHRS_LABELS_COLUMN, 
        training_weight_column = None, 
        parent = None,
        split_value = None, # predefine split or learn it
        split_metric = "gini",
        left = None, 
        right = None
    ):
    node = Node(key = key, 
                parent = parent,
                is_leaf = False, 
                training_label_column = training_label_column,
                training_feature_column = training_feature_column,
                training_weight_column = training_weight_column, 
                prod_feature_name = prod_feature_name,
                prod_output_value = None,
                split_value = split_value,
                split_metric = split_metric,
                left = left,
                right = right)
    return node
