from collections import namedtuple
from copy import deepcopy
import numpy as np
import os, json
from collections import OrderedDict
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
from StructuredDecisionTreeLearning.Node import *


class DecisionTreeNode:
    
    def __init__(
        self,
        key, 
        parent, 
        is_leaf,
        training_label_column,
        training_feature_column = None,
        training_weight_column = None,
        prod_feature_name = None, 
        prod_output_value = None,
        positive_output_set = [1], # set of outputs that are considered the positive class
        split_value = None,
        beta = BETA,
        split_metric = 'gini',
        left = None,
        right = None
        ):

        self._key = key
        self._split_metric = split_metric

        if not left and not right:
            is_leaf = True

        self._is_leaf = is_leaf

        if not parent:
            self._is_root = True
        else:
            self._is_root = False
        self._parent = parent

        self._training_label_column = training_label_column
        self._training_feature_column = training_feature_column
        if not training_weight_column:
            self._training_weight_column = "WEIGHT"
        else:
            self._training_weight_column = training_weight_column

        self._prod_feature_name = prod_feature_name # the darwin feature name

        self._prod_output_value = prod_output_value

        self._positive_output_set = positive_output_set

        if prod_output_value:
            self._predefined_decision_node = True
        else:
            self._predefined_decision_node = False

        self._left = left
        self._right = right

        self._beta = beta
        self._split_value = split_value


    def train(
            self, 
            training_dataframe,
            train_downward = True,
            force_learn = False
        ):
        """
        train this node with a slice of a dataframe
        """

        self.check_add_weights(training_dataframe)

        if force_learn:
            self._split_value = None


        self._n = sum(training_dataframe[self._training_weight_column])

        if self._n == 0:
            self._best_split = self.get_empty_node_best_split()
            self._cm_metrics = empty_cm_metrics()
            self._label = None
            self._n_pos = 0.0
            self._n_neg = 0.0
            self._prob_class_1 = 0.0

        else:
            self._n_pos = training_dataframe[self._training_label_column].dot(training_dataframe[self._training_weight_column])
            self._n_neg = float(self._n - self._n_pos)
            self._prob_class_1 = self._n_pos / float(self._n)

            if self._training_label_column:
                node_label = self.compute_label_from_data(training_dataframe)
        
            if self._predefined_decision_node:
                node_label = self.compute_label_from_predefined_output()
            
            self._label = node_label
            self._cm_metrics = self.compute_metrics(training_dataframe, self._label)

            if self._split_value:
                # debug
                #print("Node {0} split predefined as {1}, skipping training.".format(self._key, self._split_value))
                self.train_predefined_split(training_dataframe)
            elif self._is_leaf:
                self._best_split = self.get_empty_node_best_split()
            else:
                self.find_split(training_dataframe)
                
        # debug
        #print("Node {0} training complete.".format(self._key))

        if train_downward:
            self.train_down_children(training_dataframe)


    def train_predefined_split(self, training_dataframe):
        training_dataframe_sorted = training_dataframe.sort_values(by=self._training_feature_column)
        best_split = self.get_empty_node_best_split()
        best_split['split_value'] = self._split_value
        s_values = training_dataframe_sorted[self._training_feature_column]
        best_split['gte_indices'] = s_values[s_values >= best_split['split_value']].index
        best_split['lt_indices'] = s_values[s_values < best_split['split_value']].index
        self._best_split = best_split


    def check_add_weights(self, training_dataframe):
        if not self._training_weight_column in list(training_dataframe.columns):
            training_dataframe[self._training_weight_column] = Series(np.ones(len(training_dataframe)), index = training_dataframe.index)


    def predict(self, row):
        if self._is_leaf:
            return self._prod_output_value
        else:
            if row[self._training_feature_column] < self._split_value:
                return self._left.predict(row)
            else:
                return self._right.predict(row)
                
    
    def train_down_children(self, training_dataframe):
        """
        left and right node label column and feature column must already be set
        """
        if self._left:
            self._left.train(deepcopy(training_dataframe.loc[self._best_split['lt_indices'],:]))
        if self._right:
            self._right.train(deepcopy(training_dataframe.loc[self._best_split['gte_indices'],:]))


    def get_empty_node_best_split(self):
        emp_bs = {
                    'split_value' : None,
                    'split_type' : 'gte',
                    'class_counts_lt' : [0, 0],
                    'class_counts_gte' : [0, 0],
                    'split_score' : 0.0,
                    'gte_indices' : [],
                    'lt_indices' : []
                }
        return emp_bs

    def set_left(self, treeNode):
        self._left = treeNode
        self._is_leaf = False


    def set_right(self, treeNode):
        self._right = treeNode
        self._is_leaf = False


    def compute_label_from_predefined_output(self):
        if self._prod_output_value in self._positive_output_set:
            output_label = 1.0
        else:
            output_label = 0.0
        return output_label


    def compute_label_from_data(self, training_dataframe):
        if self._is_leaf and self._predefined_decision_node:
            if self._prod_output_value in self._positive_output_set:
                label = 1.0
            else:
                label = 0.0
        else:
            if self._prob_class_1 > 0.5:
                label = 1.0
            else:
                label = 0.0
        return label


    def compute_metrics(self, training_dataframe, node_label):
        prediction_column = "predicted_label"
        tmp_training_dataframe = deepcopy(training_dataframe)
        tmp_training_dataframe[prediction_column] = Series([node_label for x in range(len(tmp_training_dataframe))], index = tmp_training_dataframe.index)
        return compute_perf_metrics(tmp_training_dataframe, self._training_label_column, prediction_column, self._training_weight_column, self._beta)


    def compute_cm_from_counts(self, left_counts, right_counts):
        left_prob = left_counts[1] / float(sum(left_counts)) if float(sum(left_counts)) > 0.0 else 0.0
        right_prob = right_counts[1] / float(sum(right_counts)) if float(sum(right_counts)) > 0.0 else 0.0
        if left_prob > right_prob:
            tn = right_counts[0]
            fp = left_counts[0]
            fn = right_counts[1]
            tp = left_counts[1]
        else:
            tn = left_counts[0]
            fp = right_counts[0]
            fn = left_counts[1]
            tp = right_counts[1]
        cm = np.array([[tn, fp],[fn, tp]])
        return cm


    def compute_split_score(self, class_counts_lt, class_counts_gte):
        cm = self.compute_cm_from_counts(class_counts_lt, class_counts_gte)
        if self._split_metric == "entropy":
            split_score = -1
        elif self._split_metric == "f1":
            split_score = f1_score(cm)
        elif self._split_metric == "f1_beta":
            split_score = f1_score(cm, beta = self._beta)
        elif self._split_metric == "mcc":
            split_score = mcc(cm)
        elif self._split_metric == "acc":
            split_score = accuracy(cm)
        else:
            split_score = compute_avg_gini(class_counts_lt, class_counts_gte)
        return split_score


    def new_score_better(self, score_old, score_new):
        if self._split_metric == "entropy":
            res = score_new < score_old
        elif self._split_metric == "f1":
            res = score_new > score_old
        elif self._split_metric == "f1_beta":
            res = score_new > score_old
        elif self._split_metric == "mcc":
            res = score_new > score_old
        elif self._split_metric == "acc":
            res = score_new > score_old
        else:
            res = score_new < score_old
        return res


    def update_split(self, idx, training_dataframe_sorted, split_dict):
        # get label and value
        l = training_dataframe_sorted[self._training_label_column][idx]
        v = training_dataframe_sorted[self._training_feature_column][idx]
        # make next split dict
        next_split = deepcopy(split_dict)
        # update value and index
        next_split['split_value'] = v
        next_split['split_data_idx'] = idx
        # move lavel from right to left
        next_split['class_counts_lt'][l] += training_dataframe_sorted[self._training_weight_column][idx]
        next_split['class_counts_gte'][l] -= training_dataframe_sorted[self._training_weight_column][idx]
        next_split['split_score'] = self.compute_split_score(next_split['class_counts_lt'], next_split['class_counts_gte'])
        return next_split


    def find_split(self, training_dataframe):
        """
        training_dataframe : should be a slice of a data frame with [training_label_column, training_feature_column] order
        """
        # right labels greater than or equal to split value
        training_dataframe_sorted = training_dataframe.sort_values(by=self._training_feature_column)

        # index
        idx = training_dataframe_sorted.index[0]
        l = training_dataframe_sorted[self._training_label_column][idx]
        v = training_dataframe_sorted[self._training_feature_column][idx]
        gte_counts = [self._n-self._n_pos, self._n_pos]
        current_split = {
            'split_data_idx' : 0,
            'split_value' : v,
            'split_type' : 'gte',
            'class_counts_lt' : [0, 0],
            'class_counts_gte' : [self._n-self._n_pos, self._n_pos],
            'split_score' : self.compute_split_score([0, 0], [self._n-self._n_pos, self._n_pos]),
            }
        best_split = deepcopy(current_split)
        last_split = deepcopy(current_split)

        for idx in list(training_dataframe_sorted.index[1:]):
            # if not same as last
            if not last_split['split_value'] == current_split['split_value']:
                if self.new_score_better(best_split['split_score'], last_split['split_score']):
                    best_split = last_split

            last_split = deepcopy(current_split)
            current_split = self.update_split(idx, training_dataframe_sorted, current_split)

        # collect indicies left and right
        s_values = training_dataframe_sorted[self._training_feature_column]
        best_split['gte_indices'] = s_values[s_values >= best_split['split_value']].index
        best_split['lt_indices'] = s_values[s_values < best_split['split_value']].index
        self._best_split = best_split
        self._split_value = self._best_split['split_value']


    def print(self):
        print(self.to_string())


    def to_string(self):
        class_count = [self._n_neg, self._n_pos]
        output = """DecisionTreeNode ::
        key : {0}, is_leaf : {1}, prod_feature : {2}, class_count[neg, pos] : {3}, n : {4}, label : {5}, production_output : {6}
        best_split ::
        {7},
        classification_metrics ::
        {8}""".format(self._key, self._is_leaf, self._prod_feature_name, class_count, self._n, self._label, self._prod_output_value, self.format_best_split(), self.format_classification_metrics())
        return output


    def format_classification_metrics(self):
        if not self._cm_metrics:
            output = "classification metrics not computed"
        else:
            output = format_cm_metrics(self._cm_metrics)
        return output


    def format_best_split(self):
        if self._best_split:
            output = """split_value : {0}, split_score : {1}, class_coutns_lt : {2}, class_counts_gte : {3}""".format(
                self._best_split['split_value'],
                self._best_split['split_score'],
                self._best_split['class_counts_lt'],
                self._best_split['class_counts_gte'],
                self.format_classification_metrics()
                )
        else:
            if self._is_leaf:
                output = "is leaf, no best split"
            else:
                output = "best split not trained yet"
        return output

    def to_dict(self):
        if self._prod_feature_name and self._split_value:
            production_feature_and_split_value = [self._prod_feature_name, int(self._split_value)]
        else:
            production_feature_and_split_value = None
        
        output = {
            "key" : int(self._key),
            "is_leaf" : self._is_leaf,
            "best_split" : self._best_split,
            "split_value" : int(self._split_value) if self._split_value else None,
            "split_metric" : self._split_metric,
            "parent_id" : self._parent._key if self._parent else -1,
            "left_id" : self._left._key if self._left else -1,
            "right_id" : self._right._key if self._right else -1,
            "label_column" : self._training_label_column,
            "feature_column" : self._training_feature_column,
            "weight_column" : self._training_weight_column,
            "production_feature_name" : self._prod_feature_name,
            "production_output_value" : self._prod_output_value,
            "positive_output_set" : self._positive_output_set,
            "is_predefined_decision_node" : self._predefined_decision_node,
            "beta" : self._beta,
            "cm_metrics" : self._cm_metrics,
            "label" : self._label,
            "n" : self._n,
            "n_pos" : self._n_pos,
            "n_neg" : self._n_neg,
            "prob_class_1" : self._prob_class_1,
            "feature_and_split_value_production" : production_feature_and_split_value
            }
        return sort_dictionary(output)

    """

    """

    def print_best_split(self):
        print(self.format_best_split)






def make_treenode_from_node(parent, node, beta = BETA):
    """
    made a tree node from a Node named tuple
    """
    treenode = DecisionTreeNode(key = node.key, 
                        parent = parent, 
                        is_leaf = node.is_leaf, 
                        training_label_column = node.training_label_column,
                        training_feature_column = node.training_feature_column,
                        training_weight_column = node.training_weight_column,
                        prod_feature_name = node.prod_feature_name,
                        prod_output_value = node.prod_output_value,
                        split_value = node.split_value,
                        split_metric = node.split_metric,
                        left = node.left,
                        right = node.right,
                        beta = beta)


    if node.left:
        left_child = make_treenode_from_node(treenode, node.left, beta)
        treenode.set_left(left_child)

    if node.right:
        right_child = make_treenode_from_node(treenode, node.right, beta)
        treenode.set_right(right_child)

    return treenode