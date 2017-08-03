from collections import namedtuple
from copy import deepcopy
import numpy as np
import os, json
import scipy as sp
import scipy.stats as stats
from sklearn import metrics
from pandas import DataFrame, read_csv, Int64Index, Series
import pandas as pd
from sklearn import tree
import numpy as np
import os, sys
from StructuredDecisionTreeLearning.constants import *
from StructuredDecisionTreeLearning.math_helpers import *
from StructuredDecisionTreeLearning.gen_helpers import *
from StructuredDecisionTreeLearning.Node import *
from StructuredDecisionTreeLearning.DecisionTreeNode import *


class StructuredDecisionTree:

    def __init__(self, rootNode, beta = BETA):
        """
        Convert the Node based tree structure into DecisionDecisionTreeNodes
        """

        self.structure = rootNode
        self._num_leafs = 0
        self._leafs = {}
        self._root = self.build_tree_rec(None, rootNode, beta)
        self._positive_output_set = self._root._positive_output_set


    def train(self, training_dataframe):
        """
        Train the tree from root to leaves
        """
        self._root.train(training_dataframe, train_downward = True)


    def build_tree_rec(self, parent, node, beta = BETA):
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
                            left = node.left,
                            right = node.right,
                            beta = beta)

        if treenode._is_leaf:
            self._num_leafs += 1
            self._leafs[treenode._key] = treenode


        if node.left:
            left_child = self.build_tree_rec(treenode, node.left, beta)
            treenode.set_left(left_child)


        if node.right:
            right_child = self.build_tree_rec(treenode, node.right, beta)
            treenode.set_right(right_child)

        return treenode


    def predict(self, input, join_to_df = True, prediction_column_name = 'predictions'):
        """
        predict either on a single row (series) from a dataframe or an entire dataframe
        """

        if type(input) == Series:
            output = self._root.predict(input)
        else:
            indices = []
            predictions = []
            for idx, row in input.iterrows():
                indices.append(idx)
                predictions.append(self._root.predict(row))
            s = Series(predictions, index = indices)
            if join_to_df:
                input[prediction_column_name] = s
                output = input
            else:
                output = s

            return output

    def to_dict(self):
        leaves_dict = {}
        queue = [self._root]
        self.bf_traverse_to_dict(queue, leaves_dict)
        return sort_dictionary(leaves_dict)

     
    def bf_traverse_to_dict(self, queue, leaves_dict):
        if not len(queue) > 0:
            return
        else:
            this_node = queue.pop(0)

            if this_node._left:
                queue.append(this_node._left)

            if this_node._right:
                queue.append(this_node._right)

            leaves_dict[str(this_node._key)] = this_node.to_dict()
            self.bf_traverse_to_dict(queue, leaves_dict)


    def bf_traverse_print(self, queue):
        if not len(queue) > 0:
            return
        else:
            this_node = queue.pop(0)

            if this_node._left:
                queue.append(this_node._left)

            if this_node._right:
                queue.append(this_node._right)

            this_node.print()
            self.bf_traverse_print(queue)


    def print_trained_tree(self):
        queue = [self._root]
        self.bf_traverse_print(queue)


    def print_leaves(self):
        for key, leaf_node in sorted(self._leafs.items()):
            leaf_node.print()