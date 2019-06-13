"""
decision tree
"""

import json
import random
from model import Model

def gini_index(y_counts, total_y):
    """ calculates the gini impurity """
    gini = 1
    for count in y_counts.values():
        gini -= (count / total_y) ** 2

    return gini

def gini_split(splits, total_y):
    """ calculates the gini impurity of a split """
    gini = 0
    for split in splits:
        child_gini = gini_index(split["counts"], split["total"])
        gini += (split["total"] / total_y) * child_gini

    return gini

def value_counts(values):
    """ calculates the y counts and stores them in a dictionary """
    counts = {}
    for value in values:
        key = repr(value)
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

    return counts

class DecisionTree(Model):
    """ class that implements ML decision trees """

    def __init__(self):
        self.children = []
        self.feature = None
        self.test_bound = None
        self.y_counts = {}
        self.gini = gini_index(self.y_counts, 0)

    def test(self, row_pointers, x_vals, y_vals, random_features=None):
        """
        tests all the ways every feature can be split
        it then records this inside our object variables
        """

        self.y_counts = value_counts((y_vals[row_pointer] for row_pointer in row_pointers))
        empty_y_counts = {key:0 for key in self.y_counts}
        total_y_count = len(row_pointers)

        self.gini = gini_index(self.y_counts, total_y_count)
        best_gini = self.gini

        num_features = len(x_vals[0])
        if random_features is None:
            random_features = num_features

        for feature in random.sample(range(num_features), random_features):
            #sort the values for this column/feature
            row_pointers.sort(key=lambda row_pointer: x_vals[row_pointer][feature])

            splits = []
            splits.append({"counts": self.y_counts.copy(), "total": total_y_count})
            splits.append({"counts": empty_y_counts.copy(), "total": 0})

            #if we split by the largest x, then the right split will be empty
            #this can happen when there is more than one row with the largest x
            largest_x = x_vals[row_pointers[-1]][feature]
            for row_pointer in row_pointers:
                if x_vals[row_pointer][feature] == largest_x:
                    break

                # "move" the point from the left to the right
                y_key = repr(y_vals[row_pointer])

                splits[0]["counts"][y_key] -= 1
                splits[0]["total"] -= 1

                splits[1]["counts"][y_key] += 1
                splits[1]["total"] += 1

                # calculate the gini_index of the split, also known as the gini_split
                new_gini = gini_split(splits, total_y_count)

                if new_gini < best_gini:
                    best_gini = new_gini
                    self.feature = feature
                    self.test_bound = x_vals[row_pointer][feature]

    def train(self, row_pointers, x_vals, y_vals, **train_args):
        random_features = train_args.get("random_features")
        self.test(row_pointers, x_vals, y_vals, random_features)

        if self.feature is None:
            return

        child_row_pointers = [[], []]
        for row_pointer in row_pointers:
            if x_vals[row_pointer][self.feature] <= self.test_bound:
                child_row_pointers[0].append(row_pointer)
            else:
                child_row_pointers[1].append(row_pointer)

        for new_row_pointers in child_row_pointers:
            self.children.append(DecisionTree())
            self.children[-1].train(new_row_pointers, x_vals, y_vals, **train_args)

    def predict(self, point):
        if not self.children:
            # gets the key with the largest count
            return float(max(self.y_counts, key=lambda key: self.y_counts[key]))

        if point[self.feature] <= self.test_bound:
            return self.children[0].predict(point)
        return self.children[1].predict(point)

    @staticmethod
    def from_dict(in_dict):
        tree = DecisionTree()
        for key, value in in_dict.items():
            if key != "children":
                tree.__dict__[key] = value

        for child in in_dict.get("children"):
            tree.children.append(DecisionTree.from_dict(child))

        return tree

    @staticmethod
    def from_json(json_str):
        in_dict = json.loads(json_str)
        return DecisionTree.from_dict(in_dict)
