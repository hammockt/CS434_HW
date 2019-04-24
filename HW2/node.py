import math
import statistics

class Node():
    """ represents a node in a decision tree """
    def __init__(self, x_vals, y_vals, parent=None):
        self.parent = parent
        self.children = []
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.feature = None
        self.test_bound = None

    #https://stackoverflow.com/a/1859910
    def entropy(self):
        """ calculates the entropy (uncertainty/randomness) of the node """
        y_counts = {}
        for y_val in self.y_vals:
            #python truncates floats if they are used as keys...
            #play it safe and use the exact string representation as the key instead (str also truncates)
            key = repr(y_val)
            if key not in y_counts:
                y_counts[key] = 0
            y_counts[key] += 1
        total_y_count = sum(y_counts.values())

        #we should probably break this into two separate methods...ehh
        u_s = 0
        if not self.children:
            for count in y_counts.values():
                p_val = count / total_y_count
                u_s -= p_val * math.log2(p_val)
        else:
            for child in self.children:
                u_s += len(child.y_vals) / total_y_count * child.entropy()

        return u_s

    def test(self):
        """ tests the node to figure out the best way to split the data """
        best_feature = {
            "feature_index": None,
            "test_bound": None,
            "entropy": self.entropy()
        }
        for col in range(len(self.x_vals[0])):
            # Get column from x_vals
            feature_vals = [
                (y_index, self.x_vals[y_index][col])
                for y_index in range(len(self.y_vals))
            ]
            if len(feature_vals) <= 1:
                continue
            feature_vals.sort(key=lambda x: x[1])
            for index, val in enumerate(feature_vals[1:]):
                # Since we're skipping the first element
                j = index + 1
                # <, >=
                # Set y_val separations based on test boundary
                y_vals = {"l": [self.y_vals[k[0]] for k in feature_vals[:j]],
                          "r": [self.y_vals[k[0]] for k in feature_vals[j:]]}

                self.children.append(Node(None, y_vals["l"], self))
                self.children.append(Node(None, y_vals["r"], self))
                entropy = self.entropy()
                if best_feature["entropy"] > entropy:
                    best_feature = {
                        "feature_index": col,
                        "test_bound": val[1],
                        "entropy": entropy
                    }
                self.children = []
        self.feature = best_feature["feature_index"]
        self.test_bound = best_feature["test_bound"]
        return best_feature

    def test_and_apply(self):
        """ tests the node and applies the split """
        best_feature = self.test()
        if best_feature["feature_index"] is None:
            return best_feature
        x_vals = ([], [])
        y_vals = ([], [])
        for i, point in enumerate(self.x_vals):
            if point[self.feature] < self.test_bound:
                x_vals[0].append(point)
                y_vals[0].append(self.y_vals[i])
            else:
                x_vals[1].append(point)
                y_vals[1].append(self.y_vals[i])
        for i, x_val in enumerate(x_vals):
            self.children.append(Node(x_val, y_vals[i], self))
        return best_feature

    def predicted_value(self, point):
        """ descends the decision tree to predict the output of a point """
        if not self.children:
            return self.decision()

        if point[self.feature] < self.test_bound:
            return self.children[0].predicted_value(point)
        return self.children[1].predicted_value(point)

    def decision(self):
        """ determines how the output is picked on a leaf node """
        try:
            return statistics.mode(self.y_vals)
        except statistics.StatisticsError:
            return self.y_vals[0]
