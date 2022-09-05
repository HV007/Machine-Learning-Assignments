# Decision Tree Classifier

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if len(np.unique(y)) <= 1:
            return None
        if depth == self.max_depth:
            return None
        if len(X) == 0:
            return None
        if len(np.unique(y)) == 1:
            return y[0]
        best_feature, best_threshold, best_score = None, None, 0
        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                score = self._score(X, y, feature, threshold)
                if score > best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score
        if best_score == 0:
            return self.classes_[np.argmax(np.bincount(y))]
        left_X, left_y = X[X[:, best_feature] <= best_threshold], y[X[:, best_feature] <= best_threshold]
        right_X, right_y = X[X[:, best_feature] > best_threshold], y[X[:, best_feature] > best_threshold]
        left_tree = self._grow_tree(left_X, left_y, depth + 1)
        right_tree = self._grow_tree(right_X, right_y, depth + 1)
    
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }