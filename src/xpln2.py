from sklearn.tree import DecisionTreeClassifier

class XPLN2:
    def __init__(self, best, rest):
        self.best = best
        self.rest = rest
        self.data = None

    def decision_tree(self, data):
        self.data = data
        X_best = []
        y_best = []
        X_rest = []
        y_rest = []
        X_test = []
        for row in self.best.Rows:
            X_best.append([row.cells[col.at] for col in self.best.Cols.x])
            y_best.append('best')
        for row in self.rest.Rows:
            X_rest.append([row.cells[col.at] for col in self.rest.Cols.x])
            y_rest.append('rest')
        for row in self.data.Rows:
            X_test.append([row.cells[col.at] for col in self.data.Cols.x])

        X_train = X_best + X_rest
        y_train = y_best + y_rest
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        
        best_preds = []
        rest_preds = []
        for idx, row in enumerate(X_test):
            pred = clf.predict([row])
            if pred == "best":
                best_preds.append(self.data.Rows[idx])
            else:
                rest_preds.append(self.data.Rows[idx])
        return best_preds, rest_preds