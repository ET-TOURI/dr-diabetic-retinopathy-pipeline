from pyswarm import pso
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def objective(feat_mask, features, labels):
    selected = features[:, feat_mask > 0.5]
    X_train, X_test, y_train, y_test = train_test_split(selected, labels, test_size=0.2)
    clf = SVC()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    penalty = sum(feat_mask) / len(feat_mask)
    return -acc + 0.01 * penalty

def select_features_pso(features, labels):
    dim = features.shape[1]
    lb = [0]*dim
    ub = [1]*dim
    opt_mask, _ = pso(objective, lb, ub, args=(features, labels), maxiter=30, swarmsize=20)
    return opt_mask > 0.5