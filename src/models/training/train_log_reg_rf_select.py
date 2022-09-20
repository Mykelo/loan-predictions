import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description='Train logistic regression using features selected by Random Forest')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--random-state', type=int, help='Random state', default=42)
    args = parser.parse_args()

    X = np.load(args.X_input)
    y = np.load(args.y_input)
    
    clf_rf = RandomForestClassifier(max_depth=3, random_state=args.random_state).fit(X, y)
    ind = np.argsort(clf_rf.feature_importances_)[-100:]
    X_selected = X[:, ind]

    clf = LogisticRegression(max_iter=200, random_state=args.random_state).fit(X_selected, y)
    with open('./models/log_reg_rf_select.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open('./models/random_forest.pkl', 'wb') as f:
        pickle.dump(clf_rf, f)

main()