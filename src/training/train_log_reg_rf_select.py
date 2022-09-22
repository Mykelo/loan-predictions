import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import argparse
from src.models.custom_models import LogisticRegressionRF
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main() -> None:
    """
    Script for training Logistic Regression using features selected by Random Forest.
    """
    parser = argparse.ArgumentParser(description='Train logistic regression using features selected by Random Forest')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--random-state', type=int, help='Random state', default=42)
    args = parser.parse_args()

    X = np.load(args.X_input, allow_pickle=True)
    y = np.load(args.y_input, allow_pickle=True)

    clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegressionRF(random_state=args.random_state, features_num=100))
    ])
    clf = clf.fit(X, y)

    with open(f'./models/log_reg_rf_select/rs_{args.random_state}.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()