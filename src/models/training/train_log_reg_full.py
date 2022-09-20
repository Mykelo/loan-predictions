import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import pickle

def main() -> None:
    parser = argparse.ArgumentParser(description='Train logistic regression using all features')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    args = parser.parse_args()

    X = np.load(args.X_input)
    y = np.load(args.y_input)

    clf = LogisticRegression(max_iter=200).fit(X, y)
    with open('./models/log_reg_full.pkl', 'wb') as f:
        pickle.dump(clf, f)

main()