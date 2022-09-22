import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Train logistic regression using all features')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--random-state', type=int, help='Random state', default=42)
    args = parser.parse_args()

    X = np.load(args.X_input, allow_pickle=True)
    y = np.load(args.y_input, allow_pickle=True)

    clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=args.random_state))
    ])
    clf = clf.fit(X, y)
    with open(f'./models/log_reg_full/rs_{args.random_state}.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()