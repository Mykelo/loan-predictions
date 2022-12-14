import numpy as np
import argparse
import pickle
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    """
    Script for training a decision tree classifier.
    """
    parser = argparse.ArgumentParser(description='Train decision tree using all features')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--random-state', type=int, help='Random state', default=42)
    args = parser.parse_args()

    X = np.load(args.X_input, allow_pickle=True)
    y = np.load(args.y_input, allow_pickle=True)

    clf = DecisionTreeClassifier(ccp_alpha=0.01, random_state=args.random_state).fit(X, y)

    with open(f'./models/decision_tree/rs_{args.random_state}.pkl', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()