import numpy as np
import argparse
import xgboost as xgb
import pickle

def main() -> None:
    parser = argparse.ArgumentParser(description='Train XGBoost model using all features')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--random-state', type=int, help='Random state', default=42)
    args = parser.parse_args()

    X = np.load(args.X_input, allow_pickle=True)
    y = np.load(args.y_input, allow_pickle=True)

    xgb_model = xgb.XGBClassifier(seed=args.random_state)
    xgb_model = xgb_model.fit(X, y)

    with open(f'./models/xgboost/rs_{args.random_state}.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)


if __name__ == '__main__':
    main()