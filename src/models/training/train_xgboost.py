import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle

def main() -> None:
    parser = argparse.ArgumentParser(description='Train XGBoost model using all features')
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    args = parser.parse_args()

    X = np.load(args.X_input)
    y = np.load(args.y_input)

    xgb_model = xgb.XGBClassifier()
    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.1, 0.2, 0.3, 0.4],
        'max_depth': [3, 4, 5, 6]
    }
    grid = GridSearchCV(xgb_model, parameters, scoring='f1_macro', cv=5)
    grid = grid.fit(X, y)

    with open('./models/xgboost.pkl', 'wb') as f:
        pickle.dump(grid.best_estimator_, f)

main()