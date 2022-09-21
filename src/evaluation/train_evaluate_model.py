import os, sys
sys.path.insert(0, os.path.abspath("."))
from typing import Any
import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from src.models.custom_models import LogisticRegressionRF
from src.evaluation.utils import calculate_metrics
import json
from tqdm import tqdm


def get_model(name: str, random_state: int) -> Any:
    clf = None
    if name == 'xgboost':
        clf = xgb.XGBClassifier(seed=random_state)
    elif name == 'log_reg_full':
        clf = LogisticRegression(random_state=random_state)
    elif name == 'log_reg_rf_select':
        clf = LogisticRegressionRF(random_state=random_state)
    else:
        clf = DecisionTreeClassifier(ccp_alpha=0.01, random_state=random_state)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description='Train and evaluate selected model the given number of times')
    parser.add_argument('--X-train', type=str, help='Features input file', default='./data/processed/X_train.npy')
    parser.add_argument('--X-test', type=str, help='Features input file', default='./data/processed/X_test.npy')
    parser.add_argument('--y-train', type=str, help='Labels input file', default='./data/processed/y_train.npy')
    parser.add_argument('--y-test', type=str, help='Labels input file', default='./data/processed/y_test.npy')
    parser.add_argument('--model', type=str, choices=['xgboost', 'log_reg_full', 'log_reg_rf_select', 'decision_tree'], help='Classifier', required=True)
    parser.add_argument('--metrics-file', type=str, help='Metrics path', required=True)
    parser.add_argument('--iterations', type=int, help='Number of iterations to repeat', default=10)
    args = parser.parse_args()

    X_train = np.load(args.X_train)
    X_test = np.load(args.X_test)
    y_train = np.load(args.y_train)
    y_test = np.load(args.y_test)

    metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'auc': []
    }

    for i in tqdm(range(args.iterations)):
        model = get_model(args.model, i)
        model.fit(X_train, y_train)
        model_metrics = calculate_metrics(y_test, model.predict(X_test))
        metrics['accuracy'].append(model_metrics['accuracy'])
        metrics['sensitivity'].append(model_metrics['sensitivity'])
        metrics['specificity'].append(model_metrics['specificity'])
        metrics['auc'].append(model_metrics['auc'])

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()