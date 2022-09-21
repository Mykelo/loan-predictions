import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import argparse
import pickle
from src.evaluation.utils import calculate_metrics
import json


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a specified model')
    parser.add_argument('--model-file', type=str, help='Saved model path', required=True)
    parser.add_argument('--metrics-file', type=str, help='Metrics path', required=True)
    parser.add_argument('--X-input', type=str, help='Features input file', default='./data/processed/X_test.npy')
    parser.add_argument('--y-input', type=str, help='Labels input file', default='./data/processed/y_test.npy')
    
    args = parser.parse_args()

    X = np.load(args.X_input)
    y = np.load(args.y_input)

    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    metrics = calculate_metrics(y, model.predict(X))

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f)

main()