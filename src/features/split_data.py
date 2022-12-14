import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main() -> None:
    """
    Script responsible for splitting raw data and encoding the target column.
    """
    parser = argparse.ArgumentParser(description='Encode the target label and split data')
    parser.add_argument('--input', type=str, help='Input file', default='./data/raw/accepted_2007_to_2018Q4.csv')
    
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = data.reset_index(drop=True)

    X = data[data['loan_status'] != 'Current'].copy()
    X.loc[X['loan_status'] == 'Fully Paid', 'target'] = 0
    X.loc[X['loan_status'] != 'Fully Paid', 'target'] = 1
    X = X.drop(columns=['id', 'loan_status'])

    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['target']), X['target'], test_size=0.2, stratify=X['target'], random_state=42
    )

    X_train['target'] = y_train
    X_test['target'] = y_test

    X_train.to_csv('./data/interim/train_raw.csv', index=False)
    X_test.to_csv('./data/interim/test_raw.csv', index=False)


if __name__ == '__main__':
    main()