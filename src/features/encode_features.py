import pandas as pd
import numpy as np
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description='Encode features in the training and test datasets')
    parser.add_argument('--train-input', type=str, help='Train input file', default='./data/interim/train_clean.csv')
    parser.add_argument('--test-input', type=str, help='Test input file', default='./data/interim/test_clean.csv')
    
    args = parser.parse_args()

    date_columns = [
        'issue_d',
        'earliest_cr_line',
        'last_credit_pull_d',
        'last_pymnt_d',
    ]
    train_data = pd.read_csv(args.train_input, parse_dates=date_columns, infer_datetime_format=True)
    train_data = train_data.reset_index(drop=True)
    test_data = pd.read_csv(args.test_input, parse_dates=date_columns, infer_datetime_format=True)
    test_data = test_data.reset_index(drop=True)

    numeric_columns = train_data.select_dtypes('number').columns.values
    oh_columns = train_data.select_dtypes('object').columns.values 

    column_trans = ColumnTransformer(
        [('numeric', StandardScaler(), numeric_columns),
        ('categories', OneHotEncoder(dtype='int', sparse=True), oh_columns)],
        remainder='drop', verbose_feature_names_out=False)

    train_data = column_trans.fit_transform(train_data)
    test_data = column_trans.transform(test_data)

    with open('./data/processed/train.npy', 'wb') as f:
        np.save(f, train_data)

    with open('./data/processed/test.npy', 'wb') as f:
        np.save(f, test_data)

    with open('./models/columns_encoder.pkl', 'wb') as f:
        pickle.dump(column_trans, f)


if __name__ == '__main__':
    main()