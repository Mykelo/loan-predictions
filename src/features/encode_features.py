import pandas as pd
import numpy as np
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle


def main() -> None:
    """
    Script responsible for encoding features.    
    """
    parser = argparse.ArgumentParser(description='Encode features in the training and test datasets')
    parser.add_argument('--train-input', type=str, help='Train input file', default='./data/interim/train_clean.csv')
    parser.add_argument('--test-input', type=str, help='Test input file', default='./data/interim/test_clean.csv')
    
    args = parser.parse_args()

    # Store date columns to make sure that they're processed correctly
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

    # Convert date columns to numeric values
    train_data[date_columns] = train_data[date_columns].apply(pd.to_numeric)
    test_data[date_columns] = test_data[date_columns].apply(pd.to_numeric)

    # Seperate target columns from the rest of the datasets
    train_y = train_data['target'].to_numpy()
    test_y = test_data['target'].to_numpy()
    train_data = train_data.drop(columns=['target'], axis=1)
    test_data = test_data.drop(columns=['target'], axis=1)

    # Encode category features using one-hot encoding method
    oh_columns = train_data.select_dtypes('object').columns.values 
    column_trans = ColumnTransformer(
        [('categories', OneHotEncoder(dtype='int', sparse=False), oh_columns)],
        remainder='passthrough', verbose_feature_names_out=False)

    train_data = column_trans.fit_transform(train_data)
    test_data = column_trans.transform(test_data)

    with open('./data/processed/X_train.npy', 'wb') as f:
        np.save(f, train_data)
    with open('./data/processed/y_train.npy', 'wb') as f:
        np.save(f, train_y)

    with open('./data/processed/X_test.npy', 'wb') as f:
        np.save(f, test_data)
    with open('./data/processed/y_test.npy', 'wb') as f:
        np.save(f, test_y)

    with open('./models/columns_encoder.pkl', 'wb') as f:
        pickle.dump(column_trans, f)


if __name__ == '__main__':
    main()