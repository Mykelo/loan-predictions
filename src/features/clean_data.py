import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
from itertools import combinations


def parse_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the `emb_length` column to float. First it converts extreme cases of "< 1 year" and 
    "10+ years" with "0 years" and "11 years" respectively to separate these groups from the rest.
    """
    df_copy = df.copy()
    df_copy['emp_length'] = df_copy['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})
    df_copy['emp_length'] = df_copy['emp_length'].str.extract('(\d+)').astype('float')  
    return df_copy


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Drops specified columns.
    """
    return df.drop(cols, axis=1, errors='ignore')


def fill_na(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces missing values in columns. 
    """

    # Columns are filled using extreme values (maximum and minimum depending on a column).
    # Fill categorical columns with the most frequent value.
    fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
        'mths_since_last_major_derog', 'mths_since_last_record',
        'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
        'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
        'pct_tl_nvr_dlq']
    fill_min = np.setdiff1d(source.columns.values, fill_max) 
    
    df_copy = df.copy()
    df_copy[fill_max] = df_copy[fill_max].fillna(source[fill_max].max(numeric_only=True))
    df_copy[fill_min] = df_copy[fill_min].fillna(source[fill_min].min(numeric_only=True))

    fill_most_frequent = df_copy.select_dtypes('object').columns.values
    df_copy[fill_most_frequent] = df_copy[fill_most_frequent].fillna(source[fill_most_frequent].mode().iloc[0])

    return df_copy


def calculate_pearsonr(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates Pearson's R value for every pair of numeric columns in a data frame.
    """
    num_feat = X.select_dtypes('number').columns.values
    comb_num_feat = np.array(list(combinations(num_feat, 2)))
    corr_num_feat = np.array([])
    for comb in comb_num_feat:
        corr = pearsonr(X[comb[0]], X[comb[1]])[0]
        corr_num_feat = np.append(corr_num_feat, corr)

    return comb_num_feat, corr_num_feat


def main() -> None:
    """
    Cleans data based on analysis presented in this notebook:
    https://www.kaggle.com/code/pavlofesenko/minimizing-risks-for-loan-investments
    """
    parser = argparse.ArgumentParser(description='Clean the training and test datasets')
    parser.add_argument('--train-input', type=str, help='Train input file', default='./data/interim/train_raw.csv')
    parser.add_argument('--test-input', type=str, help='Test input file', default='./data/interim/test_raw.csv')
    
    args = parser.parse_args()

    # Store date columns to make sure that they're processed correctly
    date_columns = [
        'issue_d',
        'earliest_cr_line',
        'last_credit_pull_d',
        'last_pymnt_d',
        'sec_app_earliest_cr_line'
    ]
    train_data = pd.read_csv(args.train_input, parse_dates=date_columns, infer_datetime_format=True)
    train_data = train_data.reset_index(drop=True)
    test_data = pd.read_csv(args.test_input, parse_dates=date_columns, infer_datetime_format=True)
    test_data = test_data.reset_index(drop=True)

    # Find columns which are missing over 90% of values
    nan_mean = train_data.isna().mean()
    nan_mean = nan_mean[nan_mean != 0].sort_values()
    nan_to_drop = list(nan_mean[nan_mean > 0.9].keys())

    # These columns either have too many unique values or are correlated with other
    # categorical columns
    cat_to_drop = ['emp_title', 'url', 'policy_code', 'sub_grade', 'title', 'zip_code', 'debt_settlement_flag']

    train_data = train_data \
        .pipe(parse_emp_length) \
        .pipe(drop_columns, cat_to_drop) \
        .pipe(drop_columns, nan_to_drop)
    train_data = train_data.pipe(fill_na, train_data)

    # Use the train dataset when filling missin values in order to prevent data leakage
    test_data = test_data \
        .pipe(parse_emp_length) \
        .pipe(drop_columns, cat_to_drop) \
        .pipe(drop_columns, nan_to_drop) \
        .pipe(fill_na, train_data)

    # Drop one column from each pair of columns which are highly correlated
    combs, correlations = calculate_pearsonr(train_data)
    high_corr_combs = combs[np.abs(correlations) >= 0.9]
    high_corr_num_to_drop = np.unique(high_corr_combs[:, 0])   

    train_data = train_data.pipe(drop_columns, high_corr_num_to_drop)
    test_data = test_data.pipe(drop_columns, high_corr_num_to_drop)

    train_data['term'] = train_data['term'].str.lstrip()
    test_data['term'] = test_data['term'].str.lstrip()

    train_data.to_csv('./data/interim/train_clean.csv', index=False)
    test_data.to_csv('./data/interim/test_clean.csv', index=False)


if __name__ == '__main__':
    main()