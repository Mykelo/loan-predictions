import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations


def parse_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['emp_length'] = df_copy['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})
    df_copy['emp_length'] = df_copy['emp_length'].str.extract('(\d+)').astype('float')  
    return df_copy


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(cols, axis=1, errors='ignore')


def fill_na(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
        'mths_since_last_major_derog', 'mths_since_last_record',
        'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
        'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
        'pct_tl_nvr_dlq']
    fill_min = np.setdiff1d(source.columns.values, fill_max) 
    
    df_copy = df.copy()
    df_copy[fill_max] = df_copy[fill_max].fillna(source[fill_max].max(numeric_only=True))
    df_copy[fill_min] = df_copy[fill_min].fillna(source[fill_min].min(numeric_only=True))

    return df_copy


def calculate_pearsonr(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    num_feat = X.select_dtypes('number').columns.values
    comb_num_feat = np.array(list(combinations(num_feat, 2)))
    corr_num_feat = np.array([])
    for comb in comb_num_feat:
        corr = pearsonr(X[comb[0]], X[comb[1]])[0]
        corr_num_feat = np.append(corr_num_feat, corr)

    return comb_num_feat, corr_num_feat





# def calculate_cramer_v(X: pd.DataFrame) -> np.ndarray:
#     cat_feat = X.select_dtypes('object').columns.values
#     comb_cat_feat = np.array(list(combinations(cat_feat, 2)))
#     corr_cat_feat = np.array([])
#     for comb in comb_cat_feat:
#         table = pd.pivot_table(X, values='target', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
#         corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
#         corr_cat_feat = np.append(corr_cat_feat, corr)

#     return corr_cat_feat


def main() -> None:
    parser = argparse.ArgumentParser(description='Clean the training and test datasets')
    parser.add_argument('--train-input', type=str, help='Train input file', default='./data/interim/train_raw.csv')
    parser.add_argument('--test-input', type=str, help='Test input file', default='./data/interim/test_raw.csv')
    
    args = parser.parse_args()

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

    nan_mean = train_data.isna().mean()
    nan_mean = nan_mean[nan_mean != 0].sort_values()
    nan_to_drop = list(nan_mean[nan_mean > 0.9].keys())

    cat_to_drop = ['emp_title', 'url', 'policy_code', 'sub_grade', 'title', 'zip_code']
    train_data = train_data \
        .pipe(parse_emp_length) \
        .pipe(drop_columns, cat_to_drop) \
        .pipe(drop_columns, nan_to_drop)
    train_data = train_data.pipe(fill_na, train_data)

    test_data = test_data \
        .pipe(parse_emp_length) \
        .pipe(drop_columns, cat_to_drop) \
        .pipe(drop_columns, nan_to_drop) \
        .pipe(fill_na, train_data)

    combs, correlations = calculate_pearsonr(train_data)
    high_corr_combs = combs[np.abs(correlations) >= 0.9]
    high_corr_num_to_drop = np.unique(high_corr_combs[:, 0])   

    train_data = train_data.pipe(drop_columns, high_corr_num_to_drop)
    test_data = test_data.pipe(drop_columns, high_corr_num_to_drop)

    train_data.to_csv('./data/interim/train_clean.csv', index=False)
    test_data.to_csv('./data/interim/test_clean.csv', index=False)


if __name__ == '__main__':
    main()