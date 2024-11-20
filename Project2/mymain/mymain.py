import pandas as pd
from datetime import datetime, timedelta
import dateutil
import numpy as np
import scipy
import sklearn as sk
import statsmodels.api as sm
import patsy
import warnings
import time
# Suppress all warnings
warnings.filterwarnings("ignore")

def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks
    return data


# What have we tried (III)

def svd(train_data, d=5):
    # some random value
    final_data = []
    departments = train_data['Dept'].unique()

    for dept in departments:
        # Grabbing stores that have the current dept
        filtered_train = train_data[train_data['Dept'] == dept]

        selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]
        train_dept_ts = selected_columns.pivot(index='Date', columns='Store', values='Weekly_Sales').reset_index()

        # Replace all missing values with zero
        X_train = train_dept_ts.iloc[:, 1:]
        X_train = X_train.to_numpy()
        X_train = np.nan_to_num(X_train)

        # Center X values
        store_mean = np.mean(X_train, axis=0)
        X_centered = X_train - store_mean

        # Implement SVD
        U, D, VT = np.linalg.svd(X_centered, full_matrices=False)

        # Take first d components and fill rest of the diag with 0s
        D[d:] = 0

        # Make a reduced rank (smoothed) version of the original dataset
        X_bar = (U @ np.diag(D) @ VT) + store_mean

        stores_list = train_dept_ts.columns[1:]
        stores_df = pd.DataFrame(X_bar, columns=stores_list)
        stores_df["Date"] = train_dept_ts["Date"]

        reconstructed = pd.melt(stores_df, id_vars=['Date'], var_name='Store', value_name='Weekly_Sales',
                                value_vars=stores_list)
        reconstructed['Dept'] = dept
        reconstructed["Store"] = reconstructed["Store"].astype(np.int64)

        final_data.append(reconstructed)
    return pd.concat(final_data, ignore_index=True)


def postprocess(data):
    shift = 1
    threshold = 1.1
    final_data = data
    departments = data['Dept'].unique()

    for dept in departments:
        # Grabbing stores that have the current dept
        filtered_train = data[data['Dept'] == dept]
        selected_columns = filtered_train[['Store', 'Date', 'Weekly_Pred']]
        data_filtered = selected_columns.pivot(index='Store', columns='Date', values='Weekly_Pred').reset_index()

        result = apply_shift(data_filtered, shift, threshold).set_index('Store')

        reconstructed = pd.DataFrame(result, columns=data_filtered.columns[1:], index=data_filtered['Store'])
        reconstructed = reconstructed.reset_index().melt(id_vars=['Store'], var_name='Date', value_name='Weekly_Pred')
        reconstructed['Dept'] = dept

        for _, row in reconstructed.iterrows():
            final_data.loc[(final_data['Store'] == row['Store']) & (final_data['Date'] == row['Date']) & (
                        final_data['Dept'] == row['Dept']), 'Weekly_Pred'] = row['Weekly_Pred']

    return final_data


def apply_shift(data, shift=1, threshold=1.1):
    week_1 = '2011-12-02'
    week_2 = '2011-12-09'
    week_3 = '2011-12-16'
    week_4 = '2011-12-23'
    week_5 = '2011-12-30'

    fold_5 = data.loc[:, (data.columns >= week_1) & (data.columns <= week_5)]

    if week_1 not in fold_5:
        rows = fold_5.shape[0]
        fold_5[week_1] = np.full(rows, np.nan)

    if week_2 not in fold_5:
        rows = fold_5.shape[0]
        fold_5[week_2] = np.full(rows, np.nan)

    if week_3 not in fold_5:
        rows = fold_5.shape[0]
        fold_5[week_3] = np.full(rows, np.nan)

    if week_4 not in fold_5:
        rows = fold_5.shape[0]
        fold_5[week_4] = np.full(rows, np.nan)
    if week_5 not in fold_5:
        rows = fold_5.shape[0]
        fold_5[week_5] = np.full(rows, np.nan)

    baseline = fold_5[[week_1, week_5]].mean(axis=1).mean()
    surge = fold_5[[week_2, week_3, week_4]].mean(axis=1).mean()

    fold_5[fold_5.isna()] = 0

    if surge / baseline > threshold:
        shifted_sales = ((7 - shift) / 7) * fold_5
        shifted_sales[[week_2, week_3, week_4, week_5]] = shifted_sales[[week_2, week_3, week_4, week_5]].values + (
                    shift / 7) * fold_5[[week_1, week_2, week_3, week_4]].values
        shifted_sales[[week_1]] = fold_5[[week_1]]
        data.loc[:, (data.columns >= week_1) & (data.columns <= week_5)] = shifted_sales

    return data


def process(train_file_path='train.csv', test_file_path='test.csv', pred_file_path='mypred.csv'):
    start_time = time.time()

    # Load data
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    # Check for date range in test data for fold 5
    is_fold_5 = test['Date'].between("2011-11-04", "2011-12-30").any()

    # Apply SVD by department for dimensionality reduction
    train_svd = svd(train, 8)
    test_pred = pd.DataFrame()

    # Filter for shared store-dept pairs in train and test; filter out pairs with zero occurrences
    train_pairs = train_svd[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how='inner', on=['Store', 'Dept'])

    # Join with common pairs and add week/year columns
    train_split = unique_pairs.merge(train_svd, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)

    # set up data for each split
    X = patsy.dmatrix('Weekly_Sales + Store + Dept + Yr  + Wk',
                      data=train_split,
                      return_type='dataframe')
    train_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
    test_split = preprocess(test_split)
    X = patsy.dmatrix('Store + Dept + Yr  + Wk',
                      data=test_split,
                      return_type='dataframe')
    X['Date'] = test_split['Date']
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    keys = list(train_split)

    for key in keys:
        X_train = train_split[key]
        X_test = test_split[key]

        Y = X_train['Weekly_Sales']
        X_train = X_train.drop(['Weekly_Sales', 'Store', 'Dept'], axis=1)

        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            # Extract the current column and all previous columns
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-16:
                cols_to_drop.append(col_name)

        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)

        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

    # Post prediction adjustment for fold_5
    if is_fold_5:
        test_pred = postprocess(test_pred)

    # Join predictions with original test data
    result = test.merge(test_pred, on=["Store", "Dept", "Date"], how="left")

    # Handle missing predictions and round results
    result["Weekly_Pred"] = result["Weekly_Pred"].fillna(0)

    # Ensure we have all these colummns
    result = result[["Store", "Dept", "Date", "IsHoliday", "Weekly_Pred"]]

    # Export results
    result.to_csv(pred_file_path, index=False)
    end_time = time.time()
    print(pred_file_path)
    print(f"Execution time: {end_time - start_time:.4f} seconds")


process()