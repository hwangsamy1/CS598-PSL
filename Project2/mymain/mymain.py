import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import patsy
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks
    return data


def svd(train_data):
    # some random value
    d = 5
    final_data = []
    departments = train_data['Dept'].unique()

    for dept in departments:
        filtered_train = train_data[train_data['Dept'] == dept]

        selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]

        train_dept_ts = selected_columns.pivot(index='Store', columns='Date', values='Weekly_Sales').reset_index()

        X_train = train_dept_ts.iloc[:, 1:]
        X_train = X_train.to_numpy()
        # fix a bunch of nans
        X_train = np.nan_to_num(X_train)

        store_mean = np.mean(X_train, axis=0)

        X_centered = X_train - store_mean
        U, D, VT = np.linalg.svd(X_centered, full_matrices=False)
        # Take first d components and fill rest of the diag with 0s
        D[d:] = 0
        X_bar = (U @ np.diag(D) @ VT) + store_mean

        # need to add some logic to rebuild the train data with column labels and such
        # concat it into final_data
        # print(X_bar)


def process(train_file_path, test_file_path, pred_file_path):
    print(train_file_path)
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    train_svd = svd(train)
    test_pred = pd.DataFrame()

    train_pairs = train_svd[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how='inner', on=['Store', 'Dept'])

    train_split = unique_pairs.merge(train_svd, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)
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

    result = test.merge(test_pred, on=["Store", "Dept", "Date"], how="left")
    result["Weekly_Pred"] = result["Weekly_Pred"].fillna(0)

    # Ensure we have all these colummns
    result = result[["Store", "Dept", "Date", "IsHoliday", "Weekly_Pred"]]

    result.to_csv(pred_file_path, index=False)

num_folds = 10

for i in range(num_folds):
    train_file_path = f'./Proj2_Data/fold_{i+1}/train.csv'
    test_file_path = f'./Proj2_Data/fold_{i+1}/test.csv'
    pred_file_path = f'./Proj2_Data/fold_{i+1}/mypred.csv'

    process(train_file_path, test_file_path, pred_file_path) 
