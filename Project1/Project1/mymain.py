import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import random

imbalanced_features = ["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                        "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"]

winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                     "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                     "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

scaler = StandardScaler()

random.seed(4999)

def preprocess_regression_train(x_data):
    x_data = x_data.drop(columns=["PID", "Sale_Price"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.01, 0.03])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    # Method 4: Scaler
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    return x_data, encoder, scaler


def preprocess_regression_test(x_data, encoder, scaler):
    x_data = x_data.drop(columns=["PID"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.01, 0.03])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    one_hot_encoded = encoder.transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    # Method 4: Scaler
    x_data = scaler.transform(x_data)
    return x_data


def preprocess_tree_train(x_data):
    x_data = x_data.drop(columns=["PID", "Sale_Price"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.01, 0.01])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    # Method 4: Scaler
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    return x_data, encoder, scaler


def preprocess_tree_test(x_data, encoder, scaler):
    x_data = x_data.drop(columns=["PID"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.01, 0.01])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    one_hot_encoded = encoder.transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    # Method 4: Scaler
    x_data = scaler.transform(x_data)
    return x_data


# Step 1A: Preprocess the linear regression training data
train = pd.read_csv("train.csv")
Y_train = np.log(train["Sale_Price"])
X_train, encoder, scaler = preprocess_regression_train(train)

# Step 1B: Fit linear regression model (Lasso, Ridge, or Elasticnet penalty)
lasso_cv = LassoCV(cv=10, max_iter=10000)
lasso_cv.fit(X_train, Y_train)
lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
lasso_model.fit(X_train, Y_train)

# Step 1C: Preprocess the linear regression test data
test = pd.read_csv("test.csv")
X_test = preprocess_regression_test(test, encoder, scaler)

# Step 1D: Generate linear regression predictions
preds_lasso = lasso_model.predict(X_test)

# Step 1E: Save linear regression prediction to file
output_lasso = test[["PID"]].copy()
output_lasso.insert(1, "Sale_Price", preds_lasso.tolist())
output_lasso.to_csv("mysubmission1.txt", index=False)


# Step 2A: Preprocess the tree training data
train = pd.read_csv("train.csv")
Y_train = np.log(train["Sale_Price"])
X_train, encoder, scaler = preprocess_tree_train(train)

# Step 2B: Fit tree model (randomForest or boosting tree)
#rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
#rf_model.fit(X_train, Y_train)
xg_model = xgb.XGBRegressor(n_estimators=6000, max_depth=6,
                            eta=0.05, subsample=0.5)
xg_model.fit(X_train, Y_train)

# Step 2C: Preprocess the tree test data
test = pd.read_csv("test.csv")
X_test = preprocess_tree_test(test, encoder, scaler)

# Step 2D: Generate the tree predictions
#reds_rf = rf_model.predict(X_test)
preds_xg = xg_model.predict(X_test)

# Step 2E: Save tree prediction to file
output_xg = test[["PID"]].copy()
output_xg.insert(1, "Sale_Price", preds_xg.tolist())
output_xg.to_csv("mysubmission2.txt", index=False)
