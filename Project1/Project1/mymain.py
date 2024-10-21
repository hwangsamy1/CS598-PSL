import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

imbalanced_features = ["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                        "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"]

winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                     "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                     "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]


def preprocess_x_train(x_data):
    x_data = x_data.drop(columns=["PID", "Sale_Price"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.05, 0.05])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    scaler = StandardScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)

    return x_data, encoder, scaler


def preprocess_x_test(x_data, encoder, scaler):
    x_data = x_data.drop(columns=["PID"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=imbalanced_features)

    # Method 2: Winsorization of numerical features
    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.05, 0.05])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()

    one_hot_encoded = encoder.transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    x_data = pd.concat([x_data, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    x_data = scaler.transform(x_data)
    return x_data


def preprocess_lasso(x_data, y_data):
    # Method 4: Lasso to select features
    lasso_cv = LassoCV(cv=10, max_iter=10000)
    lasso_cv.fit(x_data, y_data)

    # Fit Lasso model with alpha_min
    lasso_model_min = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
    lasso_model_min.fit(x_data, y_data)
    lasso_coeff = lasso_model_min.coef_
    keep_indices = np.where(abs(lasso_coeff) > 0)
    keep_features = x_data.columns[keep_indices]
    #x_data.drop(x_data.columns[drop_indices], axis=1, inplace=True)

    return x_data[keep_features], keep_features


# Step 1A: Preprocess the training data
train = pd.read_csv("train.csv")
Y_train = np.log(train["Sale_Price"])
X_train, encoder, scaler = preprocess_x_train(train)
#X_train.to_csv("x_train_output.csv", index=False)
#X_train, features = preprocess_lasso(X_train, Y_train)


# Step 2A: Fit linear regression model (Lasso, Ridge, or Elasticnet penalty)
lasso_cv = LassoCV(cv=10, max_iter=10000)
lasso_cv.fit(X_train, Y_train)
lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
lasso_model.fit(X_train, Y_train)


# Step 2B: Fit tree model (randomForest or boosting tree)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, Y_train)


# Step 2A: Preprocess test data
test = pd.read_csv("test.csv")
X_test = preprocess_x_test(test, encoder, scaler)
#X_test = X_test[features]


# Step 2B: Generate predictions
preds_lasso = lasso_model.predict(X_test)
preds_rf = rf_model.predict(X_test)


# Step 3A: Save linear regression prediction to file
output_lasso = test[["PID"]].copy()
output_lasso.insert(1, "Sale_Price", preds_lasso.tolist())
output_lasso.to_csv("mysubmission1.txt", index=False)


# Step 3B: Save tree prediction to file
output_rf = test[["PID"]].copy()
output_rf.insert(1, "Sale_Price", preds_rf.tolist())
output_rf.to_csv("mysubmission2.txt", index=False)
