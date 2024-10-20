import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


def preprocess_x(x_data):
    x_data = x_data.drop(columns=["PID"])

    # Method 1: Remove imbalanced categorical features
    x_data = x_data.drop(columns=["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                                    "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"])

    # Method 2: Winsorization of numerical features
    winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                     "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                     "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

    for feat in winsorize_arr:
        x_data[feat] = winsorize(x_data[feat], limits=[0.05, 0.95])

    # Method 3: One-hot encode categorical features
    categorical_columns = x_data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(x_data[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    x_data = pd.concat([X_train, one_hot_df], axis=1)
    x_data = x_data.drop(categorical_columns, axis=1)
    x_data = x_data.fillna(0)

    return x_data


def preprocess_lasso(x_data, y_data):
    # Method 4: Lasso to select features
    #lasso_alphas = np.logspace(-10, 1, 100)
    #lasso_cv = LassoCV(alphas=lasso_alphas, cv=10).fit(x_data, y_data)
    lasso_cv = LassoCV(cv=10, max_iter=10000)
    lasso_cv.fit(x_data, y_data)

    # mean_mse = np.mean(lasso_cv.mse_path_, axis=1)
    # cv_alphas = lasso_cv.alphas_
    # min_idx = np.argmin(mean_mse)
    # alpha_min = cv_alphas[min_idx]

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
X_train = train.drop(columns=["Sale_Price"])
Y_train = np.log(train["Sale_Price"])
X_train = preprocess_x(X_train)
X_train, features = preprocess_lasso(X_train, Y_train)


# Step 2A: Fit linear regression model (Lasso, Ridge, or Elasticnet penalty)
lasso_cv = LassoCV(cv=10, max_iter=10000)
lasso_cv.fit(X_train, Y_train)
lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=50000)
lasso_model.fit(X_train, Y_train)


# Step 2B: Fit tree model (randomForest or boosting tree)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)


# Step 2A: Preprocess test data
test = pd.read_csv("test.csv")
X_test = preprocess_x(test)
X_test = X_test[features]


# Step 2B: Generate predictions
preds_lasso = lasso_model.predict(X_test)
preds_rf = rf_model.predict(X_test)


# Step 3A: Save linear regression prediction to file
output_lasso = test["PID"].insert(2, "Sale_Price", preds_lasso, True)
np.savetxt("mysubmission1.txt", output_lasso)


# Step 3B: Save tree prediction to file
output_rf = test["PID"].insert(2, "Sale_Price", preds_rf, True)
np.savetxt("mysubmission2.txt", output_rf)



