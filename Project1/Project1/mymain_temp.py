import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Step 1A: Preprocess the training data
train = pd.read_csv("train.csv")
X_train = train.drop(columns=["PID", "Sale_Price"])
Y_train = train["Sale_Price"]

# Method 1: Remove imbalanced categorical features
X_train = X_train.drop(columns=["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                                "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"])

# Method 2: Winsorization of numerical features
winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                 "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                 "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

for feat in winsorize_arr:
    X_train[feat] = winsorize(X_train[feat], limits=[0.05, 0.95])

# Method 3: One-hot encode categorical features
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoded = encoder.fit_transform(X_train[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X_train = pd.concat([X_train, one_hot_df], axis=1)
X_train = X_train.drop(categorical_columns, axis=1)
X_train = X_train.fillna(0)

# Lasso to select features
lasso_alphas = np.logspace(-10, 1, 100)
lassocv = LassoCV(alphas=lasso_alphas, cv=10)
lassocv.fit(X_train, Y_train)

mean_mse = np.mean(lassocv.mse_path_, axis=1)
cv_alphas = lassocv.alphas_
min_idx = np.argmin(mean_mse)
alpha_min = cv_alphas[min_idx]

# Fit Lasso model with alpha_min
lasso_model_min = Lasso(alpha=alpha_min, max_iter=50000)
lasso_model_min.fit(X_train, Y_train)
lasso_coeff = lasso_model_min.coef_

# Step 1B: Fit Model 2 (RandomForest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Step 2A: Preprocess test data
test = pd.read_csv("test.csv")
test_y = pd.read_csv("test_y.csv").iloc[:, 0]
X_test = test.drop(columns=["PID"])

# Apply the same preprocessing steps to the test data
X_test = X_test.drop(columns=["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                              "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"])
for feat in winsorize_arr:
    X_test[feat] = winsorize(X_test[feat], limits=[0.05, 0.95])
one_hot_encoded_test = encoder.transform(X_test[categorical_columns])
one_hot_df_test = pd.DataFrame(one_hot_encoded_test, columns=encoder.get_feature_names_out(categorical_columns))
X_test = pd.concat([X_test, one_hot_df_test], axis=1)
X_test = X_test.drop(categorical_columns, axis=1)
X_test = X_test.fillna(0)

# Step 2B: Generate predictions
preds_lasso = lasso_model_min.predict(X_test)
preds_rf = rf_model.predict(X_test)

# Save predictions to files
np.savetxt("mysubmission1.txt", preds_lasso)
np.savetxt("mysubmission2.txt", preds_rf)

# Step 3: Report RMSE for each model
rmse_lasso = mean_squared_error(test_y, preds_lasso, squared=False)
rmse_rf = mean_squared_error(test_y, preds_rf, squared=False)

print(f"RMSE for Lasso Model: {rmse_lasso}")
print(f"RMSE for Random Forest Model: {rmse_rf}")
