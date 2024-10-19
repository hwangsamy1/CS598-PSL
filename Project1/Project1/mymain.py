
# Step 0: load necessary Python packages
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Step 1A: Preprocess the training data
train = pd.read_csv("train.csv")
X_train = train.drop(columns=["PID", "Sale_Price"])
Y_train = train["Sale_Price"]

## Select features to remove.  PID is one of them.
### Method 1: Visually exclude imbalanced categorical features that represent only a tiny fraction of samples
X_train = X_train.drop(columns=["Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                            "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"])


### Method 2: Winsorization of numerical features that ought to have a ceiling
winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                 "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                 "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

for feat in winsorize_arr:
    X_train[feat] = winsorize(X_train[feat], limits=[0.05, 0.95])
#Probably need to include before and after graphs



### Method 3: Lasso to manage categorical features
#### Need to convert the categorical features into numerical features first via OneHotEncoder
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(X_train[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X_train = pd.concat([X_train, one_hot_df], axis=1)
X_train = X_train.drop(categorical_columns, axis=1)
X_train = X_train.fillna(0)
print(X_train)

#### Lasso.min
lasso_alphas = np.logspace(-10, 1, 100)
lassocv = LassoCV(alphas = lasso_alphas, cv = 10)
lassocv.fit(X_train, Y_train)

mean_mse = np.mean(lassocv.mse_path_, axis=1)
cv_alphas = lassocv.alphas_
min_idx = np.argmin(mean_mse)
alpha_min = cv_alphas[min_idx]

# Lasso with alpha_min
lasso_model_min = Lasso(alpha=alpha_min, max_iter=50000)
lasso_model_min.fit(X_train, Y_train)
lasso_coeff = lasso_model_min.coef_
print(lasso_coeff)

# Step 1B: Fit the two models
## Model 1 is based on linear regression models with Lasso or Ridge or Elasticnet penalty

## Model 2 is based on tree models such as randomForest or boosting tree.



# Step 2A: Preprocess test data,

test = pd.read_csv("test.csv")
test_y = pd.read_csv("test_y.csv")



# Step 2B: Generate predictions and save them into two files
    # mysubmission1.txt and mysubmission2.txt
    # Each file should contain predicts for the test data from one of your models.



#Step 3: Report RMSE for each fold

