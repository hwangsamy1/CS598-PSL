
# Step 0: load necessary Python packages
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats.mstats import winsorize

# Step 1A: Preprocess the training data
train = pd.read_csv("train.csv")

## Select features to remove.  PID is one of them.
### Method 1: Visually exclude imbalanced categorical features that represent only a tiny fraction of samples
train = train.drop(columns=["PID", "Street", "Utilities", "Condition_2", "Roof_Matl", "Heating", "Pool_QC",
                            "Misc_Feature", "Low_Qual_Fin_SF", "Pool_Area", "Longitude", "Latitude"])


### Method 2: Winsorization of numerical features that ought to have a ceiling
winsorize_arr = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                 "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF",
                 "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]

for feat in winsorize_arr:
    train[feat] = winsorize(train[feat], limits=[0.05, 0.95])
#Probably need to include before and after graphs

print(train)

### Method 3: Lasso to manage categorical features



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

