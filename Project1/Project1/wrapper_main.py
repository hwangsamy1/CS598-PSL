import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error

script_path = os.path.abspath("mymain.py")

directory_to_check = os.getcwd()
directories = [os.path.abspath(x[0]) for x in os.walk(directory_to_check) if "fold" in x[0]]

for i in directories:
      os.chdir(i)         # Change working Directory
      process = subprocess.run(["python", script_path])

      test_y = pd.read_csv("test_y.csv")
      output_lasso = pd.read_csv("mysubmission1.txt")
      output_rf = pd.read_csv("mysubmission2.txt")

      lasso_y_all = pd.merge(test_y, output_lasso, on="PID", how='outer')
      rf_y_all = pd.merge(test_y, output_rf, on="PID", how='outer')


      # Step 3: Report RMSE for each model
      rmse_lasso = root_mean_squared_error(lasso_y_all["Sale_Price_x"], lasso_y_all["Sale_Price_y"])
      rmse_rf = root_mean_squared_error(rf_y_all["Sale_Price_x"], rf_y_all["Sale_Price_y"])

      print(os.path.split(os.getcwd())[1])
      print(f"   RMSE for Lasso Model: {rmse_lasso}")
      print(f"   RMSE for Random Forest Model: {rmse_rf}")