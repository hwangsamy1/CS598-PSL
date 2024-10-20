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
      test_y = np.log(test_y)
      output_lasso = pd.read_csv("mysubmission1.txt")
      output_rf = pd.read_csv("mysubmission2.txt")

      # Step 3: Report RMSE for each model
      rmse_lasso = root_mean_squared_error(test_y, output_lasso)
      rmse_rf = root_mean_squared_error(test_y, output_rf)

      print(f"RMSE for Lasso Model: {rmse_lasso}")
      print(f"RMSE for Random Forest Model: {rmse_rf}")