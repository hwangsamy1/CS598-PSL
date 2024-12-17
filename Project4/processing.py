import pandas as pd

rating_matrix = pd.read_csv("I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv", index_col=0)
s100 = rating_matrix.iloc[:, :100]
s100.to_csv('S100.csv')