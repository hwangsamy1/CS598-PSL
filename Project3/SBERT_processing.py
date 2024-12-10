import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
import numpy as np

test_file_path = f'./F24_Proj3_data/split_1/test.csv'
test_y_file_path = f'./F24_Proj3_data/split_1/test_y.csv'

test = pd.read_csv(test_file_path)
test_y = pd.read_csv(test_y_file_path)

test_id = test['id']
test_idandreview = test[['id', 'review']]
review = test_idandreview['review'].to_numpy()
model = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = model.encode(review)
Y_embeddings = test.iloc[:, 2:]

model = LinearRegression()
model.fit(X_embeddings, Y_embeddings)
W_optimal = model.coef_

intercept = model.intercept_

print("Optimal weights (W):", W_optimal)
print("Intercept (bias):", intercept)

print("X shape: ", X_embeddings.shape)
print("W shape: ", W_optimal.shape)

<<<<<<< HEAD
Y_approx_embeddings = np.dot(X_embeddings, W_optimal.T) + model.intercept_

print(Y_approx_embeddings)
final = pd.concat([test_id, pd.DataFrame(Y_approx_embeddings)], axis=1)

final.to_csv('SBERT_embeddings_whole_review.csv', index=False)
=======
Y_approx_embeddings = X_embeddings @ W_optimal + model.intercept_

print(Y_approx_embeddings)
final = pd.concat([test_idandreview, Y_approx_embeddings], axis=1)

final.to_csv('SBERT_embeddings.csv', index=False)
>>>>>>> parent of 52177d1 (SBERT script and embedding data)
