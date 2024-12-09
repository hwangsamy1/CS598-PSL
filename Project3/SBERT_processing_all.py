import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression

test_file_path = f'./F24_Proj3_data/split_1/test.csv'
test_y_file_path = f'./F24_Proj3_data/split_1/test_y.csv'

test = pd.read_csv(test_file_path)
test_y = pd.read_csv(test_y_file_path)

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

Y_approx_embeddings = X_embeddings.dot(W_optimal.T) + model.intercept_
Y_out = pd.DataFrame(Y_approx_embeddings)

final = pd.concat([test_idandreview['id'], Y_out], axis=1)

final.to_csv('SBERT_embeddings_id.csv', index=False)
