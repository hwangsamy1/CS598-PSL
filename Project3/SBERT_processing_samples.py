import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression

test_file_path = f'./F24_Proj3_data/split_1/test.csv'
test_y_file_path = f'./F24_Proj3_data/split_1/test_y.csv'

test = pd.read_csv(test_file_path)
test_y = pd.read_csv(test_y_file_path)

random.seed(42)
positive_reviews = test[test_y['sentiment'] == 1].sample(5, random_state=42)
negative_reviews = test[test_y['sentiment'] == 0].sample(5, random_state=42)
selected_reviews = pd.concat([positive_reviews, negative_reviews])

review = selected_reviews['review'].to_numpy()
model = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = model.encode(review)
Y_embeddings = selected_reviews.iloc[:, 2:]

model = LinearRegression()
model.fit(X_embeddings, Y_embeddings)
W_optimal = model.coef_

intercept = model.intercept_

#print("Optimal weights (W):", W_optimal)
#print("Intercept (bias):", intercept)

#print("X shape: ", X_embeddings.shape)
#print("W shape: ", W_optimal.shape)

Y_approx_embeddings = X_embeddings.dot(W_optimal.T) + model.intercept_
Y_out = pd.DataFrame(Y_approx_embeddings)

output = selected_reviews[['id', 'review']]
final = pd.concat([output.reset_index(drop=True), Y_out.reset_index(drop=True)], axis=1, ignore_index=True)

old_name0 = final.columns[0]
final.rename(columns={old_name0: 'id'})
old_name1 = final.columns[1]
final.rename(columns={old_name1: 'review'})

final.to_csv('SBERT_embeddings_random.csv', index=False)
