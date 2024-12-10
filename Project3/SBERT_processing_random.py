import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

random.seed(42)

test_file_path = f'./F24_Proj3_data/split_1/test.csv'
test_y_file_path = f'./F24_Proj3_data/split_1/test_y.csv'

test = pd.read_csv(test_file_path)
test_y = pd.read_csv(test_y_file_path)

positive_reviews = test[test_y['sentiment'] == 1].sample(5, random_state=42)
negative_reviews = test[test_y['sentiment'] == 0].sample(5, random_state=42)
selected_reviews = pd.concat([positive_reviews, negative_reviews])

#print(selected_reviews)
test_id = selected_reviews['id'].reset_index(drop=True)
test_idandreview = test[['id', 'review']]
review = selected_reviews['review'].to_numpy()
model = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = model.encode(review)
Y_embeddings = selected_reviews.iloc[:, 2:]

lr_model = LinearRegression()
lr_model.fit(X_embeddings, Y_embeddings)
W_optimal = lr_model.coef_

intercept = lr_model.intercept_

#print("Optimal weights (W):", W_optimal)
#print("Intercept (bias):", intercept)
#print("X shape: ", X_embeddings.shape)
print("W shape: ", W_optimal.shape)

Y_approx_embeddings = np.dot(X_embeddings, W_optimal.T) + lr_model.intercept_

final = pd.concat([test_id, pd.DataFrame(Y_approx_embeddings).reset_index(drop=True)], axis=1)
final.to_csv('SBERT_embeddings_random_whole_review.csv', index=False)

# Tokenize all sample sentences.
nltk.download('punkt')
random_dict = {}
#model = SentenceTransformer("all-MiniLM-L6-v2")

for index, row in selected_reviews.iterrows():
    review_id = row['id']
    review_text = row['review']
    sentences = sent_tokenize(review_text)
    print("len(sentence):", len(sentences))

    X_embeddings = model.encode(sentences)
    #print(sentences)
    print("X_embeddings.shape:", X_embeddings.shape)

    Y_approx_embeddings = np.dot(X_embeddings, W_optimal.T) + lr_model.intercept_
    Y_approx_embeddings = pd.DataFrame(Y_approx_embeddings)
    print("Y_approx_embeddings.shape:", Y_approx_embeddings.shape)

    Y_approx_embeddings.insert(0, 'sentence', sentences)

    output_name = f'./SBERT_embedddings_sentences_{review_id}.csv'
    Y_approx_embeddings.to_csv(output_name, index=False)
