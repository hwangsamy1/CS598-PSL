import pandas as pd
import time

from sklearn.linear_model import LogisticRegression

num_splits = 5

log_reg = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.1,
    C=10,
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
train_file_path = f'./train.csv'
test_file_path = f'./test.csv'

# Load data
X_train = pd.read_csv(train_file_path).iloc[:, 3:]
y_train = pd.read_csv(train_file_path).iloc[:, 1]

X_test = pd.read_csv(test_file_path).iloc[:, 2:]

log_reg.fit(X_train, y_train)

y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

X_test_labels = pd.read_csv(test_file_path).iloc[:, 0]

df = pd.DataFrame({
    'id': X_test_labels,
    'prob': y_pred_proba
})

df.to_csv('mysubmission.csv', index=False)