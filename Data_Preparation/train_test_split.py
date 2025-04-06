import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv('../customer_churn_processed.csv')

# Example: Assume last column is the target (you may update it accordingly)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split sets
X_train.to_csv('../X_train.csv', index=False)
X_test.to_csv('../X_test.csv', index=False)
y_train.to_csv('../y_train.csv', index=False)
y_test.to_csv('../y_test.csv', index=False)

print("Train-test split completed and files saved.")