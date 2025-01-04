import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example 1D data (could be elevation values, etc.)
data = np.array([1, 3, 7, 8, 5, 3, 1, 4, 7, 6, 2, 3])

# Create labels: 1 for hill, 0 for valley
labels = np.zeros_like(data)

# Labeling: Local max as hill (1), local min as valley (0)
for i in range(1, len(data) - 1):
    if data[i] > data[i - 1] and data[i] > data[i + 1]:
        labels[i] = 1  # Hill
    elif data[i] < data[i - 1] and data[i] < data[i + 1]:
        labels[i] = 0  # Valley

# Create feature set: Use value of neighbors as features
X = np.array([[data[i-1], data[i+1]] if i > 0 and i < len(data)-1 else [0, 0] for i in range(len(data))])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels[1:-1], test_size=0.2, random_state=42)  # exclude first/last points

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
