import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = np.array([1, 5, 9, 8, 15, 3, 1, 4, 3, 6, 1
                 , 3])

X = []
y = []

for i in range(1, len(data) - 1):
    if data[i] > data[i - 1] and data[i] > data[i + 1]:
        X.append([data[i - 1], data[i + 1]])
        y.append(1)  # Hill
    elif data[i] < data[i - 1] and data[i] < data[i + 1]:
        X.append([data[i - 1], data[i + 1]])
        y.append(0)  # Valley
   

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
