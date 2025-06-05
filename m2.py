import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/vikas/Downloads/heart project/sample_heart_disease_data.csv")


# Split features and target
x = df.drop(columns='target', axis=1)
y = df['target']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Make a prediction
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 1:
    print("You are healthy")
else:
    print("You are unhealthy")
