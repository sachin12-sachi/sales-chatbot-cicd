# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dummy data (replace with your own Excel path or logic)
data = {
    "feature1": [1, 2, 3, 4],
    "feature2": [10, 20, 30, 40],
    "target": [100, 200, 300, 400]
}
df = pd.DataFrame(data)

# Model training
X = df[["feature1", "feature2"]]
y = df["target"]
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")
