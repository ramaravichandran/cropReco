import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")  # Make sure this file is in the same folder

# Separate features (N, P, K, temperature, humidity, pH, rainfall) and target ('label' - crop name)
X = data.drop('label', axis=1)   # Features
y = data['label']                # Target (crop name)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model training complete. Saved as model.pkl")
