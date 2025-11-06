import time
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
# from art.defences.detector.evasion import FeatureSqueezing
from art.defences.preprocessor import FeatureSqueezing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
df = pd.read_csv("C:/Users/hp/Desktop/ADA/CIC-DDoS2019 Dataset/cicddos2019_dataset.csv")
df.dropna(inplace=True)
df['Label'] = LabelEncoder().fit_transform(df['Label'])
df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)
df = df.select_dtypes(include=['number'])
X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x

input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
model.load_state_dict(torch.load("pytorch_nn_model.pth", map_location=torch.device('cpu')))
model.eval()
feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)

# Load XGBoost model
xgb = joblib.load("xgb_model.pkl")

# Feature Squeezing setup
fs = FeatureSqueezing(clip_values=(X_test.min(), X_test.max()), bit_depth=4)

# Simulate real-time intrusion detection
print("\n=== Real-Time IDS Simulation Started ===")
# === Real-time loop simulation ===
for i in range(10):  # simulate 10 samples
    sample = X_test[i]
    label = y_test[i]

    # Original prediction
    with torch.no_grad():
        orig_feat = feature_extractor(torch.tensor([sample], dtype=torch.float32)).numpy()
    pred = xgb.predict(orig_feat)[0]

    # Apply Feature Squeezing
    X_squeezed, _ = fs(np.array([sample]))  # squeeze the input

    # Prediction after squeezing
    with torch.no_grad():
        sq_feat = feature_extractor(torch.tensor(X_squeezed, dtype=torch.float32)).numpy()
    squeezed_pred = xgb.predict(sq_feat)[0]

    # Flag as suspicious if prediction changes
    suspicious = pred != squeezed_pred

    print(f"[{i}] True Label: {label} | Predicted: {pred} | Suspicious: {suspicious}")

print("✅ Real-Time IDS Simulation Completed.") 