# # Adaptive Defense Mechanism for Real-Time IDS
# # Complete Implementation (Hinglish Explanation Included)

# # === 1. Load Libraries ===
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import SklearnClassifier
# from art.defences.trainer import AdversarialTrainer
# import warnings
# warnings.filterwarnings('ignore')

# # === 2. Load Dataset ===
# # (Yahan aap CSV file ka path denge -download from CIC website)
# df = pd.read_csv('C:\\Users\\hp\Desktop\\ADA\\CIC-DDoS2019 Dataset\\cicddos2019_dataset.csv')  # or CIC-DDoS-2019.csv

# # === 3. Preprocessing ===
# df.dropna(inplace=True)  # missing values hatao
# label_encoder = LabelEncoder()
# df['Label'] = label_encoder.fit_transform(df['Label'])  # string labels ko numbers mein

# # Features aur Labels split karo
# X = df.drop('Label', axis=1)
# y = df['Label']

# # --- Fix: Only keep numeric columns ---
# # Drop all non-numeric columns except 'Label'
# df_numeric = df.select_dtypes(include=['number'])  # only numeric columns

# # Ensure 'Label' is present (some datasets drop it)
# if 'Label' not in df_numeric.columns:
#     df_numeric['Label'] = df['Label']

# # Separate features and target  
# X = df_numeric.drop('Label', axis=1)
# y = df_numeric['Label']

# # Normalize numeric features
# global_scaler = StandardScaler()
# X = global_scaler.fit_transform(X)

# # Split into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === 4. Build Hybrid/Ensemble Model ===
# rf = RandomForestClassifier()
# xgb = XGBClassifier(verbosity=0)
# lr = LogisticRegression()

# ensemble_model = VotingClassifier(estimators=[
#     ('rf', rf),
#     ('xgb', xgb),
#     ('lr', lr)
# ], voting='soft')

# ensemble_model.fit(X_train, y_train)

# # === 5. Evaluate on Clean Test Data ===
# y_pred_clean = ensemble_model.predict(X_test)
# print("Clean Accuracy:", accuracy_score(y_test, y_pred_clean))
# print(classification_report(y_test, y_pred_clean))

# # === 6. Adversarial Attack (FGSM) ===
# wrapped_model = SklearnClassifier(model=ensemble_model)
# attack = FastGradientMethod(estimator=wrapped_model, eps=0.1)

# # Generate adversarial examples
# X_test_adv = attack.generate(x=X_test)


# # Test model on adversarial data
# y_pred_adv = ensemble_model.predict(X_test_adv)
# print("Adversarial Accuracy:", accuracy_score(y_test, y_pred_adv))
# print(classification_report(y_test, y_pred_adv))

# # === 7. Adversarial Training (Defense Strategy) ===
# adversarial_trainer = AdversarialTrainer(estimator=wrapped_model, attacks=attack, ratio=0.5)
# adversarial_trainer.fit(X_train, y_train)

# # Re-evaluate
# y_pred_defense = ensemble_model.predict(X_test_adv)
# print("Post-defense Accuracy (on adversarial):", accuracy_score(y_test, y_pred_defense))
# print(classification_report(y_test, y_pred_defense))

# # === 8. Confusion Matrix ===
# print("Confusion Matrix (Adversarial):")
# print(confusion_matrix(y_test, y_pred_adv))


# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === 1. Load & Preprocess Dataset ===
df = pd.read_csv(r'C:\Users\hp\Desktop\ADA\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')
df.dropna(inplace=True)

# Encode string labels to numbers
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Convert multi-class to binary: 0 for benign, 1 for attack
df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)

# Keep only numeric features
df_numeric = df.select_dtypes(include=['number'])

# Split features and labels
X = df_numeric.drop('Label', axis=1)
y = df_numeric['Label']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # For BCELoss
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# === 2. Define Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()  # Required for BCELoss

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x

input_dim = X_train.shape[1]
model = SimpleNN(input_dim)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 3. Train Neural Network ===
for epoch in range(10):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === 4. Feature Extraction for XGBoost ===
with torch.no_grad():
    model.eval()
    hidden_layer = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
    X_train_features = hidden_layer(X_train_tensor).numpy()
    X_test_features = hidden_layer(X_test_tensor).numpy()

# === 5. Train XGBoost ===
xgb = XGBClassifier(verbosity=0)
xgb.fit(X_train_features, y_train)

# === 6. Evaluate
y_pred = xgb.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Hybrid Model Accuracy (NN features + XGBoost): {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 7. Save Models ===
torch.save(model.state_dict(), "pytorch_nn_model.pth")
joblib.dump(xgb, "xgb_model.pkl")
print("\n✅ Models saved as: pytorch_nn_model.pth, xgb_model.pkl")


# # Code for testing using FGSM

# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import KerasClassifier

# # Re-wrap your original NN
# from tensorflow.keras.models import Model
# art_classifier = KerasClassifier(model=nn_model, clip_values=(X_train.min(), X_train.max()))

# # Generate adversarial examples (FGSM)
# attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
# X_test_adv = attack.generate(X_test)

# # Get adversarial features from NN
# X_test_adv_features = intermediate_model.predict(X_test_adv)

# # Test XGBoost on adversarial features
# y_pred_adv = xgb.predict(X_test_adv_features)

# print("\n📊 Classification Report (Adversarial Data - FGSM):")
# print(classification_report(y_test, y_pred_adv))

