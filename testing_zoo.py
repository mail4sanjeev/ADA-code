import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from art.attacks.evasion import ZooAttack
from art.estimators.classification import PyTorchClassifier

# === Load Dataset ===
df = pd.read_csv(r'C:\Users\hp\Desktop\ADA\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')
df.dropna(inplace=True)
df['Label'] = LabelEncoder().fit_transform(df['Label'])
df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)
df = df.select_dtypes(include=['number'])

X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Define PyTorch Model (2 outputs for softmax) ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 2)  # 2-class output

    def forward(self, x):
        return self.output(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
model.load_state_dict(torch.load("pytorch_nn_model_softmax.pth"))
model.eval()

# === Wrap with ART Classifier ===
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(input_dim,),
    nb_classes=2,
    clip_values=(X_test.min(), X_test.max())
)

# === ZOO Attack (on small subset) ===
X_subset = X_test[:10].astype(np.float32)
y_subset = y_test[:10]

attack = ZooAttack(classifier=classifier, max_iter=10, batch_size=1, nb_parallel=5)
X_zoo_adv = attack.generate(X_subset)

# === Feature Squeezing
def feature_squeeze(X, decimals=1):
    return np.round(X * (10**decimals)) / (10**decimals)

# === Feature Extraction
with torch.no_grad():
    feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
    zoo_tensor = torch.tensor(X_zoo_adv, dtype=torch.float32)
    sq_tensor = torch.tensor(feature_squeeze(X_zoo_adv), dtype=torch.float32)

    zoo_features = feature_extractor(zoo_tensor).numpy()
    sq_features = feature_extractor(sq_tensor).numpy()

# === Predict
xgb = joblib.load("xgb_model.pkl")
y_pred_zoo = xgb.predict(zoo_features)
y_pred_sq = xgb.predict(sq_features)

# === Evaluation
print("📊 ZOO Attack Classification Report:")
print(classification_report(y_subset, y_pred_zoo))
print(f"⚠️ Accuracy on ZOO Adversarial Samples: {accuracy_score(y_subset, y_pred_zoo) * 100:.2f}%")

# === Feature Squeezing Suspicious Check
suspicious = y_pred_zoo != y_pred_sq
print(f"🔐 Feature Squeezing flagged {np.sum(suspicious)}/{len(y_subset)} samples as suspicious")

# === Visualization
def plot_adversarial_diff(original, adversarial, sample_idx):
    plt.figure(figsize=(12, 4))
    diff = adversarial - original
    plt.plot(original[sample_idx], label='Original', color='blue', linestyle='--')
    plt.plot(adversarial[sample_idx], label='Adversarial', color='red', alpha=0.7)
    plt.plot(diff[sample_idx], label='Difference', color='green', linestyle=':')
    plt.title(f'ZOO Perturbation Visualization - Sample {sample_idx}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Show Visualization
for i in range(len(X_subset)):
    plot_adversarial_diff(X_subset, X_zoo_adv, i)
