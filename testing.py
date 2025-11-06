# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from art.attacks.evasion import FastGradientMethod, ZooAttack
# from art.estimators.classification import PyTorchClassifier

# # === Load Dataset ===
# df = pd.read_csv(r'C:\Users\hp\Desktop\ADA\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')
# df.dropna(inplace=True)
# df['Label'] = LabelEncoder().fit_transform(df['Label'])
# df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)
# df = df.select_dtypes(include=['number'])

# X = df.drop('Label', axis=1).values
# y = df['Label'].values

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # === Define PyTorch Model (logits output for ZOO) ===
# class SimpleNN(nn.Module):
#     def __init__(self, input_dim):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.relu2 = nn.ReLU()
#         self.output = nn.Linear(64, 1)  # No sigmoid

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.output(x)  # logits
#         return x

# input_dim = X_train.shape[1]
# model = SimpleNN(input_dim)
# model.load_state_dict(torch.load("pytorch_nn_model.pth"))
# model.eval()

# feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
# xgb = joblib.load("xgb_model.pkl")

# # === Wrap Model with ART Classifier (use logits loss) ===
# classifier = PyTorchClassifier(
#     model=model,
#     loss=nn.CrossEntropyLoss(),
#     optimizer=torch.optim.Adam(model.parameters()),
#     input_shape=(input_dim,),
#     nb_classes=2,
#     clip_values=(X_test.min(), X_test.max())
# )

# # === FGSM Attack ===
# print("\n⚔️ Running FGSM Attack...")
# fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
# X_test_adv = fgsm.generate(x=X_test.astype(np.float32))

# # === ZOO Attack ===
# print("\n⚔️ Running ZOO Attack (on subset)...")
# zoo = ZooAttack(classifier=classifier, max_iter=10, batch_size=1)
# X_subset = X_test[:10].astype(np.float32)
# y_subset = y_test[:10]
# X_zoo_adv = zoo.generate(X_subset)

# # === Feature Squeezing Function ===
# def feature_squeeze(X, decimals=1):
#     return np.round(X * (10**decimals)) / (10**decimals)

# # === Evaluation Function ===
# def evaluate_adversarial(X_adv, X_orig, y_true, attack_name):
#     with torch.no_grad():
#         feat = feature_extractor(torch.tensor(X_adv, dtype=torch.float32)).numpy()
#         feat_sq = feature_extractor(torch.tensor(feature_squeeze(X_adv), dtype=torch.float32)).numpy()
#     y_pred = xgb.predict(feat)
#     y_pred_sq = xgb.predict(feat_sq)
#     acc = accuracy_score(y_true, y_pred)
#     suspicious = (y_pred != y_pred_sq)
#     flagged = np.sum(suspicious)

#     print(f"\n📊 {attack_name} Classification Report:")
#     print(classification_report(y_true, y_pred))
#     print(f"🎯 Accuracy: {acc * 100:.2f}%")
#     print(f"🔒 Feature Squeezing flagged {flagged}/{len(y_true)} samples as suspicious ({(flagged/len(y_true))*100:.2f}%)")

# # === Run Evaluations ===
# evaluate_adversarial(X_test_adv, X_test, y_test, "FGSM")
# evaluate_adversarial(X_zoo_adv, X_subset, y_subset, "ZOO")

# # === Visualization Function ===
# def plot_adversarial_diff(original, adversarial, sample_idx, title_prefix="Adversarial"):
#     plt.figure(figsize=(12, 4))
#     diff = adversarial - original
#     plt.plot(original[sample_idx], label='Original', color='blue', linestyle='--')
#     plt.plot(adversarial[sample_idx], label='Adversarial', color='red', alpha=0.7)
#     plt.plot(diff[sample_idx], label='Difference', color='green', linestyle=':')
#     plt.title(f'{title_prefix} Perturbation Visualization - Sample {sample_idx}')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Feature Value')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # === Show visualization for first 3 FGSM samples
# print("\n📈 FGSM Visualizations:")
# for i in range(3):
#     plot_adversarial_diff(X_test, X_test_adv, i, title_prefix="FGSM")

# # === Show visualization for all ZOO samples (subset)
# print("\n📈 ZOO Visualizations:")
# for i in range(len(X_subset)):
#     plot_adversarial_diff(X_subset, X_zoo_adv, i, title_prefix="ZOO")


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

# === Load Dataset ===
df = pd.read_csv(r'C:\Users\hp\Desktop\ADA\CIC-DDoS2019 Dataset\cicddos2019_dataset.csv')
df.dropna(inplace=True)
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)
df = df.select_dtypes(include=['number'])

X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Define PyTorch Model ===
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
model.load_state_dict(torch.load("pytorch_nn_model.pth"))
model.eval()

# === ART Classifier for Adversarial Testing ===
classifier = PyTorchClassifier(
    model=model,
    loss=nn.BCELoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(input_dim,),
    nb_classes=2,
    clip_values=(X_test.min(), X_test.max())
)

# === FGSM Adversarial Attack ===
attack = FastGradientMethod(estimator=classifier, eps=0.1)
X_test_adv = attack.generate(x=X_test.astype(np.float32))

# === 🎯 Feature Squeezing Function ===
def feature_squeeze(X, decimals=1):
    return np.round(X * (10**decimals)) / (10**decimals)

# === Apply Feature Squeezing to Adversarial Samples ===
X_squeezed = feature_squeeze(X_test_adv, decimals=1)

# === Feature Extraction ===
with torch.no_grad():
    feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
    X_test_adv_tensor = torch.tensor(X_test_adv, dtype=torch.float32)
    X_squeezed_tensor = torch.tensor(X_squeezed, dtype=torch.float32)

    X_test_adv_features = feature_extractor(X_test_adv_tensor).numpy()
    X_squeezed_features = feature_extractor(X_squeezed_tensor).numpy()

# === Load XGBoost & Predict ===
xgb = joblib.load("xgb_model.pkl")
y_pred_adv = xgb.predict(X_test_adv_features)
y_pred_squeezed = xgb.predict(X_squeezed_features)

# === Accuracy and Evaluation ===
print("📊 Classification Report on FGSM Adversarial Samples:")
print(classification_report(y_test, y_pred_adv))

adv_accuracy = accuracy_score(y_test, y_pred_adv)
print(f"⚠️ Accuracy on FGSM Adversarial Samples: {adv_accuracy * 100:.2f}%")

# === 🔐 Feature Squeezing Defense Check ===
suspicious = (y_pred_adv != y_pred_squeezed)
suspicious_count = np.sum(suspicious)
print(f"🔒 Feature Squeezing flagged {suspicious_count}/{len(X_test_adv)} samples as suspicious ({(suspicious_count/len(X_test_adv))*100:.2f}%)")

# === Visualization of Perturbations ===
def plot_adversarial_diff(original, adversarial, sample_idx):
    plt.figure(figsize=(12, 4))
    diff = adversarial - original
    plt.plot(original[sample_idx], label='Original', color='blue', linestyle='--')
    plt.plot(adversarial[sample_idx], label='Adversarial', color='red', alpha=0.7)
    plt.plot(diff[sample_idx], label='Difference', color='green', linestyle=':')
    plt.title(f'FGSM Perturbation Visualization - Sample {sample_idx}')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Show Visualization
for i in range(3):
    plot_adversarial_diff(X_test, X_test_adv, i)


# # ZOO attacking code

# from art.attacks.evasion import ZooAttack
# from art.estimators.classification import PyTorchClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import numpy as np
# import torch

# # === Wrap the model with ART's PyTorchClassifier ===
# classifier = PyTorchClassifier(
#     model=model,
#     loss=nn.BCELoss(),
#     optimizer=torch.optim.Adam(model.parameters()),
#     input_shape=(input_dim,),
#     nb_classes=2,
#     clip_values=(X_train.min(), X_train.max())
# )

# # === Run ZOO Attack ===
# zoo = ZooAttack(
#     classifier=classifier,
#     max_iter=10,               # Keep small to start with
#     learning_rate=0.01,
#     binary_search_steps=1,
#     initial_const=0.01,
#     confidence=0.0,
#     targeted=False,
#     nb_parallel=10,
#     batch_size=1,
#     use_resize=False,
#     use_importance=False,
# )

# # 🧪 Select a small sample for ZOO (it's slow)
# X_subset = X_test[:10].astype(np.float32)
# y_subset = y_test[:10]

# print("⚙️ Generating ZOO adversarial examples...")
# X_zoo_adv = zoo.generate(X_subset)

# # === Extract features from adversarial inputs ===
# with torch.no_grad():
#     zoo_tensor = torch.tensor(X_zoo_adv, dtype=torch.float32)
#     zoo_features = feature_extractor(zoo_tensor).numpy()

# # === Predict with XGBoost ===
# y_pred_zoo = xgb.predict(zoo_features)

# # === Evaluate ===
# print("📊 ZOO Attack Classification Report:")
# print(classification_report(y_subset, y_pred_zoo))
# print(f"⚠️ Accuracy on ZOO adversarial samples: {accuracy_score(y_subset, y_pred_zoo)*100:.2f}%")


