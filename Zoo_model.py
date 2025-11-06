# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

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

# # Convert to torch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Note: long for CrossEntropyLoss
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # === Define NN with 2 Output Neurons ===
# class SoftmaxNN(nn.Module):
#     def __init__(self, input_dim):
#         super(SoftmaxNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.relu2 = nn.ReLU()
#         self.output = nn.Linear(64, 2)  # 2 output classes

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.output(x)  # logits for CrossEntropyLoss
#         return x

# input_dim = X_train.shape[1]
# model = SoftmaxNN(input_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # === Training Loop ===
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# # === Evaluation ===
# model.eval()
# with torch.no_grad():
#     test_logits = model(X_test_tensor)
#     y_pred = torch.argmax(test_logits, dim=1).numpy()
#     acc = accuracy_score(y_test, y_pred)
#     print("\n📊 Classification Report:")
#     print(classification_report(y_test, y_pred))
#     print(f"✅ Accuracy: {acc * 100:.2f}%")

# # === Save the New Model ===
# torch.save(model.state_dict(), "pytorch_nn_model_softmax.pth")
# print("✅ Model saved as pytorch_nn_model_softmax.pth")


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
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

# === Define Model ===
class SoftmaxNN(nn.Module):
    def __init__(self, input_dim):
        super(SoftmaxNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)

input_dim = X_train.shape[1]
model = SoftmaxNN(input_dim)

# === Wrap with ART Classifier ===
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    input_shape=(input_dim,),
    nb_classes=2,
    clip_values=(X_train.min(), X_train.max())
)

# === Generate ZOO Adversarial Samples for Training ===
zoo = ZooAttack(classifier=classifier, max_iter=10, batch_size=1, nb_parallel=10)
print("Generating ZOO adversarial samples for training...")
X_train_subset = X_train[:2000].astype(np.float32)  # Limit for speed
X_train_zoo = zoo.generate(X_train_subset)
y_train_subset = y_train[:2000]

# === Combine Clean and Adversarial Data ===
X_combined = np.vstack([X_train_subset, X_train_zoo])
y_combined = np.hstack([y_train_subset, y_train_subset])  # Same labels

# === Convert to Tensors ===
X_tensor = torch.tensor(X_combined, dtype=torch.float32)
y_tensor = torch.tensor(y_combined, dtype=torch.long)

# === Train Model ===
epochs = 10
for epoch in range(epochs):
    model.train()
    outputs = model(X_tensor)
    loss = nn.CrossEntropyLoss()(outputs, y_tensor)
    classifier._optimizer.zero_grad()
    loss.backward()
    classifier._optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# === Evaluate on Clean Test Data ===
model.eval()
with torch.no_grad():
    test_logits = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred = torch.argmax(test_logits, dim=1).numpy()
    acc = accuracy_score(y_test, y_pred)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {acc * 100:.2f}%")

# === Save Retrained Model ===
torch.save(model.state_dict(), "pytorch_nn_model_softmax_zoo_trained.pth")
print("✅ Retrained model saved as pytorch_nn_model_softmax_zoo_trained.pth")
