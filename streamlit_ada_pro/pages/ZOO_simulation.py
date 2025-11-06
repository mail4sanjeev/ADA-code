# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# import joblib
# import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from art.attacks.evasion import ZooAttack
# from art.estimators.classification import PyTorchClassifier
# from sklearn.metrics import classification_report, accuracy_score

# st.set_page_config(page_title="ZOO Attack Simulation", layout="wide")
# st.title("🧪 ZOO Adversarial Attack Simulation")

# uploaded_file = st.file_uploader("📂 Upload a preprocessed CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     df.dropna(inplace=True)

#     label_col = None
#     for col in df.columns:
#         if col.strip().lower() in ['label', 'class']:
#             label_col = col
#             break

#     if not label_col:
#         st.error("❌ No 'Label' or 'Class' column found.")
#         st.stop()

#     df[label_col] = LabelEncoder().fit_transform(df[label_col])
#     df[label_col] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

#     y = df[label_col].values
#     X = df.drop(label_col, axis=1).select_dtypes(include=['number']).values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # === Define PyTorch NN Model with 2 outputs ===
#     class SoftmaxNN(nn.Module):
#         def __init__(self, input_dim):
#             super(SoftmaxNN, self).__init__()
#             self.fc1 = nn.Linear(input_dim, 128)
#             self.relu1 = nn.ReLU()
#             self.fc2 = nn.Linear(128, 64)
#             self.relu2 = nn.ReLU()
#             self.output = nn.Linear(64, 2)  # 2 output classes

#         def forward(self, x):
#             x = self.relu1(self.fc1(x))
#             x = self.relu2(self.fc2(x))
#             x = self.output(x)
#             return x

#     input_dim = X_scaled.shape[1]
#     model = SoftmaxNN(input_dim)
#     model.load_state_dict(torch.load("pytorch_nn_model_softmax.pth", map_location=torch.device('cpu')))
#     model.eval()

#     feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
#     xgb = joblib.load("xgb_model.pkl")

#     classifier = PyTorchClassifier(
#         model=model,
#         loss=nn.CrossEntropyLoss(),
#         optimizer=torch.optim.Adam(model.parameters()),
#         input_shape=(input_dim,),
#         nb_classes=2,
#         clip_values=(X_scaled.min(), X_scaled.max())
#     )

#     def feature_squeeze(X, decimals=1):
#         return np.round(X * (10**decimals)) / (10**decimals)

#     # === Run ZOO Attack ===
#     st.success("✅ Starting ZOO attack on subset...")
#     X_subset = X_scaled[:10].astype(np.float32)
#     y_subset = y[:10]

#     zoo = ZooAttack(classifier=classifier, max_iter=10, batch_size=1, nb_parallel=50)

#     try:
#         X_zoo_adv = zoo.generate(X_subset)
#         st.success("🎯 ZOO adversarial samples generated.")

#         with torch.no_grad():
#             zoo_tensor = torch.tensor(X_zoo_adv, dtype=torch.float32)
#             squeezed_tensor = torch.tensor(feature_squeeze(X_zoo_adv), dtype=torch.float32)

#             zoo_features = feature_extractor(zoo_tensor).numpy()
#             squeezed_features = feature_extractor(squeezed_tensor).numpy()

#         y_pred_zoo = xgb.predict(zoo_features)
#         y_pred_squeezed = xgb.predict(squeezed_features)

#         suspicious = y_pred_zoo != y_pred_squeezed
#         flagged = np.sum(suspicious)

#         st.write("### 🔍 ZOO Attack Evaluation")
#         st.code(classification_report(y_subset, y_pred_zoo))
#         st.write(f"✅ Accuracy on ZOO adversarial samples: {accuracy_score(y_subset, y_pred_zoo) * 100:.2f}%")
#         st.write(f"🔒 Feature Squeezing flagged {flagged}/{len(y_subset)} samples as suspicious ({(flagged / len(y_subset)) * 100:.2f}%)")

#         st.write("### 📉 Perturbation Visualization")
#         for i in range(len(X_subset)):
#             fig, ax = plt.subplots(figsize=(10, 3))
#             diff = X_zoo_adv[i] - X_subset[i]
#             ax.plot(X_subset[i], label='Original', linestyle='--', color='blue')
#             ax.plot(X_zoo_adv[i], label='Adversarial', color='red', alpha=0.7)
#             ax.plot(diff, label='Difference', color='green', linestyle=':')
#             ax.set_title(f"ZOO Perturbation - Sample {i}")
#             ax.legend()
#             ax.grid(True)
#             st.pyplot(fig)

#     except Exception as e:
#         st.error(f"❌ ZOO attack failed: {e}")


import streamlit as st
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ZooAttack
from art.defences.preprocessor import FeatureSqueezing

st.set_page_config(page_title="ZOO Attack Simulation", layout="wide")
st.title("ZOO Adversarial Attack Simulation")

uploaded_file = st.file_uploader(" Upload a preprocessed CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    label_col = None
    for col in df.columns:
        if col.strip().lower() in ['label', 'class']:
            label_col = col
            break

    if not label_col:
        st.error("No 'Label' or 'Class' column found.")
        st.stop()

    df[label_col] = LabelEncoder().fit_transform(df[label_col])
    df[label_col] = df[label_col].apply(lambda x: 0 if x == 0 else 1)

    y = df[label_col].values
    X = df.drop(label_col, axis=1).select_dtypes(include=['number']).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.output = nn.Linear(64, 2)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.output(x)
            return x

    input_dim = X_scaled.shape[1]
    model = SimpleNN(input_dim)
    model.load_state_dict(torch.load("pytorch_nn_model_softmax_zoo_trained.pth", map_location=torch.device('cpu')))
    model.eval()

    feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
    xgb = joblib.load("xgb_model.pkl")

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters()),
        input_shape=(input_dim,),
        nb_classes=2,
        clip_values=(X_scaled.min(), X_scaled.max())
    )

    bit_depths = [1, 2, 4]
    st.success("Generating ZOO Adversarial Samples (on last 10 samples)...")
    X_subset = X_scaled[-10:].astype(np.float32)
    y_subset = y[-10:]
    placeholder = st.empty()
    log = ""

    zoo = ZooAttack(classifier=classifier, max_iter=10, batch_size=1, nb_parallel=50)
    X_zoo_adv = zoo.generate(X_subset)

    with torch.no_grad():
        zoo_tensor = torch.tensor(X_zoo_adv, dtype=torch.float32)
        features = feature_extractor(zoo_tensor).numpy()
    y_pred = xgb.predict(features)

    for i in range(len(y_subset)):
        label = y_subset[i]
        pred = y_pred[i]
        is_suspicious = False
        for bit in bit_depths:
            fs = FeatureSqueezing(clip_values=(X_scaled.min(), X_scaled.max()), bit_depth=bit)
            squeezed_x, _ = fs(np.array([X_zoo_adv[i]]))
            with torch.no_grad():
                sq_feat = feature_extractor(torch.tensor(squeezed_x, dtype=torch.float32)).numpy()
            sq_pred = xgb.predict(sq_feat)[0]
            if pred != sq_pred:
                is_suspicious = True
                break

        log += f"[{i}] True Label: {'Attack' if label else 'Benign'} | Prediction: {'Attack' if pred else 'Benign'} | Suspicious: {is_suspicious}\n"
        placeholder.code(log)
        time.sleep(1)

    st.success(" ZOO Adversarial Simulation Completed.")


      # Mention accuracy
    with open("metrics.json", "r") as file:
        data = json.load(file)    

    accuracy = data.get("ZOO attack accuracy", "N/A")    

    st.metric("✅ Accuracy", accuracy)


    def plot_adversarial_diff(original, adversarial, sample_idx, title_prefix="ZOO"):
        fig, ax = plt.subplots(figsize=(12, 4))
        diff = adversarial - original
        ax.plot(original[sample_idx], label='Original', color='blue', linestyle='--')
        ax.plot(adversarial[sample_idx], label='Adversarial', color='red', alpha=0.7)
        ax.plot(diff[sample_idx], label='Difference', color='green', linestyle=':')
        ax.set_title(f'{title_prefix} Perturbation Visualization - Sample {sample_idx}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    st.markdown("# Adversarial Sample Visualizations")
    st.info("Charts below show how the input features were perturbed. Green line = difference between original & adversarial.")

    for i in range(len(X_subset)):
        st.markdown(f"### Sample {i+1}")
        st.markdown("""
        - **Blue Line**: Original input features  
        - **Red Line**: Adversarial features (after attack)  
        - **Green Line**: Change caused by attack (perturbation)
        """)
        plot_adversarial_diff(X_subset, X_zoo_adv, i, title_prefix="ZOO")
