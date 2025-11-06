import streamlit as st
import torch
import json
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from art.defences.preprocessor import FeatureSqueezing

st.set_page_config(page_title="FGSM Attack Simulation", layout="wide")
st.title("🧪 FGSM Adversarial Attack Simulation")

uploaded_file = st.file_uploader("📂 Upload a preprocessed CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    # Detect label column
    label_col = None
    for col in df.columns:
        if col.strip().lower() in ['label', 'class']:
            label_col = col
            break

    if not label_col:
        st.error("❌ No 'Label' or 'Class' column found.")
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
            self.output = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.sigmoid(self.output(x))
            return x

    input_dim = X_scaled.shape[1]
    model = SimpleNN(input_dim)
    model.load_state_dict(torch.load("pytorch_nn_model.pth", map_location=torch.device('cpu')))
    model.eval()

    feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
    xgb = joblib.load("xgb_model.pkl")
    fs = FeatureSqueezing(clip_values=(X_scaled.min(), X_scaled.max()), bit_depth=4)

    st.success("✅ Simulation running on uploaded data...")
    placeholder = st.empty()
    log = ""

    samples = X_scaled[-10:].astype(np.float32)
    labels = y[-10:]
    adv_samples = []
    orig_samples = []

    for i in range(10):
        sample = samples[i]
        label = labels[i]
        orig_samples.append(sample)

        with torch.no_grad():
            orig_feat = feature_extractor(torch.tensor([sample], dtype=torch.float32)).numpy()
        pred = xgb.predict(orig_feat)[0]

        X_squeezed, _ = fs(np.array([sample]))
        with torch.no_grad():
            sq_feat = feature_extractor(torch.tensor(X_squeezed, dtype=torch.float32)).numpy()
        squeezed_pred = xgb.predict(sq_feat)[0]

        suspicious = pred != squeezed_pred
        adv_samples.append(X_squeezed[0])

        log += f"[{i}] True Label: {'Attack' if label else 'Benign'} | Prediction: {'Attack' if pred else 'Benign'} | Suspicious: {suspicious}\n"
        placeholder.code(log)
        time.sleep(0.5)

    st.success("🎉 Real-Time IDS Simulation Completed.")

    
    # Mention accuracy
    with open("metrics.json", "r") as file:
        data = json.load(file)    

    accuracy = data.get("FGSM attack accuracy", "N/A")    

    st.metric("✅ Accuracy", accuracy)



    # === Visualization of Perturbations ===
    st.markdown("---")
    st.subheader("📊 Visualization of Adversarial Perturbations")
    st.markdown("This section shows how the adversarial samples differ from the original inputs for better interpretability.")

    def plot_adversarial_diff(original, adversarial, sample_idx):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        diff = adversarial - original
        ax.plot(original, label='Original', linestyle='--', color='blue')
        ax.plot(adversarial, label='Adversarial', alpha=0.7, color='red')
        ax.plot(diff, label='Difference', linestyle=':', color='green')
        ax.set_title(f"FGSM Perturbation Visualization - Sample {sample_idx}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Feature Value")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    for i in range(3):
        plot_adversarial_diff(orig_samples[i], adv_samples[i], i)
