# ✅ 1. Home Page
# File: pages/1_Home.py

import streamlit as st

st.set_page_config(page_title="Real-Time IDS", layout="wide")
st.title("🔐 Real-Time Intrusion Detection System using Adversarial ML")

st.markdown("""
### Project Overview
This project detects **real-time network intrusions** even under adversarial ML attacks (e.g., FGSM).

**Key Components:**
- 🧠 Hybrid Model (Neural Network + XGBoost)
- 🔐 FGSM adversarial attack detection
- 🛡️ Feature Squeezing defense

### Why This Model?
- ✅ Accurate detection
- ✅ Defense against small adversarial changes
- ✅ Ready for real-time deployment
""")

# ✅ 2. Upload/Test Page
# File: pages/2_Upload_Test.py

import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title("📂 Upload and Test Network Data")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    df = df.select_dtypes(include=['number'])

    if 'Label' in df.columns:
        y = df['Label'].values
        X = df.drop('Label', axis=1).values
    else:
        y = None
        X = df.values

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

    with torch.no_grad():
        feature_extractor = nn.Sequential(model.fc1, model.relu1, model.fc2, model.relu2)
        features = feature_extractor(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    xgb = joblib.load("xgb_model.pkl")
    predictions = xgb.predict(features)

    df_results = pd.DataFrame({"Prediction": predictions})
    if y is not None:
        df_results["Actual"] = y

    st.write("### Results:")
    st.dataframe(df_results.head(20))

# ✅ 3. Live Simulation Page
# File: pages/3_Live_Simulation.py

import time

st.title("⏱️ Real-Time Simulation")
st.write("Simulating live intrusion detection...")

# Placeholder logic for real-time simulation
test_data = [
    {"True Label": 1, "Prediction": 1, "Suspicious": True},
    {"True Label": 0, "Prediction": 0, "Suspicious": False},
    {"True Label": 1, "Prediction": 1, "Suspicious": False},
]

for i, row in enumerate(test_data):
    st.write(f"[{i}] True Label: {row['True Label']} | Prediction: {row['Prediction']} | Suspicious: {row['Suspicious']}")
    time.sleep(1)

# ✅ 4. Model Insight Page
# File: pages/4_Model_Insight.py


st.title("🧠 Model Insight")

st.markdown("""
### Hybrid Model Architecture
- Neural Network for deep feature extraction
- XGBoost for final classification

### FGSM Attack
- Fast Gradient Sign Method perturbs input slightly
- Can fool weak models

### Feature Squeezing
- Reduces feature precision (bit depth)
- Makes adversarial noise ineffective

### Architecture Diagram (You can upload image or diagram below):
""")

st.image("model_architecture.png", caption="Model Architecture", use_column_width=True)

# ✅ 5. Accuracy & Metrics Page
# File: pages/5_Accuracy_Metrics.py


import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Accuracy & Metrics")

st.markdown("""
### Sample Evaluation:
- Accuracy on clean data: 98%
- Accuracy on FGSM data: 86%
- Suspicious flag rate (Feature Squeezing): 0.10%
""")

fig, ax = plt.subplots()
labels = ['Clean', 'Adversarial']
accuracies = [98, 86]
ax.bar(labels, accuracies, color=['green', 'red'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
st.pyplot(fig)

# ✅ 6. Contact Page
# File: pages/6_Contact.py


st.title("📬 Contact / About")

st.markdown("""
### Developed By:
**Mohd Saif**

- GitHub: [github.com/yourprofile](https://github.com/yourprofile)
- Email: your.email@example.com
- Project Report: [Link to PDF or Google Drive]
""")
