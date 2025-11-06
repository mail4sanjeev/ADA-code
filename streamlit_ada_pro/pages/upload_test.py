import streamlit as st
import json
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Upload & Test", layout="wide")
st.title("📂 Upload and Test Network Data")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    df = df.select_dtypes(include=['number', 'object'])  # Keep 'Class'

    # Detect 'Class' column for ground truth labels
    if 'Class' in df.columns:
        y = df['Class'].values
        df = df.drop(['Class'], axis=1)
    else:
        y = None

    df = df.select_dtypes(include=['number'])  # Final feature set
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
    pred_labels = ["Benign" if p == 0 else "Attack" for p in predictions]

    # Display results
    df_results = pd.DataFrame({"Prediction": pred_labels})
    if y is not None:
        df_results["Actual"] = y


    # Mention accuracy
    with open("metrics.json", "r") as file:
        data = json.load(file)    

    accuracy = data.get("Hybrid_model_accuracy", "N/A")    

    st.metric("✅ Accuracy", accuracy)


    # === Colored Table ===
    def highlight_result(row):
        if "Actual" in row and row["Prediction"] == row["Actual"]:
            return ['background-color: #d4edda; color: black'] * len(row)  # green
        else:
            return ['background-color: #f8d7da; color: black'] * len(row)  # red

    st.write("### 🧾 Prediction Results:")
    st.dataframe(df_results, use_container_width=True)
    # === Pie Chart ===
    st.write("### 📊 Prediction Distribution")
    labels = df_results["Prediction"].value_counts().index
    sizes = df_results["Prediction"].value_counts().values
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'])
    ax1.axis('equal')
    st.pyplot(fig1)

    # === Confusion Matrix ===
    if y is not None:
        st.write("### 📉 Confusion Matrix")
        cm = confusion_matrix(df_results["Actual"], df_results["Prediction"], labels=["Benign", "Attack"])
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig2)

    # === Download Button ===
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
