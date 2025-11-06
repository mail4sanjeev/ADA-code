import streamlit as st

st.set_page_config(page_title="Real-Time IDS", layout="wide")
st.title("🔐 Real-Time Intrusion Detection System using Adversarial ML")

st.markdown("""
### Project Overview
This project detects **real-time network intrusions** even under adversarial ML attacks.

**Key Components:**
- 🧠 Hybrid Model (NN + XGBoost)
- 🔐 FGSM adversarial attack detection
- 🔐 ZOO adversarial attack detection
- 🛡️ Feature Squeezing defense

### Why This Model?
- ✅ High detection accuracy
- ✅ Robust against adversarial inputs
- ✅ Supports real-time traffic simulation
""")
