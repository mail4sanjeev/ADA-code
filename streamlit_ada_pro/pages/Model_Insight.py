import streamlit as st

st.title("🧠 Model Insight")

st.markdown("""
### Hybrid Model Architecture
- Neural Network → feature extraction
- XGBoost → final classification

### FGSM Attack
- Fast Gradient Sign Method creates small input changes to fool models

### Feature Squeezing
- Reduces input precision to weaken adversarial noise

📌 You can visualize your model structure by uploading a diagram (e.g., `model_architecture.png`)
""")
