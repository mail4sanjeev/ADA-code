import streamlit as st
import matplotlib.pyplot as plt

st.title("📊 Accuracy & Metrics")

st.markdown("""
### Sample Evaluation:
- Accuracy on clean data: 86%
- Accuracy on FGSM data: 79%
- accuracy on ZOO data : 60%
- Suspicious flag rate (Feature Squeezing): 0.10%
""")

fig, ax = plt.subplots()
labels = ['Clean', 'Adversarial']
accuracies = [98, 86]
ax.bar(labels, accuracies, color=['green', 'red'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
st.pyplot(fig)
