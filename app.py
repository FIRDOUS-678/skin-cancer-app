import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("best_skin_cancer_model.keras")
    with open("class_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_assets()
IMG_SIZE = info["img_size"]
NAMES = info["class_names"]
CODES = info["class_codes"]

st.title("🩺 Skin Lesion Classifier")
st.caption(f"Model: {info['best_model']} · trained on HAM10000 (7 lesion classes)")
st.warning("Educational demo only. This is NOT a medical diagnosis — please consult a dermatologist.")

file = st.file_uploader("Upload a dermatoscopic / skin lesion image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = tf.image.resize(np.array(img), (IMG_SIZE, IMG_SIZE)).numpy().astype("float32")[None, ...]
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    top = order[0]
    st.success(f"Prediction: **{NAMES[top]}** ({CODES[top]}) — confidence {probs[top] * 100:.1f}%")

    st.subheader("Class probabilities")
    st.bar_chart({NAMES[i]: float(probs[i]) for i in order})

    if CODES[top] in ("mel", "bcc", "akiec"):
        st.error("The model flags a potentially malignant / pre-malignant lesion. Please see a doctor.")

with st.expander("About this model"):
    st.json(info["test_metrics"])
