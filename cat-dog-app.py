import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from pathlib import Path


# Page settings
st.set_page_config(
    page_title="Cat vs Dog AI",
    page_icon="🐾",
    layout="wide"
)


# Luxury Black + Gold Theme
st.markdown("""
<style>

.stApp {
    background-color: #080808;
}

h1 {
    color: #D4AF37 !important;
    text-align: center;
    font-size: 55px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #D4AF37 !important;
}

p, label {
    color: #F5F5F5 !important;
}

[data-testid="stFileUploader"] {
    background-color: #111111;
    border: 1px solid #D4AF37;
    border-radius: 15px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background-color: #111111;
}

.stProgress > div > div > div > div {
    background-color: #D4AF37;
}

</style>
""", unsafe_allow_html=True)


# Title
st.title("🐾 CAT VS DOG 🐾")

st.markdown(
    "<p style='text-align:center; color:#D4AF37;'>"
    "AI Image Classification using Convolutional Neural Network"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# Load model
@st.cache_resource
def load_cnn_model():

    model_path = Path(__file__).parent / "cat_dog_cnn.h5"

    return load_model(str(model_path))


try:
    model = load_cnn_model()

except Exception:
    st.error("❌ Model could not be loaded.")
    st.info(
        "Make sure cat_dog_cnn.h5 is in the same folder as this app file."
    )
    st.stop()


# Two columns
col1, col2 = st.columns(2)


# Upload section
with col1:

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a Cat or Dog image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:

        st.info("Please upload an image to start prediction.")


# Prediction section
with col2:

    st.subheader("✨ Prediction")

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        st.image(
            img,
            caption="Uploaded Image",
            use_container_width=True
        )

        # Resize image
        img = img.resize((128, 128))

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        score = float(prediction[0][0])


        # Cat = 0
        # Dog = 1

        if score >= 0.5:

            result = "🐶 DOG"

            confidence = score * 100

        else:

            result = "🐱 CAT"

            confidence = (1 - score) * 100


        st.success(result)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )

    else:

        st.write("🐱  CAT")
        st.write("🐶  DOG")

        st.caption(
            "The CNN model will predict the uploaded image."
        )


st.divider()


# Model information
st.subheader("🧠 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Model", "CNN")

with info2:
    st.metric("Input Size", "128 × 128")

with info3:
    st.metric("Classes", "Cat / Dog")


st.divider()

st.caption("🐾 Cat vs Dog Image Classification | Deep Learning Project")