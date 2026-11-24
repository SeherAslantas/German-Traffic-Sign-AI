
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="SIGNAL AI | Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# SINIF İSİMLERİ
# --------------------------------------------------

labels = [
    "Speed limit (20 km/h)",
    "Speed limit (30 km/h)",
    "Speed limit (50 km/h)",
    "Speed limit (60 km/h)",
    "Speed limit (70 km/h)",
    "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)",
    "Speed limit (100 km/h)",
    "Speed limit (120 km/h)",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]

# --------------------------------------------------
# MODELİ YÜKLE
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "German_Traffic_Sign_CNN.keras"
    )

# --------------------------------------------------
# TASARIM
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c32);
    color: #f1f5f9;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172b4d, #243b63);
    border: 1px solid #334b70;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.section-title {
    color: #e2e8f0;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}

.info-card {
    background: #17243a;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #30435f;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #93c5fd;
    font-size: 15px;
    margin-bottom: 10px;
}

.info-card p {
    color: white;
    font-size: 21px;
    font-weight: 700;
    margin: 0;
}

.result-card {
    background: #17243a;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #3b82f6;
    margin-top: 15px;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# ÜST ALAN
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🚦 SIGNAL AI</h1>
    <p>German Traffic Sign Recognition System</p>
    <p>
        Deep Learning powered traffic sign classification
        using Convolutional Neural Networks.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL BİLGİLERİ
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>MODEL</h3>
        <p>CNN</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>CLASS COUNT</h3>
        <p>43 Traffic Signs</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>TEST ACCURACY</h3>
        <p>95.03%</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# GÖRSEL YÜKLEME
# --------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">📤 Upload Traffic Sign</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        file_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is not None:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Uploaded Traffic Sign",
                use_container_width=True
            )

            predict_button = st.button(
                "🔍 ANALYZE TRAFFIC SIGN"
            )
        else:
            st.error("Image could not be read.")
            predict_button = False
    else:
        st.info("Please upload a traffic sign image.")
        predict_button = False

# --------------------------------------------------
# TAHMİN
# --------------------------------------------------

with right:
    st.markdown(
        '<div class="section-title">🧠 AI Prediction</div>',
        unsafe_allow_html=True
    )

    if uploaded_file and predict_button and image is not None:

        try:
            model = load_model()

            resized = cv2.resize(image, (32, 32))
            normalized = resized.astype(np.float32) / 255.0
            input_image = np.expand_dims(normalized, axis=0)

            predictions = model.predict(
                input_image,
                verbose=0
            )[0]

            predicted_class = int(np.argmax(predictions))
            confidence = float(predictions[predicted_class])

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.success("Prediction completed!")

            st.markdown(
                f"### {labels[predicted_class]}"
            )

            st.metric(
                "AI Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

            st.markdown("</div>", unsafe_allow_html=True)

            # META GÖRSELİ
            meta_image_path = os.path.join(
                "Meta",
                f"{predicted_class}.png"
            )

            if os.path.exists(meta_image_path):
                st.write("**Reference Traffic Sign**")
                st.image(
                    meta_image_path,
                    width=180
                )

            # İLK 3 TAHMİN
            st.write("")
            st.markdown("### Top 3 Predictions")

            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:
                score = float(predictions[index])

                st.write(
                    f"**{labels[index]}** — {score * 100:.2f}%"
                )

                st.progress(score)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.markdown("""
        <div class="info-card">
            <h3>WAITING FOR IMAGE</h3>
            <p>AI analysis results will appear here.</p>
            <br>
            <p style="font-size:14px; color:#94a3b8;">
                Upload an image and click Analyze
                to start the prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# ALT BİLGİ
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<p style="text-align:center; color:#94a3b8;">
    SIGNAL AI | German Traffic Sign Recognition<br>
    Powered by CNN & TensorFlow
</p>
""", unsafe_allow_html=True)
