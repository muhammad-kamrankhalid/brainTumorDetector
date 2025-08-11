import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
from PIL import Image

# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tumor-detected {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .no-tumor {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'best_model.h5' is in the same directory as this script.")
        return None

# ===============================
# CLASS NAMES
# ===============================
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(uploaded_image):
    """Preprocess uploaded image for model prediction (RGB)"""
    img = Image.open(uploaded_image)
    img = tf.image.decode_image(file_contents, channels=3)  # force RGB

    # Ensure RGB format (model trained with RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))  # Match training size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    return img_array, img

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_tumor(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence, prediction[0]

# ===============================
# CONFIDENCE COLOR
# ===============================
def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# ===============================
# MAIN APP
# ===============================
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detector</h1>', unsafe_allow_html=True)

    # Sidebar Info
    st.sidebar.header("📋 About")
    st.sidebar.info("""
        This app uses a deep learning model (EfficientNetB0) to detect brain tumors from MRI scans.
        
        **Tumor Types:**
        - 🔴 Glioma
        - 🟡 Meningioma
        - 🟢 No Tumor
        - 🔵 Pituitary
    """)

    # Load Model
    model = load_model()
    if model is None:
        st.stop()

    # File Upload
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI brain scan image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear MRI brain scan image for analysis"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")

        with col2:
            st.subheader("🔍 Analysis Results")

            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait..."):
                    try:
                        # Preprocess & Predict
                        img_array, processed_img = preprocess_image(uploaded_file)
                        predicted_class, confidence, all_confidences = predict_tumor(model, img_array)

                        predicted_label = class_names[predicted_class]
                        confidence_pct = confidence * 100
                        is_tumor = predicted_label != 'notumor'
                        box_class = "tumor-detected" if is_tumor else "no-tumor"

                        # Prediction Box
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h3>📌 Prediction: {predicted_label.title()}</h3>
                            <h4 class="{get_confidence_color(confidence)}">
                                🔍 Confidence: {confidence_pct:.2f}%
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)

                        # Detailed Info
                        if is_tumor:
                            st.warning(f"⚠️ **{predicted_label.title()} detected** - Please consult a doctor.")
                        else:
                            st.success("✅ No tumor detected.")

                        # Probability Chart
                        st.subheader("📊 Detailed Probability Breakdown")
                        fig = px.bar(
                            x=all_confidences * 100,
                            y=[name.title() for name in class_names],
                            orientation='h',
                            title="Prediction Probabilities",
                            labels={'x': 'Probability (%)', 'y': 'Tumor Type'},
                            color=[('red' if name == predicted_label else 'lightblue') for name in class_names],
                            color_discrete_map={'red': '#ff4444', 'lightblue': '#87ceeb'}
                        )
                        fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                        st.info("⚕️ This tool is for educational purposes only and should not replace medical advice.")

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

    else:
        st.info("👆 Please upload an MRI scan image to begin.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧠 Brain Tumor Detector | Built with Streamlit & TensorFlow</p>
        <p><small>For educational purposes only</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

