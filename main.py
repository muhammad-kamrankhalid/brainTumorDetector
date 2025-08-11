import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'best_model.h5' is in the same directory as this script.")
        return None

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess image
# Function to preprocess image
# Function to preprocess image
def preprocess_image(uploaded_image):
    """Preprocess uploaded image for model prediction"""
    img = Image.open(uploaded_image)
    
    # Force RGB mode for EfficientNetB0
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0  # shape: (224,224,3)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

# Function to predict tumor type
def predict_tumor(model, img_array):
    """Make prediction using the loaded model"""
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    all_confidences = prediction[0]
    
    return predicted_class, confidence, all_confidences

# Function to get confidence color
def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📋 About")
    st.sidebar.info(
        """
        This app uses a deep learning model to detect brain tumors from MRI images.
        
        **Tumor Types:**
        - 🔴 Glioma
        - 🟡 Meningioma  
        - 🟢 No Tumor
        - 🔵 Pituitary
        
        **Instructions:**
        1. Upload an MRI brain scan image
        2. Wait for the analysis
        3. View the prediction results
        """
    )
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File upload
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose an MRI brain scan image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear MRI brain scan image for analysis"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Add prediction button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... Please wait."):
                    try:
                        # Preprocess image
                        img_array, processed_img = preprocess_image(uploaded_file)
                        
                        # Make prediction
                        predicted_class, confidence, all_confidences = predict_tumor(model, img_array)
                        
                        # Display results
                        predicted_label = class_names[predicted_class]
                        confidence_pct = confidence * 100
                        
                        # Determine if tumor detected
                        is_tumor = predicted_label != 'notumor'
                        box_class = "tumor-detected" if is_tumor else "no-tumor"
                        
                        # Main prediction box
                        st.markdown(f"""
                        <div class="prediction-box {box_class}">
                            <h3>📌 Prediction: {predicted_label.title()}</h3>
                            <h4 class="{get_confidence_color(confidence)}">
                                🔍 Confidence: {confidence_pct:.2f}%
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Interpretation
                        if is_tumor:
                            if predicted_label == 'glioma':
                                st.warning("⚠️ **Glioma detected** - A type of brain tumor that originates from glial cells.")
                            elif predicted_label == 'meningioma':
                                st.warning("⚠️ **Meningioma detected** - A tumor that arises from the meninges.")
                            elif predicted_label == 'pituitary':
                                st.warning("⚠️ **Pituitary tumor detected** - A tumor in the pituitary gland.")
                        else:
                            st.success("✅ **No tumor detected** - The scan appears normal.")
                        
                        # Confidence interpretation
                        if confidence >= 0.8:
                            st.success("🎯 High confidence prediction")
                        elif confidence >= 0.6:
                            st.warning("⚡ Medium confidence prediction")
                        else:
                            st.error("⚠️ Low confidence prediction - Consider getting a second opinion")
                        
                        # Detailed probability chart
                        st.subheader("📊 Detailed Probability Breakdown")
                        
                        # Create probability dataframe
                        prob_data = {
                            'Class': [name.title() for name in class_names],
                            'Probability': all_confidences * 100,
                            'Color': ['red' if name == predicted_label else 'lightblue' for name in class_names]
                        }
                        
                        # Create bar chart
                        fig = px.bar(
                            x=prob_data['Probability'],
                            y=prob_data['Class'],
                            orientation='h',
                            title="Prediction Probabilities for Each Class",
                            labels={'x': 'Probability (%)', 'y': 'Tumor Type'},
                            color=prob_data['Color'],
                            color_discrete_map={'red': '#ff4444', 'lightblue': '#87ceeb'}
                        )
                        
                        fig.update_layout(
                            showlegend=False,
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional info
                        st.info("""
                        ⚕️ **Important Disclaimer:** 
                        This tool is for educational purposes only and should not replace professional medical diagnosis. 
                        Always consult with a qualified healthcare provider for proper medical evaluation.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    else:
        st.info("👆 Please upload an MRI brain scan image to get started.")
        
        # Show example of expected input
        st.subheader("📋 Expected Input Format")
        st.write("""
        - **File types:** JPG, JPEG, PNG, BMP, TIFF
        - **Content:** Clear MRI brain scan images
        - **Quality:** Higher resolution images generally provide better results
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 Brain Tumor Detector | Built with Streamlit & TensorFlow</p>
    <p><small>For educational purposes only - Not for clinical diagnosis</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()


