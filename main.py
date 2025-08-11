import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Define class names (match the order of training data folders)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to load and preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict tumor type
def predict_tumor(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"📌 Prediction: {class_names[predicted_class]}")
    print(f"🔍 Confidence: {confidence * 100:.2f}%")

    # Show image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {class_names[predicted_class]}")
    plt.show()

# --------- Test it with your image ----------
# Replace with the path to your image
img_path = 'sampleM.jpg'
predict_tumor(img_path)

img_path = 'sampleG.jpg'
predict_tumor(img_path)

img_path = 'sampleM2.jpg'
predict_tumor(img_path)

img_path = 'sampleG2.jpg'
predict_tumor(img_path)

