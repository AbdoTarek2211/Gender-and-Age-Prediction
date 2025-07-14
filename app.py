import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Age & Gender Prediction",
    page_icon="🧑‍👩‍👧‍👦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧑‍👩‍👧‍👦 Age & Gender Prediction")
st.markdown("Upload an image to predict age and gender using a ResNet50 model trained on UTK Face dataset")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.info("""
**Model Architecture:** ResNet50
**Dataset:** UTK Face
**Gender Accuracy:** ~90%
**Age MAE:** ~6 years
""")

# Model loading function
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('resnet50_best_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize to model input size
    img_resized = cv2.resize(img_array, (256, 256))
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# Prediction function
def predict_age_gender(model, image):
    """Make prediction on preprocessed image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Extract gender and age predictions
        gender_pred = float(predictions[0][0][0])  # Convert to Python float
        age_pred = float(predictions[1][0][0])     # Convert to Python float
        
        # Convert gender prediction to label
        gender_label = "Female" if gender_pred > 0.5 else "Male"
        gender_confidence = gender_pred if gender_pred > 0.5 else 1 - gender_pred
        
        return {
            'gender': gender_label,
            'gender_confidence': float(gender_confidence),  # Ensure Python float
            'age': max(0.0, age_pred)  # Ensure age is not negative and is Python float
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    # Load model
    model = load_model()

    if model is None:
        st.error("Model could not be loaded. Please ensure 'resnet50_best_model.keras' is in the current directory.")
        st.stop()

    # Initialize uploaded_file
    uploaded_file = None

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear face image for best results"
    )

    # Sample images section
    st.markdown("---")
    st.subheader("📸 Don't have an image? Try these samples:")

    # Create 4 columns for sample images
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

    # Display sample images
    with sample_col1:
        st.write("**Sample 1**")
        st.image("image1.jpg", use_column_width=True)
        if st.button("Use Sample 1", key="sample1"):
            uploaded_file = open("image1.jpg", "rb")

    with sample_col2:
        st.write("**Sample 2**")
        st.image("image2.jpg", use_column_width=True)
        if st.button("Use Sample 2", key="sample2"):
            uploaded_file = open("image2.jpg", "rb")

    with sample_col3:
        st.write("**Sample 3**")
        st.image("image3.jpg", use_column_width=True)
        if st.button("Use Sample 3", key="sample3"):
            uploaded_file = open("image3.jpg", "rb")

    with sample_col4:
        st.write("**Sample 4**")
        st.image("image4.jpg", use_column_width=True)
        if st.button("Use Sample 4", key="sample4"):
            uploaded_file = open("image4.jpg", "rb")

    # Check if either uploaded file or sample image is selected
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image information
                st.subheader("Image Information")
                if hasattr(uploaded_file, 'name'):
                    st.write(f"**Filename:** {uploaded_file.name}")
                else:
                    st.write("**Filename:** Sample Image")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return
        
        with col2:
            # Make prediction
            st.subheader("Prediction Results")
            
            with st.spinner("Analyzing image..."):
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Make prediction
                results = predict_age_gender(model, image)
                
                if results:
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Gender prediction
                    st.metric(
                        label="Predicted Gender",
                        value=results['gender'],
                        delta=f"{results['gender_confidence']:.1%} confidence"
                    )
                    
                    # Age prediction
                    st.metric(
                        label="Predicted Age",
                        value=f"{results['age']:.1f} years"
                    )
                    
                    # Progress bars for visualization
                    st.subheader("Confidence Visualization")
                    
                    # Gender confidence bar
                    st.write("**Gender Confidence:**")
                    st.progress(float(results['gender_confidence']))
                    
                    # Age range visualization
                    age_range_min = max(0, results['age'] - 5)
                    age_range_max = min(100, results['age'] + 5)
                    st.write(f"**Estimated Age Range:** {age_range_min:.1f} - {age_range_max:.1f} years")
                    
                    # Additional insights
                    st.subheader("Additional Insights")
                    if results['age'] < 18:
                        st.info("🧒 Predicted as minor (under 18)")
                    elif results['age'] > 60:
                        st.info("👴👵 Predicted as senior (over 60)")
                    else:
                        st.info("🧑‍🦱 Predicted as adult (18-60)")
    
    # Sample images section
    st.markdown("---")
    st.subheader("📸 Don't have an image? Try these samples:")

    # Create 4 columns for sample images
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

    # Display sample images
    with sample_col1:
        st.write("**Sample 1**")
        st.image("image1.jpg", use_column_width=True)
        if st.button("Use Sample 1", key="sample1"):
            uploaded_file = open("image1.jpg", "rb")

    with sample_col2:
        st.write("**Sample 2**")
        st.image("image2.jpg", use_column_width=True)
        if st.button("Use Sample 2", key="sample2"):
            uploaded_file = open("image2.jpg", "rb")

    with sample_col3:
        st.write("**Sample 3**")
        st.image("image3.jpg", use_column_width=True)
        if st.button("Use Sample 3", key="sample3"):
            uploaded_file = open("image3.jpg", "rb")

    with sample_col4:
        st.write("**Sample 4**")
        st.image("image4.jpg", use_column_width=True)
        if st.button("Use Sample 4", key="sample4"):
            uploaded_file = open("image4.jpg", "rb")

    # Tips section
    st.markdown("""
    **Tips for better predictions:**
    - Use clear, well-lit images
    - Ensure the face is clearly visible
    - Avoid heavy makeup or filters
    - Front-facing images work best
    """)
    
    # Model details in expander
    with st.expander("🔍 Model Details"):
        st.markdown("""
        **Architecture:** ResNet50 with transfer learning
        **Training Dataset:** UTK Face dataset
        **Input Size:** 256x256 pixels
        **Output:** 
        - Gender: Binary classification (Male/Female)
        - Age: Regression (0-100+ years)
        
        **Performance:**
        - Gender Classification Accuracy: ~90%
        - Age Prediction MAE: ~6 years
        
        **Preprocessing:**
        - Images are resized to 256x256
        - Pixel values normalized to [0,1]
        - No additional augmentation during inference
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")

if __name__ == "__main__":
    main()