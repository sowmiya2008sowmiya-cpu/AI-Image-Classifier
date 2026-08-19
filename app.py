"""
AI Image Classifier using ViT (Vision Transformer)
Student Mini Project - Streamlit + Hugging Face Transformers
"""

import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# -------------------- Load Model (cached) --------------------
@st.cache_resource
def load_model():
    """
    Load the pre-trained ViT model and processor.
    Cached so it doesn't reload on every interaction.
    """
    model_name = "google/vit-base-patch16-224"
    
    # Load processor (handles image preprocessing)
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    # Load model
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    return processor, model

# -------------------- Prediction Function --------------------
def classify_image(image, processor, model):
    """
    Run inference on the uploaded image and return top prediction.
    """
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Run model inference (no gradient calculation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the best prediction
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_idx].item()
    
    # Get the human-readable label
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label, confidence

# -------------------- Main App UI --------------------
def main():
    # Title and description
    st.title("🖼️ AI Image Classifier")
    st.markdown(
        """
        Upload an image and let the **Vision Transformer (ViT)** model tell you what's in it!
        
        *Powered by [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)*
        """
    )
    st.divider()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This app uses a pre-trained **Vision Transformer** model from Hugging Face 
            to classify images into 1,000 categories (ImageNet classes).
            
            **Model:** `google/vit-base-patch16-224`
            """
        )
        st.info("💡 The model downloads automatically on first run (~330 MB).")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    # If user uploaded an image
    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🤖 Prediction")
            
            # Load model (with spinner for first time)
            with st.spinner("Loading AI model... (one-time only)"):
                processor, model = load_model()
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    label, confidence = classify_image(image, processor, model)
                
                # Display results in a nice box
                st.success("Classification Complete!")
                
                st.markdown("### 🏷️ Predicted Label")
                st.markdown(f"## {label}")
                
                st.markdown("### 📊 Confidence")
                
                # Show confidence as progress bar and percentage
                confidence_pct = confidence * 100
                st.progress(confidence / 100.0, text=f"{confidence_pct:.2f}%")
                st.markdown(f"**{confidence_pct:.2f}%**")
                
                # Fun interpretation
                if confidence_pct > 90:
                    st.balloons()
                    st.caption("🌟 The model is very confident!")
                elif confidence_pct > 70:
                    st.caption("👍 The model is fairly confident.")
                else:
                    st.caption("🤔 The model is unsure. Try a clearer image.")
    else:
        # Placeholder when no image is uploaded
        st.info("👆 Upload an image to get started!")
        
        # Show example of what the app does
        with st.expander("📝 How it works"):
            st.markdown(
                """
                1. **Upload** a JPG, JPEG, or PNG image
                2. Click the **"Classify Image"** button
                3. The AI model analyzes the image and returns:
                   - The predicted object **label**
                   - A **confidence score** (how sure the model is)
                """
            )

# -------------------- Run App --------------------
if __name__ == "__main__":
    main()