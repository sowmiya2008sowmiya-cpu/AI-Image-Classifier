"""
🌈 AI Image Classifier using ViT (Vision Transformer)
Student Mini Project - Streamlit + Hugging Face Transformers
"""

import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS - COLORFUL UI
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #fdf2ff 0%,
        #eef5ff 45%,
        #e9fff8 100%
    );
}

/* Main container */
.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #ff1493,
        #7b2cff,
        #0066ff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #444;
    margin-bottom: 20px;
}

/* Model badge */
.model-badge {
    background: linear-gradient(
        90deg,
        #fff1fb,
        #eef0ff
    );
    border-radius: 30px;
    padding: 12px 20px;
    text-align: center;
    font-weight: 600;
    color: #5426a8;
    border: 2px solid #e1c9ff;
    margin: 15px 0 25px 0;
}

/* Upload card */
.upload-card {
    background: rgba(255,255,255,0.95);
    padding: 30px;
    border-radius: 25px;
    border: 2px dashed #a855f7;
    box-shadow: 0 10px 35px rgba(90,50,150,0.12);
    text-align: center;
    margin-top: 20px;
}

/* Section headings */
.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #5426a8;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f5efff
    );
    padding: 25px;
    border-radius: 22px;
    border-left: 7px solid #8b5cf6;
    box-shadow: 0 8px 25px rgba(100,60,160,0.15);
}

/* Label */
.predicted-label {
    font-size: 32px;
    font-weight: 800;
    color: #5b21b6;
    text-transform: capitalize;
}

/* Confidence */
.confidence-number {
    font-size: 30px;
    font-weight: 800;
    color: #ec4899;
}

/* Info box */
.info-card {
    background: linear-gradient(
        135deg,
        #e0f2fe,
        #ede9fe
    );
    padding: 18px 22px;
    border-radius: 18px;
    margin-top: 20px;
    color: #312e81;
}

/* Footer */
.footer {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-top: 35px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f3e8ff,
        #e0f2fe,
        #ecfdf5
    );
}

/* Buttons */
.stButton > button {
    border-radius: 15px;
    font-weight: 700;
    border: none;
    padding: 12px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: linear-gradient(
        135deg,
        #fff0fa,
        #eef2ff
    );
    border-radius: 18px;
    padding: 15px;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(
        90deg,
        #ff1493,
        #7b2cff,
        #0066ff
    );
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_name = "google/vit-base-patch16-224"

    processor = ViTImageProcessor.from_pretrained(
        model_name
    )

    model = ViTForImageClassification.from_pretrained(
        model_name
    )

    model.eval()

    return processor, model


# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

def classify_image(image, processor, model):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.nn.functional.softmax(
        logits,
        dim=-1
    )

    predicted_class_idx = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class_idx
    ].item()

    predicted_label = model.config.id2label[
        predicted_class_idx
    ]

    return predicted_label, confidence


# =========================================================
# MAIN APP
# =========================================================

def main():

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">🖼️ AI Image Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        ✨ Upload an image and let Artificial Intelligence
        identify what's inside it!
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="model-badge">
        ⚡ Powered by Google Vision Transformer (ViT)
        &nbsp; | &nbsp;
        🧠 ImageNet • 1000 Classes
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.markdown("## 🤖 AI Image Classifier")

        st.markdown("---")

        st.markdown("### 💡 About the Project")

        st.write(
            """
            This application uses a pre-trained
            **Vision Transformer (ViT)** model to
            recognize objects in images.
            """
        )

        st.markdown("### 🧠 Technology")

        st.markdown(
            """
            🔹 Python  
            🔹 Streamlit  
            🔹 PyTorch  
            🔹 Hugging Face Transformers  
            🔹 Vision Transformer (ViT)
            """
        )

        st.markdown("---")

        st.info(
            "💡 The AI model downloads automatically "
            "the first time you run the application."
        )

    # -----------------------------------------------------
    # UPLOAD SECTION
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="upload-card">
            <div style="font-size:45px;">☁️</div>
            <h2>Upload Your Image</h2>
            <p>📸 JPG • JPEG • PNG</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "📤 Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    # -----------------------------------------------------
    # WHEN IMAGE IS UPLOADED
    # -----------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        with col1:

            st.markdown(
                '<div class="section-title">📷 Your Image</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"📁 {uploaded_file.name}"
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with col2:

            st.markdown(
                '<div class="section-title">🤖 AI Prediction</div>',
                unsafe_allow_html=True
            )

            with st.spinner(
                "🧠 Loading AI model..."
            ):

                processor, model = load_model()

            if st.button(
                "🔍 CLASSIFY IMAGE",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "✨ AI is analyzing your image..."
                ):

                    label, confidence = classify_image(
                        image,
                        processor,
                        model
                    )

                confidence_pct = confidence * 100

                # -----------------------------------------
                # RESULT CARD
                # -----------------------------------------

                st.markdown(
                    """
                    <div class="prediction-card">
                    <h3>🏷️ Predicted Object</h3>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="predicted-label">
                    {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                st.markdown("### 📊 Confidence")

                st.progress(
                    confidence,
                    text=f"{confidence_pct:.2f}% confidence"
                )

                st.markdown(
                    f"""
                    <div class="confidence-number">
                    {confidence_pct:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # -----------------------------------------
                # CONFIDENCE MESSAGE
                # -----------------------------------------

                if confidence_pct > 90:

                    st.success(
                        "🌟 Excellent! The model is highly confident."
                    )

                    st.balloons()

                elif confidence_pct > 70:

                    st.info(
                        "👍 Good prediction! The model is fairly confident."
                    )

                elif confidence_pct > 50:

                    st.warning(
                        "🤔 The model has moderate confidence."
                    )

                else:

                    st.warning(
                        "💭 The model is unsure. "
                        "Try a clearer image."
                    )

    # -----------------------------------------------------
    # NO IMAGE
    # -----------------------------------------------------

    else:

        st.markdown(
            """
            <div class="info-card">
                👆 <b>Upload an image to get started!</b><br>
                <small>
                The AI will analyze your image and predict
                what object it contains.
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # HOW IT WORKS
        # -------------------------------------------------

        with st.expander(
            "💡 How does this AI Image Classifier work?"
        ):

            st.markdown(
                """
                ### 🔄 Classification Process

                **1️⃣ Upload**
                
                Upload a JPG, JPEG, or PNG image.

                **2️⃣ Preprocessing**
                
                The image is converted into a format
                that the Vision Transformer can understand.

                **3️⃣ AI Analysis**
                
                The pre-trained **ViT model** analyzes
                visual patterns in the image.

                **4️⃣ Prediction**
                
                The model selects the most likely object
                from **1000 ImageNet categories**.

                **5️⃣ Confidence Score**
                
                A percentage shows how confident the
                model is about its prediction.
                """
            )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="footer">
        🌈 AI Image Classifier • Student Mini Project<br>
        Built with Python 🐍 + Streamlit ⚡ + PyTorch 🔥
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()
