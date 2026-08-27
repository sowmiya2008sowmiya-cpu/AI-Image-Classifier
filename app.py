"""
🌈 AI Image Classifier using ViT
Student Mini Project
Streamlit + Hugging Face Transformers + PyTorch
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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #fff0fb 0%,
        #eef4ff 45%,
        #eafff6 100%
    );
}

/* Main container */
.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #ff1493,
        #7b2cff,
        #0066ff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    color: #555;
    margin-bottom: 20px;
}

/* Model badge */
.model-badge {
    background: linear-gradient(
        90deg,
        #fff0fa,
        #eeeeff
    );
    border: 2px solid #d8b4fe;
    border-radius: 30px;
    padding: 12px;
    text-align: center;
    color: #5b21b6;
    font-weight: 700;
    margin-bottom: 25px;
}

/* Upload card */
.upload-card {
    background: rgba(255,255,255,0.95);
    padding: 30px;
    border-radius: 25px;
    border: 3px dashed #a855f7;
    box-shadow: 0 10px 30px rgba(90,50,150,0.12);
    text-align: center;
    margin-bottom: 20px;
}

/* Section title */
.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #5b21b6;
    margin-bottom: 10px;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f4edff
    );
    padding: 22px;
    border-radius: 22px;
    border-left: 7px solid #8b5cf6;
    box-shadow: 0 8px 25px rgba(100,60,160,0.15);
}

/* Prediction */
.predicted-label {
    font-size: 30px;
    font-weight: 800;
    color: #6d28d9;
    text-transform: capitalize;
}

/* Confidence */
.confidence-number {
    font-size: 28px;
    font-weight: 800;
    color: #ec4899;
}

/* Info card */
.info-card {
    background: linear-gradient(
        135deg,
        #e0f2fe,
        #ede9fe
    );
    padding: 20px;
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
    padding: 12px;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: linear-gradient(
        135deg,
        #fff0fa,
        #eef2ff
    );
    border-radius: 18px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD HUGGING FACE MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Hugging Face model
    model_name = "google/vit-base-patch16-224"

    # Image processor
    processor = ViTImageProcessor.from_pretrained(
        model_name
    )

    # Pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        model_name
    )

    # Evaluation mode
    model.eval()

    return processor, model


# =========================================================
# CLASSIFY IMAGE
# =========================================================

def classify_image(image, processor, model):

    # ---------------------------------------------
    # Preprocess image
    # ---------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # ---------------------------------------------
    # Model prediction
    # ---------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

    # ---------------------------------------------
    # Convert logits to probabilities
    # ---------------------------------------------

    probabilities = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )[0]

    # ---------------------------------------------
    # Get Top 5 predictions
    # ---------------------------------------------

    top5 = torch.topk(
        probabilities,
        5
    )

    results = []

    for score, class_idx in zip(
        top5.values,
        top5.indices
    ):

        label = model.config.id2label[
            class_idx.item()
        ]

        confidence = score.item()

        results.append(
            (label, confidence)
        )

    return results


# =========================================================
# MAIN APP
# =========================================================

def main():

    # =====================================================
    # HEADER
    # =====================================================

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
        🤗 Hugging Face &nbsp; • &nbsp;
        🧠 Vision Transformer (ViT) &nbsp; • &nbsp;
        📚 ImageNet 1000 Classes
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.markdown("## 🤖 AI Image Classifier")

        st.markdown("---")

        st.markdown("### 💡 About")

        st.write(
            """
            This project uses a pre-trained
            **Vision Transformer (ViT)** model
            from Hugging Face to classify images.
            """
        )

        st.markdown("### 🧠 Technologies")

        st.markdown(
            """
            🐍 Python

            ⚡ Streamlit

            🔥 PyTorch

            🤗 Hugging Face Transformers

            🧠 Vision Transformer
            """
        )

        st.markdown("---")

        st.info(
            """
            💡 The model will be downloaded
            automatically during the first run.
            """
        )


    # =====================================================
    # UPLOAD SECTION
    # =====================================================

    st.markdown(
        """
        <div class="upload-card">

            <div style="font-size:50px;">
            ☁️
            </div>

            <h2>Upload Your Image</h2>

            <p>
            📸 JPG &nbsp; • &nbsp;
            JPEG &nbsp; • &nbsp;
            PNG
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "📤 Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload JPG, JPEG or PNG image"
    )


    # =====================================================
    # IMAGE UPLOADED
    # =====================================================

    if uploaded_file is not None:

        # Open image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )


        # =================================================
        # IMAGE PREVIEW
        # =================================================

        with col1:

            st.markdown(
                """
                <div class="section-title">
                📷 Uploaded Image
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

            st.caption(
                f"📁 {uploaded_file.name}"
            )


        # =================================================
        # AI PREDICTION
        # =================================================

        with col2:

            st.markdown(
                """
                <div class="section-title">
                🤖 AI Prediction
                </div>
                """,
                unsafe_allow_html=True
            )


            # Load model
            with st.spinner(
                "🧠 Loading AI model..."
            ):

                processor, model = load_model()


            # Classify button
            if st.button(
                "🔍 CLASSIFY IMAGE",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "✨ AI is analyzing your image..."
                ):

                    results = classify_image(
                        image,
                        processor,
                        model
                    )


                # =================================================
                # BEST PREDICTION
                # =================================================

                best_label = results[0][0]
                best_confidence = results[0][1]

                confidence_pct = (
                    best_confidence * 100
                )


                st.success(
                    "✅ Classification Complete!"
                )


                st.markdown(
                    """
                    <div class="prediction-card">

                    <h3>🏆 Best Prediction</h3>

                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
                    <div class="predicted-label">
                    {best_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


                # =================================================
                # CONFIDENCE
                # =================================================

                st.markdown(
                    "### 📊 Confidence"
                )


                st.progress(
                    best_confidence,
                    text=f"{confidence_pct:.2f}%"
                )


                st.markdown(
                    f"""
                    <div class="confidence-number">
                    {confidence_pct:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =================================================
                # CONFIDENCE MESSAGE
                # =================================================

                if confidence_pct >= 90:

                    st.success(
                        "🌟 The model is highly confident!"
                    )

                    st.balloons()


                elif confidence_pct >= 70:

                    st.info(
                        "👍 The model is fairly confident."
                    )


                elif confidence_pct >= 50:

                    st.warning(
                        "🤔 The model has moderate confidence."
                    )


                else:

                    st.warning(
                        "💭 The model is unsure. "
                        "Try a clearer image."
                    )


        # =================================================
        # TOP 5 RESULTS
        # =================================================

        st.markdown("---")

        st.markdown(
            "## 🏅 Top 5 Predictions"
        )

        st.caption(
            "The model's five most likely ImageNet classes:"
        )


        for i, (label, confidence) in enumerate(
            results,
            start=1
        ):

            percentage = confidence * 100

            col_a, col_b = st.columns(
                [3, 1]
            )

            with col_a:

                st.markdown(
                    f"**{i}. 🏷️ {label}**"
                )

            with col_b:

                st.markdown(
                    f"**{percentage:.2f}%**"
                )

            st.progress(
                confidence
            )


    # =====================================================
    # NO IMAGE UPLOADED
    # =====================================================

    else:

        st.markdown(
            """
            <div class="info-card">

            👆 <b>Upload an image to get started!</b>

            <br><br>

            📷 The AI will analyze your image.

            <br>

            🧠 ViT will process the image.

            <br>

            🏷️ The predicted object will be displayed.

            <br>

            📊 You will also get a confidence score.

            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # HOW IT WORKS
        # =================================================

        with st.expander(
            "💡 How does this AI Image Classifier work?"
        ):

            st.markdown(
                """
                ### 🔄 AI Classification Process

                **1️⃣ Upload Image**

                Upload a JPG, JPEG or PNG image.

                **2️⃣ Preprocessing**

                The image is processed using the
                Hugging Face ViT image processor.

                **3️⃣ Vision Transformer**

                The pre-trained
                `google/vit-base-patch16-224`
                model analyzes the image.

                **4️⃣ Prediction**

                The model compares the image against
                ImageNet categories.

                **5️⃣ Top 5 Results**

                The five most likely classes are displayed.

                **6️⃣ Confidence**

                Each prediction has a probability score.
                """
            )


    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown(
        """
        <div class="footer">

        🌈 AI Image Classifier

        <br>

        Student Mini Project

        <br>

        🐍 Python • ⚡ Streamlit • 🔥 PyTorch • 🤗 Hugging Face

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    main()
