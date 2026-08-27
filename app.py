"""
AI Image Classifier using ViT
Student Mini Project
Streamlit + Hugging Face Transformers + PyTorch
"""

import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch


# =========================================================
# PAGE CONFIGURATION
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

st.markdown(
    """
<style>

/* ================================
   MAIN BACKGROUND
================================ */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #ffe4f3 0%, transparent 25%),
        radial-gradient(circle at 90% 10%, #dbeafe 0%, transparent 25%),
        radial-gradient(circle at 50% 100%, #dcfce7 0%, transparent 30%),
        linear-gradient(135deg, #fff7fc, #f2f5ff);
}


/* ================================
   MAIN CONTAINER
================================ */

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================================
   TITLE
================================ */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    margin-bottom: 5px;

    background: linear-gradient(
        90deg,
        #ec008c,
        #8b2cff,
        #0066ff
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* ================================
   SUBTITLE
================================ */

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 20px;
}


/* ================================
   MODEL BADGE
================================ */

.model-badge {
    background: linear-gradient(
        90deg,
        #fff0fa,
        #eeeaff
    );

    border: 2px solid #d8b4fe;
    border-radius: 30px;

    padding: 12px 18px;

    text-align: center;

    color: #6d28d9;

    font-weight: 700;

    margin-bottom: 25px;
}


/* ================================
   UPLOAD CARD
================================ */

.upload-card {
    background: rgba(255, 255, 255, 0.96);

    border: 3px dashed #a855f7;

    border-radius: 25px;

    padding: 30px;

    text-align: center;

    box-shadow:
        0 10px 35px rgba(100, 50, 150, 0.15);

    margin-bottom: 20px;
}


/* ================================
   UPLOAD ICON
================================ */

.upload-icon {
    font-size: 55px;
}


/* ================================
   SECTION TITLE
================================ */

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #5b21b6;
    margin-bottom: 10px;
}


/* ================================
   PREDICTION CARD
================================ */

.prediction-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f4edff
    );

    border-radius: 22px;

    border-left: 7px solid #8b5cf6;

    padding: 22px;

    box-shadow:
        0 8px 25px rgba(100, 60, 160, 0.15);

    margin-top: 10px;
}


/* ================================
   PREDICTED LABEL
================================ */

.predicted-label {
    font-size: 30px;
    font-weight: 900;
    color: #6d28d9;
    text-transform: capitalize;
}


/* ================================
   CONFIDENCE
================================ */

.confidence-number {
    font-size: 30px;
    font-weight: 900;
    color: #ec4899;
}


/* ================================
   INFO CARD
================================ */

.info-card {
    background: linear-gradient(
        135deg,
        #e0f2fe,
        #ede9fe
    );

    padding: 22px;

    border-radius: 20px;

    margin-top: 20px;

    color: #312e81;

    box-shadow:
        0 6px 20px rgba(80, 70, 150, 0.10);
}


/* ================================
   TOP 5 CARD
================================ */

.top-card {
    background: white;

    padding: 15px 20px;

    border-radius: 16px;

    margin-bottom: 10px;

    box-shadow:
        0 5px 15px rgba(80, 50, 130, 0.10);
}


/* ================================
   FOOTER
================================ */

.footer {
    text-align: center;

    color: #777;

    font-size: 14px;

    margin-top: 35px;

    padding-top: 20px;

    border-top: 1px solid #ddd;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f3e8ff,
        #e0f2fe,
        #ecfdf5
    );
}


/* ================================
   BUTTON
================================ */

.stButton > button {
    border-radius: 15px;

    font-weight: 800;

    min-height: 50px;

    border: none;

    transition: 0.3s;
}


.stButton > button:hover {
    transform: scale(1.02);
}


/* ================================
   FILE UPLOADER
================================ */

[data-testid="stFileUploader"] {
    background: linear-gradient(
        135deg,
        #fff0fa,
        #eef2ff
    );

    border-radius: 18px;

    padding: 15px;
}


/* ================================
   DIVIDER
================================ */

hr {
    border: none;

    height: 2px;

    background: linear-gradient(
        90deg,
        #ec008c,
        #8b2cff,
        #0066ff
    );
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD HUGGING FACE MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_name = "google/vit-base-patch16-224"

    # Image processor
    processor = ViTImageProcessor.from_pretrained(
        model_name
    )

    # Pre-trained model from Hugging Face
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
    # AI prediction
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
        k=5
    )

    results = []

    for score, class_index in zip(
        top5.values,
        top5.indices
    ):

        label = model.config.id2label[
            class_index.item()
        ]

        confidence = score.item()

        results.append(
            {
                "label": label,
                "confidence": confidence
            }
        )

    return results


# =========================================================
# MAIN FUNCTION
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
This application uses a pre-trained
**Vision Transformer (ViT)** model from
Hugging Face to classify images.
"""
        )

        st.markdown("### 🧠 Technologies")

        st.markdown(
            """
🐍 **Python**

⚡ **Streamlit**

🔥 **PyTorch**

🤗 **Hugging Face**

🧠 **Vision Transformer**
"""
        )

        st.markdown("---")

        st.markdown("### 📦 Model")

        st.code(
            "google/vit-base-patch16-224"
        )

        st.info(
            """
💡 The model downloads automatically
during the first run.
"""
        )


    # =====================================================
    # UPLOAD CARD
    # =====================================================

    st.markdown(
        """
<div class="upload-card">

<div class="upload-icon">
☁️
</div>

<h2>Upload Your Image</h2>

<p>
📸 JPG &nbsp; • &nbsp;
JPEG &nbsp; • &nbsp;
PNG
</p>

<p>
✨ Let AI discover what's inside your image!
</p>

</div>
""",
        unsafe_allow_html=True
    )


    # =====================================================
    # FILE UPLOADER
    # =====================================================

    uploaded_file = st.file_uploader(
        "📤 Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Supported formats: JPG, JPEG and PNG"
    )


    # =====================================================
    # NO IMAGE
    # =====================================================

    if uploaded_file is None:

        st.markdown(
            """
<div class="info-card">

👆 <b>Upload an image to get started!</b>

<br><br>

📷 Upload your image

<br>

🧠 AI will analyze it

<br>

🏷️ Get the predicted object

<br>

📊 See the confidence score

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

**1️⃣ Upload Image**

Upload a JPG, JPEG or PNG image.

**2️⃣ Image Processing**

The Hugging Face image processor
prepares the image for the AI model.

**3️⃣ Vision Transformer**

The pre-trained
`google/vit-base-patch16-224`
model analyzes the image.

**4️⃣ Classification**

The model compares the image
with ImageNet categories.

**5️⃣ Top 5 Predictions**

The five most likely categories
are displayed.

**6️⃣ Confidence Score**

Each prediction receives a
probability score.
"""
            )


    # =====================================================
    # IMAGE UPLOADED
    # =====================================================

    else:

        # ---------------------------------------------
        # Open image
        # ---------------------------------------------

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "❌ Unable to read this image."
            )

            return


        st.markdown("---")


        # ---------------------------------------------
        # Two columns
        # ---------------------------------------------

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )


        # =================================================
        # LEFT COLUMN
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
        # RIGHT COLUMN
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


            # -----------------------------------------
            # CLASSIFY BUTTON
            # -----------------------------------------

            classify_button = st.button(
                "🔍 CLASSIFY IMAGE",
                type="primary",
                use_container_width=True
            )


            # -----------------------------------------
            # CLASSIFICATION
            # -----------------------------------------

            if classify_button:

                # -------------------------------------
                # Load model
                # -------------------------------------

                with st.spinner(
                    "🧠 Loading Hugging Face AI model..."
                ):

                    processor, model = load_model()


                # -------------------------------------
                # Predict
                # -------------------------------------

                with st.spinner(
                    "✨ AI is analyzing your image..."
                ):

                    results = classify_image(
                        image,
                        processor,
                        model
                    )


                # -------------------------------------
                # Best result
                # -------------------------------------

                best_result = results[0]

                best_label = best_result["label"]

                best_confidence = best_result[
                    "confidence"
                ]

                confidence_percentage = (
                    best_confidence * 100
                )


                # -------------------------------------
                # Success message
                # -------------------------------------

                st.success(
                    "✅ Classification Complete!"
                )


                # -------------------------------------
                # Prediction card
                # -------------------------------------

                st.markdown(
                    f"""
<div class="prediction-card">

<h3>🏆 Best Prediction</h3>

<div class="predicted-label">
{best_label}
</div>

<br>

<b>📊 Confidence</b>

</div>
""",
                    unsafe_allow_html=True
                )


                # -------------------------------------
                # Confidence progress
                # -------------------------------------

                st.progress(
                    best_confidence,
                    text=f"{confidence_percentage:.2f}%"
                )


                st.markdown(
                    f"""
<div class="confidence-number">
{confidence_percentage:.2f}%
</div>
""",
                    unsafe_allow_html=True
                )


                # -------------------------------------
                # Confidence interpretation
                # -------------------------------------

                if confidence_percentage >= 90:

                    st.success(
                        "🌟 The model is highly confident!"
                    )

                    st.balloons()

                elif confidence_percentage >= 70:

                    st.info(
                        "👍 The model is fairly confident."
                    )

                elif confidence_percentage >= 50:

                    st.warning(
                        "🤔 The model has moderate confidence."
                    )

                else:

                    st.warning(
                        "💭 The model is unsure. "
                        "Try a clearer image."
                    )


                # =================================================
                # TOP 5 PREDICTIONS
                # =================================================

                st.markdown("---")

                st.markdown(
                    "## 🏅 Top 5 Predictions"
                )

                st.caption(
                    "Here are the five most likely ImageNet classes:"
                )


                for index, result in enumerate(
                    results,
                    start=1
                ):

                    label = result["label"]

                    confidence = result[
                        "confidence"
                    ]

                    percentage = (
                        confidence * 100
                    )


                    st.markdown(
                        f"""
<div class="top-card">

<b>{index}. 🏷️ {label}</b>

<br>

<span>
Confidence: {percentage:.2f}%
</span>

</div>
""",
                        unsafe_allow_html=True
                    )


                    st.progress(
                        confidence
                    )


    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown(
        """
<div class="footer">

🌈 <b>AI Image Classifier</b>

<br><br>

Student Mini Project

<br>

🐍 Python &nbsp; • &nbsp;
⚡ Streamlit &nbsp; • &nbsp;
🔥 PyTorch &nbsp; • &nbsp;
🤗 Hugging Face

</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    main()
