# 🖼️ AI Image Classifier

An interactive **AI Image Classification web application** built using **Python, Streamlit, PyTorch, and Hugging Face Transformers**.

The application uses the pre-trained **Vision Transformer (ViT)** model `google/vit-base-patch16-224` to analyze uploaded images and predict the most likely ImageNet categories.

## ✨ Features

* 📤 Upload JPG, JPEG, and PNG images
* 🧠 AI-powered image classification
* 🤗 Uses a pre-trained Hugging Face Vision Transformer
* 🏆 Displays the best prediction
* 📊 Shows confidence percentage
* 🏅 Displays Top 5 predictions
* 🎨 Colourful and interactive Streamlit interface
* ⚡ Model is cached for faster subsequent predictions

## 🧠 AI Model

**Model:** `google/vit-base-patch16-224`

**Architecture:** Vision Transformer (ViT)

**Dataset:** ImageNet

**Number of Classes:** 1000

The model is loaded directly from the Hugging Face Model Hub.

## 🛠️ Technologies Used

* 🐍 Python
* ⚡ Streamlit
* 🔥 PyTorch
* 🤗 Hugging Face Transformers
* 🖼️ Pillow (PIL)

## 🔄 How It Works

```text
User Uploads Image
        ↓
Image Preprocessing
        ↓
Vision Transformer (ViT)
        ↓
AI Classification
        ↓
Top 5 Predictions
        ↓
Confidence Scores
```

## 📁 Project Structure

```text
AI-Image-Classifier/
│
├── app.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd AI-Image-Classifier
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

Create a file named `requirements.txt`:

```text
streamlit
torch
transformers
pillow
```

## 🖥️ Application Workflow

1. Open the application.
2. Upload an image.
3. Click **Classify Image**.
4. The ViT model analyzes the image.
5. The application displays the predicted class.
6. The confidence score is shown.
7. Top 5 predictions are displayed.

## 📊 Example Output

```text
🏆 Best Prediction

Golden Retriever

Confidence: 94.32%

🏅 Top 5 Predictions

1. Golden Retriever — 94.32%
2. Labrador Retriever — 2.41%
3. Kuvasz — 1.20%
4. Tibetan Mastiff — 0.87%
5. Dog — 0.65%
```

*Example output only. Actual predictions depend on the uploaded image.*

## ⚠️ Limitations

This application uses an ImageNet-trained model with **1000 predefined categories**. Therefore, it may not correctly recognize every possible object or image.

The confidence score represents the model's probability for its selected ImageNet classes and should not be treated as guaranteed accuracy.

## 🚀 Future Improvements

* 🔹 Add more image categories
* 🔹 Add image history
* 🔹 Add prediction charts
* 🔹 Add drag-and-drop upload
* 🔹 Deploy the application online
* 🔹 Improve classification with a newer or task-specific model

## 👩‍💻 Project

**AI Image Classifier**

A student mini project demonstrating the use of **Artificial Intelligence, Deep Learning, Computer Vision, and Generative AI-related technologies**.

---

⭐ If you find this project useful, consider giving the repository a star!
