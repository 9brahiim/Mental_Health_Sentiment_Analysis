# Mental Health Sentiment Analysis

A machine learning project for sentiment analysis of mental health-related text using fine-tuned LLaMA 2 model.

## 📋 Overview

This project trains a sentiment classification model on mental health text data using Hugging Face's Transformers library. The model is based on LLaMA 2 (7B chat), a powerful large language model fine-tuned for mental health sentiment classification.

## 🚀 Features

- Fine-tuned LLaMA 2 model for mental health sentiment analysis
- 4-bit quantization support for efficient memory usage
- Interactive Streamlit web application
- Support for multiple sentiment labels
- Easy-to-use prediction interface
- Works in both Google Colab and local environments

## ⚠️ Prerequisites

**Important:** LLaMA 2 requires:
1. A Hugging Face account (sign up at https://huggingface.co)
2. Access to LLaMA 2 model (request at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
3. A Hugging Face token (create at https://huggingface.co/settings/tokens)
4. **GPU with at least 16GB VRAM** (recommended for training)
5. For Google Colab: Add your HF token as a secret named `HF_TOKEN`
6. For local: Set `HF_TOKEN` environment variable or use `huggingface-cli login`

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone this repository:
```bash
git clone https://github.com/9brahiim/Mental_Health_Sentiment_Analysis.git
cd Mental_Health_Sentiment_Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The model is trained on the RMHD (Reddit Mental Health Dataset) labeled merged dataset. The dataset should be downloaded automatically when running the notebook, or you can download it manually from Google Drive (ID: `1B5QpclAWO_x78sYTx63Q05yv1kwc_hno`).

## 🎯 Usage

### Training the Model

1. Open `Mental_Health_Sentiment_Analysis.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to train the model
3. The trained model will be saved in the `./sentiment_model` directory

### Using the Streamlit App

1. Make sure you have trained the model first (see above)
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your browser and navigate to `http://localhost:8501`
4. Enter text in the text area and click "Analyze Sentiment"

### Using the Model Programmatically

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")

# Load label map
with open("./sentiment_model/label_map.json", "r") as f:
    label_map = json.load(f)

# Predict sentiment
text = "I am feeling very anxious and stressed about my exams."
inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_label_id = torch.argmax(outputs.logits).item()
predicted_sentiment = label_map[str(predicted_label_id)]
print(f"Predicted Sentiment: {predicted_sentiment}")
```

## 📁 Project Structure

```
Mental_Health_Sentiment_Analysis/
├── Mental_Health_Sentiment_Analysis.ipynb  # Main training notebook
├── app.py                                   # Streamlit web application
├── requirements.txt                         # Python dependencies
├── README.md                               # This file
└── sentiment_model/                        # Trained model directory (created after training)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── label_map.json
```

## 🔧 Model Details

- **Base Model**: `NousResearch/Llama-2-7b-chat-hf`
- **Task**: Sequence Classification
- **Number of Labels**: 12 (varies based on dataset)
- **Max Sequence Length**: 256 tokens
- **Training Epochs**: 3
- **Batch Size**: 2 (with gradient accumulation = 8 effective)
- **Quantization**: 4-bit (NF4) for memory efficiency
- **Mixed Precision**: FP16 enabled

## 📝 Notes

- The model works best with mental health-related text
- **Training requires a GPU with at least 16GB VRAM** (highly recommended)
- LLaMA 2 is much larger than DistilBERT - training will take significantly longer
- 4-bit quantization is used to reduce memory requirements
- The dataset path is automatically detected (Colab vs local environment)
- Label mapping is saved with the model for easy inference
- Make sure you have accepted the LLaMA 2 license on Hugging Face before running

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- The creators of the RMHD dataset
- DistilBERT model developers

