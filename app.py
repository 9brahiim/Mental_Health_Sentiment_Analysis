"""
Streamlit App for Mental Health Sentiment Analysis
"""
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import json

# Define the path to your saved model and tokenizer
output_dir = "./sentiment_model"

# Check if model exists
if not os.path.exists(output_dir):
    st.error(f"Model directory '{output_dir}' not found. Please train the model first using the notebook.")
    st.stop()

# Load the saved model and tokenizer
@st.cache_resource
def load_model():
    try:
        loaded_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return loaded_model, loaded_tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

loaded_model, loaded_tokenizer = load_model()

if loaded_model is None or loaded_tokenizer is None:
    st.stop()

# Load label map
label_map = {}
try:
    label_map_path = os.path.join(output_dir, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
            # Convert keys to int if they're strings
            label_map = {int(k): v for k, v in label_map.items()}
        st.success("Label map loaded successfully!")
    else:
        st.warning("Label map file not found. Using default labels.")
        label_map = {i: f"Label {i}" for i in range(12)}  # Default fallback
except Exception as e:
    st.error(f"Error loading label map: {e}")
    label_map = {i: f"Label {i}" for i in range(12)}  # Default fallback

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using the loaded model.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        str: The predicted sentiment label.
    """
    if not text:
        return "Please enter text for analysis."

    # Tokenize the input text
    inputs = loaded_tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    # Get the predicted label (index with the highest score)
    predicted_label_id = torch.argmax(outputs.logits).item()

    # Map the label ID back to the sentiment label string
    predicted_sentiment = label_map.get(predicted_label_id, "Unknown Label")

    return predicted_sentiment

# Streamlit App Interface
st.title("🧠 Mental Health Sentiment Analysis")
st.markdown("Analyze the sentiment of mental health-related text using a fine-tuned DistilBERT model.")

st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a fine-tuned DistilBERT model to classify mental health text into different sentiment categories.

**How to use:**
1. Enter your text in the text area below
2. Click "Analyze Sentiment"
3. View the predicted sentiment label
""")

user_input = st.text_area("Enter text here:", "", height=150, placeholder="Example: I've been feeling anxious about my upcoming presentation...")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment = predict_sentiment(user_input)
        st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")

# Display label map in sidebar
if label_map:
    with st.sidebar.expander("Available Labels"):
        for label_id, label_name in sorted(label_map.items()):
            st.text(f"{label_id}: {label_name}")
