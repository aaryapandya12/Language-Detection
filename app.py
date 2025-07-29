import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

@st.cache_resource
def load_model():
    model = joblib.load('language_detection_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, label_encoder

model, label_encoder = load_model()

st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-box {
        border-radius: 5px;
        padding: 20px;
        margin-top: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🌐 Language Detection Tool</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        Detect the language of any text using machine learning
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("""
    This app uses an ensemble machine learning model to detect the language of input text.
    It supports detection of multiple languages including:
    - English
    - French
    - Spanish
    - German
    - Russian
    - And many more...
    """)
    
    st.header("How to Use")
    st.write("""
    1. Enter text in the input box
    2. Click the 'Detect Language' button
    3. View the results
    """)
    
    st.header("Model Information")
    st.write("""
    - Model: Ensemble of Naive Bayes and Logistic Regression
    - Features: Character n-grams (1-4)
    - Accuracy: >95% on test data
    """)

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area(
        "Enter text to detect language:",
        height=200,
        placeholder="Type or paste text here in any language..."
    )
    
    detect_button = st.button("Detect Language", use_container_width=True)
    
    if detect_button and input_text:
        with st.spinner('Analyzing text...'):
        
            prediction = model.predict([input_text])
            predicted_language = label_encoder.inverse_transform(prediction)[0]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([input_text])[0]
                top_3_indices = probabilities.argsort()[-3:][::-1]
                top_3_languages = label_encoder.inverse_transform(top_3_indices)
                top_3_probs = probabilities[top_3_indices]
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Detection Results")
            
            st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <h3 style="color: #2c3e50;">Predicted Language: <span style="color: #3498db;">{predicted_language}</span></h3>
                </div>
            """, unsafe_allow_html=True)
            
            if hasattr(model, 'predict_proba'):
                st.markdown("""
                    <div style="margin-top: 15px;">
                        <h4>Confidence Scores:</h4>
                """, unsafe_allow_html=True)
                
                for lang, prob in zip(top_3_languages, top_3_probs):
                    progress = int(prob * 100)
                    st.markdown(f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{lang}</span>
                                <span>{progress}%</span>
                            </div>
                            <progress value="{progress}" max="100" style="width: 100%; height: 10px;"></progress>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Try These Examples:")
    
    examples = {
        "English": "The quick brown fox jumps over the lazy dog",
        "French": "Le renard brun rapide saute par-dessus le chien paresseux",
        "Spanish": "El rápido zorro marrón salta sobre el perro perezoso",
        "German": "Der schnelle braune Fuchs springt über den faulen Hund",
        "Russian": "Быстрая коричневая лиса перепрыгивает через ленивую собаку",
        "Japanese": "速い茶色の狐が怠惰な犬を飛び越えます",
        "Arabic": "يقفز الثعلب البني السريع فوق الكلب الكسول"
    }
    
    for lang, text in examples.items():
        if st.button(f"{lang}: {text[:20]}...", use_container_width=True):
            input_text = text


st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #7f8c8d;">
        Language Detection App • Powered by Machine Learning
    </div>
""", unsafe_allow_html=True)