import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import os

# Enhanced UI CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Poppins:wght@300;500;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #000000 100%);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }

    .title {
        font-size: 3.5rem;
        color: #d4af37;
        text-align: center;
        font-family: 'Playfair Display', serif;
        text-shadow: 0px 0px 25px rgba(212,175,55,0.8);
        animation: fadeIn 1.2s ease-in-out;
    }

    .subtitle {
        font-size: 1.6rem;
        color: #e4e4e4;
        text-align: center;
        font-weight: 300;
        animation: fadeIn 2s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton > button {
        background: linear-gradient(to right, #d4af37, #b8860b);
        color: #ffffff;
        padding: 12px 28px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 12px;
        border: 1px solid #d4af37;
        box-shadow: 0px 0px 10px rgba(212,175,55,0.5);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #b8860b, #d4af37);
        transform: scale(1.05);
        box-shadow: 0px 0px 20px rgba(212,175,55,0.8);
    }

    .css-1d391kg {
        background-color: #121212 !important;
        border-right: 2px solid #d4af37;
    }

    .prediction-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid #d4af37;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 0px 20px rgba(212,175,55,0.4);
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        0% { box-shadow: 0px 0px 10px rgba(212,175,55,0.3); }
        100% { box-shadow: 0px 0px 25px rgba(212,175,55,0.6); }
    }

    .prediction-box h3 {
        color: #ffffff;
        font-size: 1.8rem;
    }

    .prediction-box span {
        color: #d4af37;
        font-weight: bold;
    }

    .music-notes {
        font-size: 2.5rem;
        animation: bounce 1.5s infinite ease-in-out;
        color: #d4af37;
    }

    @keyframes bounce {
        0%,100% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
    }

    .css-1d6wzja {
        background: #2e2e2e !important;
        border: 1px solid #d4af37 !important;
        border-radius: 10px !important;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r"C:\Users\Deepak\OneDrive\Desktop\Project\Trained_model.keras")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Audio preprocessing
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        data = []
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)

        return np.array(data)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None

# Prediction
def model_prediction(X_test):
    model = load_model()
    if model is None or X_test is None:
        return None
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #d4af37;'>🎶 Control Panel</h2>", unsafe_allow_html=True)
    app_mode = st.selectbox("Choose a Mode", ["Welcome", "Genre Prediction"], key="nav")

# Main app
if app_mode == "Welcome":
    st.markdown('<p class="title">MelodyPulse</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your Classic Music Companion</p>', unsafe_allow_html=True)

    st.markdown("""
        ### Discover Your Music  
        Step into the world of timeless tunes with MelodyPulse.  
        Head to **Genre Prediction** to uncover the soul of your tracks!
    """)

elif app_mode == "Genre Prediction":
    st.markdown('<p class="title">Genre Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Unveil the Essence of Your Music</p>', unsafe_allow_html=True)

    test_mp3 = st.file_uploader("Upload an MP3 File", type=["mp3"], key="uploader")

    if test_mp3:
        filepath = f"Test_Music/{test_mp3.name}"
        os.makedirs("Test_Music", exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🎵 Play Track"):
                st.audio(filepath)
                st.success("Playing your track!")

        with col2:
            if st.button("🎸 Predict Genre"):
                with st.spinner("Analyzing the rhythm..."):
                    X_test = load_and_preprocess_data(filepath)
                    if X_test is not None:
                        result_index = model_prediction(X_test)
                        if result_index is not None:
                            label = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

                            st.markdown(f"""
                            <div class='prediction-box'>
                                <h3>Your Genre is: <span>{label[result_index]}</span></h3>
                                <p class='music-notes'>♪ ♫</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.image("https://images.unsplash.com/photo-1514320291848-d8f9fdedce0e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80",
                                     caption=f"Feeling the {label[result_index]} vibes!")

                        else:
                            st.error("Prediction failed. Try again.")

    if test_mp3 and os.path.exists(filepath):
        os.remove(filepath)
