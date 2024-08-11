import streamlit as st
import torch
import torchaudio
import torchvision
import pickle

# Load models
multi_modal_model = pickle.load(open('multi_modal_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
mlp_model = pickle.load(open('mlp_model.pkl', 'rb'))

st.title('Multi-Modal Emotion Recognition')

uploaded_audio = st.file_uploader("Choose an audio file", type='wav')
uploaded_video = st.file_uploader("Choose a video file", type='mp4')

if uploaded_audio and uploaded_video:
    audio, sr = torchaudio.load(uploaded_audio)
    video, _, _ = torchvision.io.read_video(uploaded_video)
    
    # Preprocess audio and video
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    video = video.permute(3, 0, 1, 2)  # THWC to CTHW
    
    # Make predictions
    with torch.no_grad():
        multi_modal_pred = multi_modal_model(audio, video)
        rf_pred = random_forest_model.predict(audio.numpy().flatten())
        svm_pred = svm_model.predict(audio.numpy().flatten())
        mlp_pred = mlp_model.predict(audio.numpy().flatten())
    
    # Display results
    st.write("Multi-Modal Model Prediction:", multi_modal_pred.argmax().item())
    st.write("Random Forest Prediction:", rf_pred[0])
    st.write("SVM Prediction:", svm_pred[0])
    st.write("MLP Prediction:", mlp_pred[0])