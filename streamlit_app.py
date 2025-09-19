import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
from keras.models import load_model
import numpy as np
from collections import deque
import threading


st.title("Hand sign language detector")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
          'V', 'W', 'X', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                      min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Prediction smoothing
latest_prediction = {"label": None, "confidence": 0.0}
prediction_buffer = deque(maxlen=5)  # Keep last 5 predictions
confidence_threshold = 0.6  # Only show predictions above this confidence

lock = threading.Lock()

@st.cache_resource
def load_keras_model():
    return load_model("cnn_model.keras", compile=False)

model = load_keras_model()

def preprocess_hand_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
   
    h, w = thresh.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = thresh  # Fixed: use thresh, not gray
    
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

def predict_in_background(hand_img):
    global latest_prediction
    processed = preprocess_hand_image(hand_img)
    prediction_probs = model.predict(processed, verbose=0)[0]

    predicted_class = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs)

    with lock:
        if confidence > confidence_threshold:
            latest_prediction = {
                "label": labels[predicted_class],
                "confidence": confidence
            }
        else:
            latest_prediction = {"label": None, "confidence": confidence}



def video_frame_callback(frame: av.VideoFrame):
    global latest_prediction
    image = frame.to_ndarray(format="bgr24")
    image = cv2.flip(image, 1)
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = image.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
            y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
            
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            if x_max - x_min > 0 and y_max - y_min > 0:
                hand_img = image[y_min:y_max, x_min:x_max]

                threading.Thread(target=predict_in_background, args=(hand_img,)).start()

                with lock:
                    if latest_prediction["label"] is not None:
                        label = latest_prediction["label"]
                        conf = latest_prediction["confidence"]
                        cv2.putText(
                            image, f'{label} ({conf:.2f})',
                            (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
                        )


              
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="hand_sign_detector",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)