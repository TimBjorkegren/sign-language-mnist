import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
from keras.models import load_model
import numpy as np


st.title("Hand sign language detector")

labels_dict = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H',
    8:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 
    16:'P', 17:'Q', 18:'R', 19:'S', 20:'T', 21:'U', 22:'V', 
    23:'W', 24:'X'
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

@st.cache_resource
def load_keras_model():
    return load_model("cnn_model.keras", compile=False)

model = load_keras_model()

def hand_tracking(frame: av.VideoFrame):
    image = frame.to_ndarray(format="bgr24")
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = image.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x1, x2 = max(min(x_coords)-10, 0), min(max(x_coords)+10, w)
            y1, y2 = max(min(y_coords)-10, 0), min(max(y_coords)+10, h)

            hand_crop = image[y1:y2, x1:x2]
            if hand_crop.size == 0:
                continue

            hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            hand_resized = cv2.resize(hand_gray, (28,28))
            hand_resized = hand_resized.astype(np.float32)/255.0
            hand_resized = hand_resized.reshape(1,28,28,1)

            prediction = model.predict(hand_resized, verbose=0)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            label_text = f"{labels_dict[class_id]} ({confidence:.2f})"
            

            cv2.putText(image, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="hand_sign_detector",
    video_frame_callback=hand_tracking,
    media_stream_constraints={"video": True, "audio": False},
)