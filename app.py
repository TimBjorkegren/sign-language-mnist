import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
from keras.models import load_model
import numpy as np

st.title("Hand Sign Detector")

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Global variabel för modell
model = None

def hand_tracking(frame: av.VideoFrame):
    global model
    # Ladda modellen första gången i callback-tråden
    if model is None:
        model = load_model("cnn_model.keras", compile=False)

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

            try:
                prediction = model.predict(hand_resized, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                label_text = f"{labels[class_id]} ({confidence:.2f})"
            except Exception as e:
                label_text = "Prediction error"
                print("Prediction error:", e)

            cv2.putText(image, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="hand_sign_detector",
    video_frame_callback=hand_tracking,
    media_stream_constraints={"video": True, "audio": False},
)