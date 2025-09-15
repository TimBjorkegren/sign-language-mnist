import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av


st.title("Hand Sign Detector")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def hand_tracking(frame: av.VideoFrame):
    image = frame.to_ndarray(format="bgr24") #Convert image to array 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Using cv2 to handle the image and convert it to RGB INSTEAD OF BGR (Mediapipe only works on RGB IMAGES)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return av.VideoFrame.from_ndarray(image, format="bgr24") #Taking the numpy image array and coverting it back to a AV frame and changing the color format to bgr which is what stream webrtc is expecting

webrtc_streamer(key="hand_tracking", video_frame_callback=hand_tracking)