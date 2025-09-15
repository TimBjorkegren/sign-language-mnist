import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp


st.title("Hand Sign Detector")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
