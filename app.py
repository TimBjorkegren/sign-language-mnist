import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque

model = load_model("cnn_model.keras")
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
          'V', 'W', 'X', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                      min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Prediction smoothing
prediction_buffer = deque(maxlen=5)  # Keep last 5 predictions
confidence_threshold = 0.6  # Only show predictions above this confidence

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
    square[y_offset:y_offset+h, x_offset:x_offset+w] = thresh  # ✅ Fixed: use thresh, not gray
    
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

def get_smoothed_prediction(prediction_probs):
    """Smooth predictions using recent history"""
    predicted_class = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs)
    
    # Add to buffer
    prediction_buffer.append((predicted_class, confidence))
    
    # Only return prediction if confidence is high enough
    if confidence < confidence_threshold:
        return None, confidence
    
    # Get most common prediction from recent frames
    if len(prediction_buffer) >= 3:
        recent_predictions = [pred[0] for pred in list(prediction_buffer)[-3:]]
        most_common = max(set(recent_predictions), key=recent_predictions.count)
        return most_common, confidence
    
    return predicted_class, confidence

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kunde inte läsa frame från kameran")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
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
                hand_img = frame[y_min:y_max, x_min:x_max]
                processed = preprocess_hand_image(hand_img)
                
                # Get prediction probabilities (not just argmax)
                prediction_probs = model.predict(processed, verbose=0)[0]
                
                # Apply smoothing
                predicted_class, confidence = get_smoothed_prediction(prediction_probs)
                
                if predicted_class is not None:
                    # Color based on confidence: green = high, yellow = medium, red = low
                    if confidence > 0.8:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.6:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                    
                    cv2.putText(frame, f'{labels[predicted_class]} ({confidence:.2f})',
                               (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Optional: Show top 3 predictions for debugging
                top3_indices = np.argsort(prediction_probs)[-3:][::-1]
                for i, idx in enumerate(top3_indices):
                    prob = prediction_probs[idx]
                    cv2.putText(frame, f'{labels[idx]}: {prob:.2f}',
                               (x_min, y_max + 20 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("sign language recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()