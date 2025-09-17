import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

model = load_model("cnn_model.keras")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
          'W', 'X', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)



def preprocess_hand_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = gray

    resized = cv2.resize(square, (28, 28))
    normalized = resized.astype("float32") / 255
    reshaped = np.reshape(normalized, (1, 28, 28, 1))
    return reshaped

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
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _= frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
            y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max - x_min > 0 and y_max - y_min > 0:
                hand_img = frame[y_min:y_max, x_min:x_max]
                processed = preprocess_hand_image(hand_img)
                prediction = model.predict(processed)
                predicted_class = np.argmax(prediction)

                cv2.putText(frame, f'Predicted: {labels[predicted_class]}',
                (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("sign language recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()