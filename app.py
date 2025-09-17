import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

model = load_model("cnn_model.keras")


labels_dict = {}
label = 0
for i in range(24):
    if i == 9:
        continue
    labels_dict[label] = str(i)
    label += 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
model.predict(dummy_input)
hands.process(np.zeros((240, 320, 3), dtype=np.uint8))

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

    if data_aux and x_ and y_:

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        img = np.zeros((28, 28), dtype=np.float32)
        for i in range(0, len(data_aux), 2):
            px = int(np.clip(data_aux[i] * 28, 0, 27))
            py = int(np.clip(data_aux[i+1] * 28, 0, 27))
            img[py, px] = 1.0


        input_array = img.reshape(1, 28, 28, 1)


        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction[0])
        predicted_char = labels_dict.get(predicted_class, "?")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        print(predicted_char)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
