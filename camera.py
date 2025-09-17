import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = load_model("cnn_model.keras")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


labels_dict = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H',
    8:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 
    16:'P', 17:'Q', 18:'R', 19:'S', 20:'T', 21:'U', 22:'V', 
    23:'W', 24:'X'
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Kunde inte läsa frame från kameran")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        img = np.zeros((28, 28), dtype=np.float32)
        for x, y in zip(x_, y_):
            px = int(x * 28)
            py = int(y * 28)
            px = min(max(px, 0), 27)
            py = min(max(py, 0), 27)
            img[py, px] = 1

        img = img.reshape(1, 28, 28, 1)
        #prediction = model.predict([np.asarray(data_aux)])
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_char = labels_dict.get(predicted_class)

        print(predicted_char)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()