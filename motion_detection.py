import cv2
import numpy as np
from keras.models import load_model

emotion_model = load_model(
    r"C:\Users\tanis\OneDrive\Desktop\cv\fer2013_mini_XCEPTION.102-0.66.hdf5",
    compile=False
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_DISPLAY = {
    'Angry':    '😠 Angry',
    'Disgust':  '🤢 Disgusted',
    'Fear':     '😨 Fearful',
    'Happy':    '😄 Happy',
    'Sad':      '😢 Sad',
    'Surprise': '😲 Surprised',
    'Neutral':  '😐 Neutral',
}

EMOTION_COLOR = {
    'Angry':    (0, 0, 255),
    'Disgust':  (0, 128, 0),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 255, 255),
    'Sad':      (255, 0, 0),
    'Surprise': (0, 165, 255),
    'Neutral':  (200, 200, 200),
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

emotion = "Detecting..."
second_emotion = ""
confidence = 0.0
frame_count = 0
PREDICT_EVERY = 3  # predict every 3 frames

while cap.isOpened():
    if not ret:
        break

    small = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    gray1 = cv2.cvtColor(cv2.resize(frame1, (0,0), fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(blur1, blur2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(c) > 800 for c in contours)

    faces = face_cascade.detectMultiScale(gray2, 1.1, 4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        x2, y2, w2, h2 = x*2, y*2, w*2, h*2

        if motion_detected and frame_count % PREDICT_EVERY == 0:
            face_roi = gray2[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0
            face_roi = np.reshape(face_roi, (1, 64, 64, 1))
            preds = emotion_model.predict(face_roi, verbose=0)[0]
            top2 = np.argsort(preds)[::-1][:2]
            emotion = emotion_labels[top2[0]]
            second_emotion = emotion_labels[top2[1]]
            confidence = preds[top2[0]] * 100

        color = EMOTION_COLOR.get(emotion, (0, 255, 0))
        label = EMOTION_DISPLAY.get(emotion, emotion)
        cv2.rectangle(frame2, (x2, y2), (x2+w2, y2+h2), color, 2)
        cv2.putText(frame2, f"{label} ({confidence:.0f}%)", (x2, y2-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame2, f"also: {second_emotion}", (x2, y2-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Emotion + Motion Detection", frame2)

    frame1 = frame2
    ret, frame2 = cap.read()
    frame_count += 1

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
