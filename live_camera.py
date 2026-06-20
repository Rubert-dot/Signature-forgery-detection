import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/calsignet.h5")

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = img.reshape(1,128,128,1)
    return img

cap = cv2.VideoCapture(0)

print("Press 'S' to capture signature, 'Q' to quit")

while True:
    ret, frame = cap.read()
    cv2.imshow("Live Signature Capture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img = preprocess(frame)
        pred = model.predict(img)[0][0]

        if pred > 0.5:
            print("✅ Genuine Signature")
        else:
            print("❌ Forged Signature")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()