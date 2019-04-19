import cv2
import pickle

face_cascade = cv2.CascadeClassifier('/Users/a222/Desktop/python/Machine_learning/mtuci_lessions/face_recognition/cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

# на выбор, можно использовать одну из нижеприведенных функций для изменения размера окна вёбкамеры
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


def rescale_frame(frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


make_480p()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = rescale_frame(frame, percent=140)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]

        # recognize ? deep learned model predict keras tensorflow ...
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:   # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)   # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
