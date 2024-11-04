import cv2
import numpy as np
import os

# Khởi tạo bộ nhận diện khuôn mặt và tải mô hình đã huấn luyện
recognize = cv2.face.LBPHFaceRecognizer_create()
recognize.read('trainer/trainer.yml')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

# Danh sách tên tương ứng với ID
names = ['0', 'lai trung lam', '2', '3', '4']

# Mở camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Đặt chiều rộng
cam.set(4, 480)  # Đặt chiều cao

# Kích thước tối thiểu cho phát hiện khuôn mặt
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật hình ảnh theo chiều ngang (như gương)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    
    # Xử lý từng khuôn mặt phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognize.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            id = names[id]  # Lấy tên từ danh sách
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "{0}%".format(round(100 - confidence))  # Sửa định dạng hiển thị

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    
    cv2.imshow('Nhận diện khuôn mặt', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Nhấn phím Esc để thoát
        break

print("\n [INFO] THOÁT")
cam.release()
cv2.destroyAllWindows()
