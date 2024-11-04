import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Độ rộng video
cam.set(4, 480)  # Chiều cao video

# Tải bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = input('\n Nhập ID khuôn mặt: ')
print("\n [INFO] Khởi tạo camera...")
count = 0

# Kiểm tra và tạo thư mục datashet nếu chưa có
if not os.path.exists("datashet"):
    os.makedirs("datashet")

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật hình ảnh theo chiều ngang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Lưu ảnh khuôn mặt đã phát hiện
        cv2.imwrite("datashet/user." + str(face_id) + "." + str(count) + ".jpg", gray[y:y + h, x:x + w])

    # Hiển thị khung hình từ camera
    cv2.imshow('image', img)

    # Nhấn 'ESC' để thoát
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    if count >=500:
        break

print("\n [INFO] Thoát...")
cam.release()
cv2.destroyAllWindows()
