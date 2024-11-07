import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến thư mục chứa dữ liệu hình ảnh
path = 'datashet'

# Tạo đối tượng nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Sửa tên hàm thành LBPHFaceRecognizer_create
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images(path):
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        # Mở hình ảnh và chuyển đổi sang ảnh xám
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')  # Sửa 'unit8' thành 'uint8'
        
        # Lấy ID từ tên tệp
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Sửa dấu phẩy thành dấu chấm

        # Phát hiện khuôn mặt
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

print("\n [INFO] Đang huấn luyện dữ liệu...")

faces, ids = get_images(path)

# Huấn luyện mô hình
recognizer.train(faces, np.array(ids))  # Sửa tên hàm từ write thành train
recognizer.save("trainer/trainer.yml")  # Lưu mô hình vào tệp
print("\n [INFO] {0} khuôn mặt đã được huấn luyện. THOÁT.".format(len(np.unique(ids))))
