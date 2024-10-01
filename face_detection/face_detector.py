import cv2
import os
import face_recognition
import numpy as np

class FaceDetector:
    def __init__(self, database_path):
        self.database_path = database_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_save(self, frame, person_id):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Save image
            image_path = os.path.join(self.database_path, "known_faces", str(person_id), "images")
            os.makedirs(image_path, exist_ok=True)
            cv2.imwrite(os.path.join(image_path, f"{person_id}_{len(os.listdir(image_path))}.jpg"), face)

            # Save encoding
            encoding_path = os.path.join(self.database_path, "known_faces", str(person_id), "encodings")
            os.makedirs(encoding_path, exist_ok=True)
            encoding = face_recognition.face_encodings(face)
            if encoding:
                np.save(os.path.join(encoding_path, f"{person_id}_{len(os.listdir(encoding_path))}.npy"), encoding[0])

        return len(faces) > 0

    def capture_person(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        person_id = len(os.listdir(os.path.join(self.database_path, "known_faces")))
        frame_count = 0

        while frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            if self.detect_and_save(frame, person_id):
                frame_count += 1

            cv2.imshow(f'Capturing Frames (Camera {camera_id})', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return person_id