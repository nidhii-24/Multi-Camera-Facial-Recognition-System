import cv2
import os
import face_recognition
import numpy as np

class FaceRecognizer:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_face_encodings = []
        self.known_face_ids = []
        self.load_known_faces()

    def load_known_faces(self):
        known_faces_dir = os.path.join(self.database_path, "known_faces")
        for person_id in os.listdir(known_faces_dir):
            encodings_path = os.path.join(known_faces_dir, person_id, "encodings")
            for encoding_file in os.listdir(encodings_path):
                encoding = np.load(os.path.join(encodings_path, encoding_file))
                self.known_face_encodings.append(encoding)
                self.known_face_ids.append(int(person_id))

    def recognize_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_ids = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_id = "Unknown"

            if True in matches:
                matched_indices = [i for i, match in enumerate(matches) if match]
                counts = {}
                for i in matched_indices:
                    face_id = self.known_face_ids[i]
                    counts[face_id] = counts.get(face_id, 0) + 1
                face_id = max(counts, key=counts.get)

            face_ids.append(str(face_id))

        return face_locations, face_ids

    def draw_results(self, frame, face_locations, face_ids):
        for (top, right, bottom, left), face_id in zip(face_locations, face_ids):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, face_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame