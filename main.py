import cv2
import threading
from face_detection.face_detector import FaceDetector
from face_recognition.face_recognizer import FaceRecognizer

DATABASE_PATH = "database"

def process_camera(camera_id, detector, recognizer):
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_locations, face_ids = recognizer.recognize_face(frame)
        frame = recognizer.draw_results(frame, face_locations, face_ids)

        cv2.imshow(f'Camera {camera_id}', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            new_person_id = detector.capture_person(camera_id)
            recognizer.load_known_faces()  # Reload known faces after adding a new person

    cap.release()
    cv2.destroyAllWindows()

def main():
    detector = FaceDetector(DATABASE_PATH)
    recognizer = FaceRecognizer(DATABASE_PATH)

    # Create threads for each camera
    camera_threads = []
    for camera_id in range(2):  # Assuming we have two cameras with IDs 0 and 1
        thread = threading.Thread(target=process_camera, args=(camera_id, detector, recognizer))
        camera_threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in camera_threads:
        thread.join()

if __name__ == "__main__":
    main()