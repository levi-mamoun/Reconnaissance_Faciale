import cv2
import face_recognition

class FaceDetection:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.face_locations = face_recognition.face_locations(rgb_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
        return self.face_locations, self.face_encodings
