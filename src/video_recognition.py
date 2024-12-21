import cv2
import face_recognition
import numpy as np
from src.face_detection import FaceDetection


class VideoRecognition:
    def __init__(self, video_path, known_face_encodings, known_face_names, tolerance=0.6):
        self.video_path = video_path
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.face_detection = FaceDetection()
        self.tolerance = tolerance  # Ajuster la tolérance pour la correspondance des visages

    def recognize(self):
        video_capture = cv2.VideoCapture(self.video_path)
        if not video_capture.isOpened():
            print("Erreur : Impossible d'ouvrir la vidéo.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Fin de la vidéo ou erreur de lecture.")
                break

            # Redimensionner la frame pour accélérer le traitement
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Détection des visages
            face_locations, face_encodings = self.face_detection.detect_faces(rgb_small_frame)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
                name = "Inconnu"
                confidence = 0  # Initialisation du pourcentage de similitude

                # Calcul de la distance minimale
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if matches else None

                if best_match_index is not None and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100  # Convertir en pourcentage

                # Agrandir les coordonnées du rectangle
                top, right, bottom, left = [coord * 4 for coord in face_location]
                padding = 20
                top, right, bottom, left = top - padding, right + padding, bottom + padding, left - padding

                # S'assurer que les coordonnées restent dans les limites de l'image
                top, right, bottom, left = max(0, top), min(frame.shape[1], right), min(frame.shape[0], bottom), max(0, left)

                # Dessiner le rectangle autour du visage
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

                # Afficher le nom et le pourcentage de correspondance
                text = f"{name} ({confidence:.2f}%)"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                cv2.putText(frame, text, (left, top - 15), font, font_scale, (0, 255, 0), thickness)

            # Afficher la frame avec les annotations
            cv2.imshow("Reconnaissance Faciale sur Vidéo", frame)

            # Quitter si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
