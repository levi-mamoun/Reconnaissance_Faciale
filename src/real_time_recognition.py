import cv2
import numpy as np
import face_recognition
from src.face_detection import FaceDetection

class RealTimeRecognition:
    def __init__(self, known_face_encodings, known_face_names, tolerance=0.6):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.tolerance = tolerance
        self.face_detection = FaceDetection()

    def recognize(self):
        # Ouvre la webcam (index 0 pour la caméra principale)
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erreur : La caméra ne s'est pas ouverte correctement.")
            return

        print("Appuyez sur 'q' pour quitter.")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erreur lors de la lecture de la webcam.")
                break

            # Conversion en RGB pour la détection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Détecter les visages
            face_locations, face_encodings = self.face_detection.detect_faces(rgb_frame)

            # Parcourir les visages détectés
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, self.tolerance)
                name = "Inconnu"
                similarity_percentage = 0.0

                # Calculer les distances des visages et trouver la meilleure correspondance
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        similarity_percentage = (1 - face_distances[best_match_index]) * 100

                # Dessiner le cadre autour du visage
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Ajouter le texte avec le nom et le pourcentage de similitude
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                thickness = 2
                text = f"{name} ({similarity_percentage:.2f}%)"
                cv2.putText(frame, text, (left, top - 10), font, font_scale, (0, 255, 0), thickness)

            # Afficher le flux vidéo en temps réel
            cv2.imshow("Reconnaissance Faciale en Temps Réel", frame)

            # Appuyer sur 'q' pour quitter
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libérer les ressources
        video_capture.release()
        cv2.destroyAllWindows()
