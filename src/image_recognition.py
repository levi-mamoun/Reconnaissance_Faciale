import cv2
import face_recognition
import numpy as np

class ImageRecognition:
    def __init__(self, image_path, known_face_encodings, known_face_names):
        self.image_path = image_path
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

    def recognize(self):
        image = face_recognition.load_image_file(self.image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Inconnu"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if matches else None

            if best_match_index is not None and matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            top, right, bottom, left = face_location

            # Améliorer le cadre : Agrandir un peu la détection du visage
            padding = 10  # Ajouter un padding autour du visage pour mieux le cadrer
            top, right, bottom, left = top - padding, right + padding, bottom + padding, left - padding

            # S'assurer que les coordonnées restent dans les limites de l'image
            top, right, bottom, left = max(0, top), min(rgb_image.shape[1], right), min(rgb_image.shape[0], bottom), max(0, left)

            # Dessiner le rectangle autour du visage
            cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 3)

            # Améliorer l'affichage du texte (police plus grande, épaisseur plus grande)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Taille de la police
            thickness = 3  # Épaisseur du texte
            cv2.putText(rgb_image, name, (left, top - 10), font, font_scale, (0, 255, 0), thickness)

        # Redimensionner l'image pour l'affichage
        resized_image = cv2.resize(rgb_image, (800, 620))

        # Afficher l'image redimensionnée
        cv2.imshow("Reconnaissance Faciale sur Image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
