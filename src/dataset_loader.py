import os
import pickle
import face_recognition

class DatasetLoader:
    def __init__(self, dataset_dir="./src/dataset/person", db_path="./src/face_data.pkl"):
        self.dataset_dir = dataset_dir
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []

    def load_from_db(self):
        """Charger les données encodées depuis la base de données (fichier pickle)."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as db_file:
                data = pickle.load(db_file)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
                print("Données chargées depuis la base de données.")
        else:
            print("Aucune base de données trouvée. Chargement initial requis.")

    def update_db(self):
        """Mettre à jour la base de données avec les nouvelles données encodées."""
        with open(self.db_path, "wb") as db_file:
            data = {
                "encodings": self.known_face_encodings,
                "names": self.known_face_names
            }
            pickle.dump(data, db_file)
            print("Base de données mise à jour avec succès.")

    def process_new_images(self):
        """Traiter uniquement les nouvelles images non encodées."""
        processed_images = set(self.known_face_names)
        for person_name in os.listdir(self.dataset_dir):
            person_path = os.path.join(self.dataset_dir, person_name)
            if os.path.isdir(person_path):
                for image_name in os.listdir(person_path):
                    if person_name not in processed_images:
                        image_path = os.path.join(person_path, image_name)
                        if os.path.isfile(image_path) and (image_path.endswith(".jpg") or image_path.endswith(".png")):
                            try:
                                print(f"Traitement de l'image : {image_name}")
                                image = face_recognition.load_image_file(image_path)
                                encodings = face_recognition.face_encodings(image)
                                if encodings:
                                    self.known_face_encodings.append(encodings[0])
                                    self.known_face_names.append(person_name)
                            except Exception as e:
                                print(f"Erreur lors du traitement de {image_path}: {e}")
        self.update_db()

    def load(self):
        """Combiner le chargement de la base de données et le traitement des nouvelles images."""
        self.load_from_db()
        self.process_new_images()
        return self.known_face_encodings, self.known_face_names
