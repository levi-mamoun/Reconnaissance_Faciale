from src.dataset_loader import DatasetLoader
from src.real_time_recognition import RealTimeRecognition
from src.video_recognition import VideoRecognition
from src.image_recognition import ImageRecognition
from src.face_detection import FaceDetection

def main():
    dataset_loader = DatasetLoader()
    known_face_encodings, known_face_names = dataset_loader.load()

    if len(known_face_encodings) == 0:
        print("Aucun visage chargé. Vérifiez votre dataset.")
        return

    print("Dataset chargé avec succès.")
    print("Choisissez le type de traitement :")
    print("1. Flux en direct depuis la webcam")
    print("2. Vidéo enregistrée")
    print("3. Image enregistrée")

    choice = input("Entrez 1, 2 ou 3 : ")

    if choice == '1':
        real_time_recognition = RealTimeRecognition(known_face_encodings, known_face_names)
        real_time_recognition.recognize()
    elif choice == '2':
        video_path = input("Entrez le chemin de la vidéo enregistrée : ")
        video_recognition = VideoRecognition(video_path, known_face_encodings, known_face_names)
        video_recognition.recognize()
    elif choice == '3':
        image_path = input("Entrez le chemin de l'image enregistrée : ")
        image_recognition = ImageRecognition(image_path, known_face_encodings, known_face_names)
        image_recognition.recognize()
    else:
        print("Choix invalide.")

if __name__ == "__main__":
    main()
