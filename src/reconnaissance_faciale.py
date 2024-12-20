import cv2

# Charger le classifieur Haar-Cascade pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser la webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Lire une image de la webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris pour améliorer les performances
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner des cadres autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher le flux vidéo avec les cadres
    cv2.imshow('Video', frame)

    # Quitter en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
