# Projet : Reconnaissance Faciale en Temps Réel

## Description
Ce projet a pour but de développer un système de **reconnaissance faciale en temps réel**. En utilisant Python et des bibliothèques comme OpenCV et `face_recognition`, le système détecte et reconnaît les visages capturés par une webcam ou une vidéo préenregistrée.

Le projet suit les étapes suivantes :
1. Détection des visages dans une vidéo en direct ou préenregistrée.
2. Identification des visages détectés, en les associant à des personnes connues à partir d'un dataset préalablement entraîné.

## Prérequis

Avant de lancer le projet, assurez-vous que vous avez installé les bibliothèques nécessaires :

- Python 3.x
- OpenCV
- Dlib
- face-recognition

Les bibliothèques peuvent être installées avec `pip` :

```bash
pip install opencv-python dlib numpy face-recognition
