import cv2
import numpy as np

def LAB(path):
    print(path)
    # Chargement de l'image
    image = cv2.imread(path)
    # cv2.imshow('Image originale', image)
    # cv2.waitKey(0)

    # Conversion de l'image en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extraction du canal de valeur (V)
    v_channel = hsv[:, :, 2]

    # Application d'un flou gaussien pour lisser l'éclairage (estimation du fond)
    blurred = cv2.GaussianBlur(v_channel, (55, 55), 0)

    # Soustraction du fond pour normaliser l'éclairage
    normalized = cv2.subtract(v_channel, blurred)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    # Remplacement du canal V par l'image normalisée
    hsv[:, :, 2] = normalized

    # Conversion de l'image HSV en RGB
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite('HSV_'+path, result)


    # Conversion de l'image en LAB
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre morphologique pour détecter les ombres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Soustraction de l'arrière-plan
    shadow_free = cv2.divide(gray, background, scale=255)

    # Optionnel : Réduction du bruit pour un meilleur rendu
    shadow_free = cv2.GaussianBlur(shadow_free, (5, 5), 0)

    # Conversion en image couleur (pour combiner avec l'image originale si nécessaire)
    result1 = cv2.cvtColor(shadow_free, cv2.COLOR_GRAY2BGR)

    cv2.imwrite('LAB_'+path, result1)
    print(path)

for j in range(5):
    for i in range(0, 10):
        if i!=0 and j!=0:
            LAB('images/image_0{}{}.jpg'.format(j, i))

# cv2.imshow('Image LAB', result1)
# cv2.imshow('Image originale', image)
# cv2.imshow('Image HSV', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
