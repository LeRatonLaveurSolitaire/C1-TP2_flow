import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def calcul_moyenne_ecart_type_N_images(Lpath,affichage=False):
    # Lpath est une liste de N chemins d'accès vers des images
    # On suppose que toutes les images ont la même taille
    N = len(Lpath)
    Limages=[]
    for i in range(N):
        img = cv.imread(Lpath[i], cv.IMREAD_GRAYSCALE)  #ouverture en niveau de gris
        Limages.append(img)
    
    # Pour cahque pixel, on calcule la moyenne et l'écart type
    h, w = Limages[0].shape
    moyenne = np.zeros((h, w), dtype=np.float32)
    ecart_type = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel = [image[i, j] for image in Limages]
            moyenne[i, j] = np.mean(pixel)
            ecart_type[i, j] = np.std(pixel)

    
    cv.imwrite("moyenne_{}.png".format(N), moyenne.astype(np.uint8))
    cv.imwrite("ecart_type_{}.png".format(N), ecart_type.astype(np.uint8))

    # Calcul de l'histogramme de l'écart type
    img_ecart_type = ecart_type.astype(np.uint8)
    Lx = np.arange(256)  # Valeurs de 0 à 255
    Ly, _ = np.histogram(img_ecart_type, bins=256, range=(0, 256))

    # Tracer l'histogramme
    plt.plot(Lx, Ly, label="Histogramme des niveaux de gris")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Fréquence")
    plt.title("Distribution des niveaux de gris")
    plt.grid(True)
    plt.legend()
    plt.savefig("histogramme_ecart_type_{}.png".format(N))
    plt.show(block=False)
    if affichage:
        cv.imshow("Moyenne", moyenne.astype(np.uint8))
        cv.imshow("Ecart type", ecart_type.astype(np.uint8))
        cv.waitKey(0)



if __name__ == "__main__":
    # lecture du dossier images
    Lpath=sorted(os.listdir("images"))
    Lpath=["images/"+path for path in Lpath]
    print(Lpath)
    calcul_moyenne_ecart_type_N_images(Lpath[:])
