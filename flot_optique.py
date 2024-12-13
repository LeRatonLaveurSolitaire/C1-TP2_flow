import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from flowToColor import *
from hierarchicalsigmadelta import *


def calcul_moyenne_ecart_type_N_images(Lpath,affichage=False):
    # Lpath est une liste de N chemins d'accès vers des images
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

    if not(affichage):
        return moyenne, ecart_type

    
    cv.imwrite("moyenne_{}.png".format(N), moyenne.astype(np.uint8))
    cv.imwrite("ecart_type_{}.png".format(N), ecart_type.astype(np.uint8))

    # Calcul de l'histogramme de l'écart type
    img_ecart_type = ecart_type.astype(np.uint8)
    Lx = np.arange(256)  # Valeurs de 0 à 255
    Ly, _ = np.histogram(img_ecart_type, bins=256, range=(0, 256))
    seuil=Lx[np.argmin(Ly[:50])]

    # Tracer l'histogramme
    plt.plot(Lx, Ly,'b', label="Histogramme des niveaux de gris")
    plt.plot(seuil,Ly[seuil],'ro',label="Seuil")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Fréquence")
    plt.title("Distribution des niveaux de gris")
    plt.grid(True)
    plt.legend()
    plt.savefig("histogramme_ecart_type_{}.png".format(N))
    plt.show(block=False)
    cv.imshow("Moyenne", moyenne.astype(np.uint8))
    cv.imshow("Ecart type", ecart_type.astype(np.uint8))
    cv.waitKey(0)

    return moyenne, ecart_type
    # image_seuillee = np.zeros((h, w), dtype=np.uint8)
    # for i in range(h):
    #     for j in range(w):
    #         if ecart_type[i, j] > seuil:
    #             image_seuillee[i, j] = 255
    #         else:
    #             image_seuillee[i, j] = 0
    # if affichage:
    #     cv.imshow("Image seuillée", image_seuillee)
    #     cv.waitKey(0)
    # cv.imwrite("image_seuillee_{}.png".format(N), image_seuillee)


def horn_schunck(Lpath_images, alpha=0, n_iter=20):
    # Initialiser les champs u et v
    image0=cv.imread(Lpath_images[0])
    img0=cv.cvtColor(image0, cv.COLOR_BGR2GRAY)
    # cv.imshow("image",img0)
    # cv.waitKey(0)
    h, w = img0.shape
    
    for t in range(len(Lpath_images) - 1):
        u = np.zeros((h, w))
        v = np.zeros((h, w))
        print(f"Processing frames {t} and {t+1}...")
        image1=cv.imread(Lpath_images[t])
        img1=cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2=cv.imread(Lpath_images[0])
        img2=cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        # img1 = gaussian_filter(images[t], sigma=1)
        # img2 = gaussian_filter(images[t+1], sigma=1)

        # Calcul des dérivées
        Ex = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=5) + cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=5)
        Ey = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=5) + cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=5)
        Et = img2 - img1

        # Iterative solution
        for _ in range(n_iter):
            # Calcul des Laplaciens
            u_avg = cv.blur(u, (3, 3))
            v_avg = cv.blur(v, (3, 3))

            # Mise à jour des champs de flot
            num = (Ex * u_avg + Ey * v_avg + Et)
            denom = alpha**2 + Ex**2 + Ey**2
            u = u_avg - (Ex * num) / denom
            v = v_avg - (Ey * num) / denom
    
        img = flowToColor(u, v)

        print("flot_optique/{}".format(Lpath_images[t].split('/')[1]))
        cv.imwrite("flot_optique/{}".format(Lpath_images[t].split('/')[1]), img)
        # cv.imshow("Flot optique", img)
    cv.waitKey(5000)
        
    return u, v

def creationDeLImageSeuil(image_path,image_moyenne,image_ecart_type):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    alpha = 1 # A ajuster
    h, w = image.shape
    image_seuil = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if image_ecart_type[i,j]>15:
                image_seuil[i,j]=255
            else:
                image_seuil[i,j]=0
    print("image_apres_seuil/{}".format(image_path.split('/')[1]))
    cv.imwrite("image_apres_seuil/{}".format(image_path.split('/')[1]), image_seuil)
    return image_seuil




if __name__ == "__main__":
    # lecture du dossier images et selection des pixels
    Lpath_brut=sorted(os.listdir("images"))
    Lpath=["images/"+path for path in Lpath_brut]
    # print(Lpath)


    n=2# nombre d'images 
    for i in range(len(Lpath[:40])):
        print(i)
        avg, sigma=calcul_moyenne_ecart_type_N_images(Lpath[max(0,i-n):min(len(Lpath),i+n)],affichage=False)
        creationDeLImageSeuil(Lpath[i],avg,sigma)



    # from hierarchicalsigmadelta import *
    # image_seuil=cv.imread("image_seuillee_614.png", cv.IMREAD_GRAYSCALE)
    # video_path = "video.avi"
    # width, height = 384, 288  # Adjust to your video's dimensions
    # detector = HierarchicalSigmaDeltaMotionDetector(width, height)
    # process_video(video_path, detector,image_seuil)

    # image_seuil=cv.imread("image_seuillee_614.png", cv.IMREAD_GRAYSCALE)
    # Lpath=sorted(os.listdir("images"))
    # for img_path in Lpath:
    #     print(img_path)
    #     img = cv.imread("images/"+img_path, cv.IMREAD_GRAYSCALE)
    #     img = img * (image_seuil // 255)
    #     cv.imwrite("images_seuillees/"+img_path, img)

    Lpath=["image_apres_seuil/"+path for path in Lpath_brut[:40]]
    horn_schunck(Lpath[:40], alpha=10, n_iter=100)
    

    # image flot optique faite
    Lpath_flot=sorted(os.listdir("flot_optique"))
    for image in Lpath_flot[:]:
        print(image)
        img=cv.imread("flot_optique/{}".format(image))
        print("image_apres_seuil/{}".format(image))

        img_seuil=cv.imread("image_apres_seuil/{}".format(image))
        img_seuil.shape
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img_seuil[i,j,0]==0:
                    img[i,j,0]=255
                    img[i,j,1]=255
                    img[i,j,2]=255
        cv.imwrite("flot_apres_masquage/{}".format(image), img)

    
