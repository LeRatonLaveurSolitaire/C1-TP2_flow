import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def compute_movement(N = 5, img_start = 30, offset = 2):
    imgs = []
    for i in range(N):
        imgs.append(cv.imread(f"./images/image_{img_start+i:03}.jpg", cv.IMREAD_GRAYSCALE))
        print(f"Opening ./images/image_{img_start+i:03}.jpg")
    
    imgs = np.array(imgs)
    _, h, w = imgs.shape

    mean_img = np.zeros(( h, w ))
    std_img = np.zeros(( h, w ))

    for i in range(h):
        for j in range(w):
            mean_img[i,j] = np.mean(imgs[:,i,j])
            std_img[i,j] = np.std(imgs[:,i,j])

    mean_std = np.mean(std_img)

    background = np.full((h,w),255)


    
    for i in range(h):
        for j in range(w):
            #if (imgs[offset,i,j] < (mean_img[i,j] - alpha * std_img[i,j])) or  (imgs[offset,i,j] > (mean_img[i,j] + alpha * std_img[i,j])):
            if std_img[i,j] > 15:
               background[i,j] = 0 

    cv.imwrite(f"./mu_sigma/img_{img_start + offset}.png", imgs[offset].astype(np.uint8))
    cv.imwrite(f"./mu_sigma/mean_{img_start + offset}.png", mean_img.astype(np.uint8))
    cv.imwrite(f"./mu_sigma/std_{img_start + offset}.png", std_img.astype(np.uint8))
    cv.imwrite(f"./mu_sigma/background_{img_start + offset}.png", background.astype(np.uint8))
 

def pixel_200_150_variations():
    pixel_values = []

    video_path = "video.avi"
    width, height = 384, 288
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pixel_values.append(gray[200,150])

    cap.release()

    plt.plot(list(range(len(pixel_values))),pixel_values)
    plt.show()


def main():

    compute_movement()
    # pixel_200_150_variations()



if __name__ == "__main__":
    main()