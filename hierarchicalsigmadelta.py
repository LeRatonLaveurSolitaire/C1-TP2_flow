import numpy as np


class HierarchicalSigmaDeltaMotionDetector:
    def __init__(self, width, height, N=4, Vmin=10, Vmax=240):
        self.width = width
        self.height = height
        self.N = N
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.M = np.zeros((height, width), dtype=np.uint8)
        self.V = np.full((height, width), Vmin, dtype=np.uint8)
        self.O = np.zeros((height, width), dtype=np.uint8)
        self.E = np.zeros((height, width), dtype=np.uint8)

    def update(self, frame):
        frame = frame.astype(np.uint8)

        # Step 1: Update M (background estimation)
        update_mask = self.E == 0
        self.M[update_mask & (frame > self.M)] += 1
        self.M[update_mask & (frame < self.M)] -= 1

        # Step 2: Compute O (frame difference)
        self.O = np.abs(frame - self.M)

        # Step 3: Update V (variance estimation)
        self.V[self.O > self.V / self.N] += 1
        self.V[self.O < self.V / self.N] -= 1
        np.clip(self.V, self.Vmin, self.Vmax, out=self.V)

        # Step 4: Estimate E (motion detection)
        self.E = (self.O > self.V).astype(np.uint8)

        return self.E * 255


def process_video(video_path, detector,image_seuil=None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Seuillage
        if image_seuil is not None:
            gray = gray * (image_seuil // 255)
            
        motion_mask = detector.update(gray)

        if i == 32:
            cv2.imwrite(f"./mu_sigma/sigmadelta_32.png",motion_mask.astype(np.uint8))

        cv2.imshow("Motion Detection", motion_mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i+=1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.avi"
    width, height = 384, 288  # Adjust to your video's dimensions
    detector = HierarchicalSigmaDeltaMotionDetector(width, height, N=20)
    process_video(video_path, detector)
