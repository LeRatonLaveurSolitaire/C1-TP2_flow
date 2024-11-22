import numpy as np


class SigmaDeltaMotionDetector:
    def __init__(self, width, height, N=4):
        self.width = width
        self.height = height
        self.N = N
        self.M = np.zeros((height, width), dtype=np.uint8)
        self.V = np.zeros((height, width), dtype=np.uint8)
        self.O = np.zeros((height, width), dtype=np.uint8)
        self.D = np.zeros((height, width), dtype=np.uint8)

    def update(self, frame):
        frame = frame.astype(np.uint8)

        # Update M (background estimation)
        self.M[frame > self.M] += 1
        self.M[frame < self.M] -= 1

        # Update V (variance estimation)
        self.V[np.abs(frame - self.M) > self.V] += 1
        self.V[np.abs(frame - self.M) <= self.V] -= 1

        # Compute O (foreground detection)
        self.O = np.abs(frame - self.M) > self.N * self.V

        # Update D (motion detection)
        self.D = np.logical_or(self.D, self.O)
        self.D = np.logical_and(self.D, self.O)

        return self.D.astype(np.uint8) * 255


def process_video(video_path, detector):
    import cv2

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_mask = detector.update(gray)

        cv2.imshow("Motion Detection", motion_mask)
        cv2.imshow("Original", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.avi"
    width, height = 384, 288  # Adjust to your video's dimensions
    detector = SigmaDeltaMotionDetector(width, height, N =2)
    process_video(video_path, detector)
