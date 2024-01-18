import cv2
import mediapipe as mp
import numpy as np
import torch

from emotion_detector import EmotionDetector
from imutils.video import FPS

class EmotionGestureCompiler(EmotionDetector):
    def __init__(
        self,
        model_name: str = "resnet18.onnx",
        model_option: str = "onnx",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        providers: int = 1,
        fp16=False,
        num_faces=None,
    ):
        super().__init__(
        model_name = model_name,
        model_option = model_option,
        backend_option = backend_option,
        providers = providers,
        fp16=fp16,
        num_faces=num_faces,
        )

        self.skeleton = mp.solutions.pose
        
    def video (self, video_path: str = 0):
        
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        success, self.img = cap.read()
        self.height, self.width = self.img.shape[:2]
        fps = FPS().start()

        with self.skeleton.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while success:

                # facial detection
                self.process_frame()
                
                # draw skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    self.img,
                    pose.process(self.img).pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                
                cv2.imshow("Capturing", self.img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                fps.update()
                success, self.img = cap.read()
            
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    emo = EmotionGestureCompiler()
    emo.video(0)

