import cv2
import mediapipe as mp
import numpy as np
import torch

from emotion_detector import EmotionDetector
from GestureDetector import GestureDetector
from imutils.video import FPS
from pandas import DataFrame

class EmotionGestureCompiler:
    def __init__(
        self,
        model_name: str = "resnet18.onnx",
        model_option: str = "onnx",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        providers: int = 1,
        fp16=False,
        num_faces=None,
        train_path: str = 'aleatorio/Base_de_dados_20_2'
    ):
        self.Emotion = EmotionDetector(model_name, model_option, backend_option, providers, fp16, num_faces)
        self.Gesture = GestureDetector(['A', 'B', 'C', 'D', 'E'], train_path)

        return
        
    def video (self, video_path: str = 0):
        
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)      # video capture
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        success, img = cap.read()
        fps = FPS().start()

        with self.Gesture.skeleton.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while success:

                # facial detection
                self.Emotion.process_frame(img)

                # movimente detection
                detection = pose.process(img)

                try:
                    self.Gesture.cap_landmarks(detection)
                except:
                    print('Could not capture skeleton properly')

                # self.print_landmark_data(img)

                # draw skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    img,
                    detection.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                
                cv2.imshow("Capturing", img)   # draw image
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                fps.update()
                success, img = cap.read()
            
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

