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
        train_path: str = 'Base_de_dados'
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

        while success:

            # facial detection
            self.Emotion.process_frame(img)
            self.Gesture.process_frame(img)
            
            cv2.imshow("Capturing", img)   # draw image
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            fps.update()
            success, img = cap.read()
            
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

