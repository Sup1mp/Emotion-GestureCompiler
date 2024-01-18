import cv2
import mediapipe as mp
import numpy as np
import torch

from emotion_detector import EmotionDetector
from imutils.video import FPS
from pandas import DataFrame

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

        self.skeleton = mp.solutions.pose   # skeleton
        self.skeleton_landmarks = DataFrame(
            columns = ['x', 'y', 'z'],
            index = ['nose', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri']
        ).applymap(lambda x:0)

    def cap_landmarks (self, detection):
        # simplifications
        landmark = detection.pose_landmarks.landmark
        m = mp.solutions.pose.PoseLandmark
        memb = [m.NOSE, m.LEFT_SHOULDER, m.RIGHT_SHOULDER, m.LEFT_ELBOW, m.RIGHT_ELBOW, m.LEFT_WRIST, m.RIGHT_WRIST]
        
        # saves landmarks on DataFrame
        for i in range(len(self.skeleton_landmarks)):
            self.skeleton_landmarks.iloc[i] = [landmark[memb[i]].x, landmark[memb[i]].y, landmark[memb[i]].z]
        
        return
    
    def print_landmarks (self, image):
        # print parameters
        font = cv2.FONT_HERSHEY_DUPLEX
        size = 0.7
        color = (0, 255, 0)
        thickness = 1

        # print on image
        for i in range(len(self.skeleton_landmarks)):
            text = [self.skeleton_landmarks.iloc[i][j] for j in range(len(self.skeleton_landmarks.iloc[i]))]
            
            cv2.putText(
                image,
                f'{self.skeleton_landmarks.index[i]}   {list(map(lambda x : round(x, 3), text))}',
                (0, round((1 + i)*size*45)),
                font, size, color, thickness
            )
        return
        
    def video (self, video_path: str = 0):
        
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)      # video capture
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

                # movimente detection
                try:
                    detection = pose.process(self.img)
                    self.cap_landmarks(detection)
                except:
                    pass

                self.print_landmarks(self.img)
                
                # draw skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    self.img,
                    detection.pose_landmarks,
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

