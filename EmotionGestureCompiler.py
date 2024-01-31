import cv2
import numpy as np
import torch

from emotion_detector import EmotionDetector
from GestureDetector import GestureDetector
from imutils.video import FPS

class EmotionGestureCompiler:
    def __init__(
        self,
        model_name: str = "resnet18.onnx",
        model_option: str = "onnx",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        providers: int = 1,
        fp16: bool = False,
        num_faces: int = 1,
        train_path: str = 'Base_de_dados',
        k : int = 7,
        video: bool = True
    ):
        self.gestures_list = ['A', 'B', 'C', 'D', 'E']
        self.emotions_list = {0: "BAD", 1: "GOOD", 2: "NEUTRAL"}
        
        self.Emotion = EmotionDetector(model_name, model_option, backend_option, providers, fp16, num_faces)
        self.Gesture = GestureDetector(self.gestures_list, train_path, k, video)

        return
    
    def get_emotion (self, img):

        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.Emotion.face_model.setInput(blob)
        predictions = self.Emotion.face_model.forward()

        for i in range(predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                (x_min, y_min, x_max, y_max) = bbox.astype("int")

                # draws red rectangle
                cv2.rectangle(
                    img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2
                )

                face = img[y_min:y_max, x_min:x_max]

                # makes prediction 
                emotion = self.Emotion.recognize_emotion(face)

                # writes emotion name
                cv2.putText(
                    img,
                    self.emotions_list[emotion],
                    (x_min + 5, y_min - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return emotion, img

    def video (self, video_path: str = 0):
        
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)      # creates capture
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        # image reading
        success, img = cap.read()
        fps = FPS().start()

        # starts video
        while success:

            emotion, img = self.get_emotion(img)    # captures emotion

            if not self.Gesture.resp in self.gestures_list:
                img = self.Gesture.process_frame(img)   # captures gesture

            else:
                # match predictions at will
                match self.emotions_list[emotion]:
                    case "GOOD":        # confirms prediction
                        #self.Gesture.saves_to_dataBase()
                        print("BOM GAROTO")

                    case "BAD":         # rejects prediction
                        self.Gesture.reset_pred()
                        print("MAU GAROTO")

                    case "NEUTRAL":     # does nothing (indecision/analysin)
                        print("GAROTO NEUTRO")
                        pass

            cv2.imshow("Capturing", img)   # draw image
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # image update
            fps.update()
            success, img = cap.read()
        
        # ends transmition
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

