import os
import cv2

import pandas as pd
import numpy as np
import mediapipe as mp

from utils.Mconfusao import Mconfusao
from sklearn.neighbors import KNeighborsClassifier
from imutils.video import FPS

#%%
def get_allFiles (data_directory):
    # função que retorna todos os dados de um diretório
    return [f for f in os.listdir(data_directory) if f.endswith('.xlsx')]

def extract_features(data):
    # Função para extrair as features dos dados
    mat = data.T @ data
    return mat.flatten()

#%%
class GestureDetector:
    def __init__(
            self, gestures:list,
            train_path:str,
            k:int = 5
    ):

        self.gesture_name = gestures    # all gestures and their names

        # Criar classificador KNN com k = 5 (numero de vizinhos)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

        self.skeleton = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.frames = []        # frames of the video (list of dataframes)
        self.frame_size = 10    # max size of video in frames

        # pd.DataFrame(
        #     [np.array([landmark[i].x, landmark[i].y, landmark[i].z]) for i in memb],
        #     columns = ['x', 'y', 'z'],
        #     index = ['nose', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri']
        # )

        self.train_xlsx(train_path)
        return
    
    def cap_landmarks (self, new_frame):

        detection = self.skeleton.process(new_frame)

        # simplifications
        landmark = detection.pose_landmarks.landmark
        m = mp.solutions.pose.PoseLandmark
        memb = [m.NOSE, m.LEFT_SHOULDER, m.RIGHT_SHOULDER, m.LEFT_ELBOW, m.RIGHT_ELBOW, m.LEFT_WRIST, m.RIGHT_WRIST]

        return [np.array([landmark[i].x, landmark[i].y, landmark[i].z]) for i in memb]
    
    def video_cap (self, new_frame):
        # # organize and save a few frames
        # if len(self.frames) < self.frame_size:
        #     self.frames.append(self.cap_landmarks(new_frame))
        # else:
        #     self.frames = self.frames[1:]
        #     self.video_cap(new_frame)

        if len(self.frames) < self.frame_size:
            self.frames.append(self.cap_landmarks(new_frame))
        else:
            self.frames[0] = self.cap_landmarks(new_frame)
            self.frames = np.roll(self.frames, -1)
        return

    def process_video (self, video_path:str = 0):
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        success, img = cap.read()
        #self.height, self.width = self.img.shape[:2]
        fps = FPS().start()

        # image processing
        while success:

            self.video_cap(img)
            print(self.frames)

            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            fps.update()
            success, img = cap.read()
        
        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

    def print_landmark_data (self, image):
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

    def train_xlsx (self, trainData_path:str):
        X = []  # train data
        Y = []  # target values

        for file in get_allFiles(trainData_path):
            # colect the data
            dados = pd.read_excel(os.path.join(trainData_path, file)).to_numpy()
            
            # saves in array-like
            X.append(extract_features(dados))
            Y.append(file.split('_')[0])    # awnser in the name of the file
        
        self.knn_classifier.fit(X, Y)
        return

    def classify_xlsx(self, validation_path:str, threshold:float=0.9):
        # Função para classificar um novo arquivo xlsx

        self.gesture_counter = {i : 0 for i in self.gesture_name}

        # labels
        self.predicted_label = []
        self.real_label = []

        self.MC = Mconfusao(self.gesture_name, True)    # confusion matrix

        # getes results
        for file in get_allFiles(validation_path):
            # collect the data
            dados = pd.read_excel(os.path.join(validation_path, file)).to_numpy()

            # makes prediction
            prob = self.knn_classifier.predict_proba([extract_features(dados)])[0]

            # saves predicted and real ones
            self.real_label.append(file.split('_')[0])
            self.predicted_label.append(self.knn_classifier.classes_[prob.argmax()] if max(prob) >= threshold else 'I')
        
        # process results
        for i in range(len(self.real_label)):
            self.MC.add_one(self.real_label[i], self.predicted_label[i])

            if self.real_label[i] == self.predicted_label[i]:
                self.gesture_counter[self.real_label[i]] += 1      # counts individualy

        # shows confusion matrix
        self.MC.render()
        self.MC.great_analysis()
        return

#%%
if __name__ == "__main__":
    data_directory = [
        'aleatorio/Base_de_dados_20_2',
        'validar_pa/validar_30_2'
    ]
    G = GestureDetector(['A', 'B', 'C', 'D', 'E'], data_directory[0])
    # G.classify_xlsx(data_directory[1])
    G.process_video()
