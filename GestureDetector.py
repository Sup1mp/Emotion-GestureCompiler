import os
import cv2
import time

import pandas as pd
import numpy as np
import mediapipe as mp

from utils.Mconfusao import Mconfusao
from sklearn.neighbors import KNeighborsClassifier
from imutils.video import FPS

#%%
# Calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

# função que retorna todos os dados de um diretório
def get_allFiles (data_directory):
    return [f for f in os.listdir(data_directory) if f.endswith('.xlsx')]

def get_trainData (trainData_path):
    X = []  # train data
    Y = []  # target values

    for file in get_allFiles(trainData_path):
        # colect the data
        dados = pd.read_excel(os.path.join(trainData_path, file)).to_numpy()
        
        # saves in array-like
        X.append(extract_features(dados))
        Y.append(file.split('_')[0])    # awnser in the name of the file
    
    return X, Y

# Função para extrair as features dos dados
def extract_features(data):
    mat = data.T @ data
    return mat.flatten()

#%%
class GestureDetector:
    def __init__(
            self, gestures:list,
            train_path:str,
            k:int = 7,
            video: bool = True
    ):
        self.gesture_name = gestures    # all gestures and their names
        self.video = video

        self.recording = False
        self.start_time = time.time()
        self.matrix = np.zeros((1,18))  # matrix with data from many frames
        self.resp = '??'                # current awnser

        self.file_counter = {}
        self.name_order = [
                'ShoulderR_X',
                'ShoulderR_Y',
                'ShoulderR_Z',
                'ShoulderL_X',
                'ShoulderL_Y',
                'ShoulderL_Z',
                'ElbowR_X',
                'ElbowR_Y',
                'ElbowR_Z',
                'ElbowL_X',
                'ElbowL_Y',
                'ElbowL_Z',
                'WristR_X',
                'WristR_Y',
                'WristR_Z',
                'WristL_X',
                'WristL_Y',
                'WristL_Z'
            ]

        # Criar classificador KNN com k = 5 (numero de vizinhos)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

        self.skeleton = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.train_xlsx(train_path) # train KNN
        return
    
    def cap_landmarks (self, frame):

        #  # RECOLOR IMAGE TO RGB
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image.flags.writeable = False
        # # get skeleton from frame
        self.detection = self.skeleton.process(frame)
        # # RECOLOR BACK TO BGR
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # simplifications
        landmark = self.detection.pose_landmarks.landmark
        m = mp.solutions.pose.PoseLandmark
        memb = [
            m.RIGHT_SHOULDER,
            m.LEFT_SHOULDER,
            m.RIGHT_ELBOW,
            m.LEFT_ELBOW,
            m.RIGHT_WRIST,
            m.LEFT_WRIST
        ]

        # reference
        nose = np.array([landmark[m.NOSE].x, landmark[m.NOSE].y, landmark[m.NOSE].z])
        
        # return formated data
        return np.array([
            np.array([
                np.array([landmark[i].x, landmark[i].y, landmark[i].z]) - nose for i in memb
            ]).flatten()
        ])
    
    def process_frame(self, img):

        # MAKE DETECTION
        vector = self.cap_landmarks(img)  # vector of current frame

        # prints results
        if self.video:
            img = self.print_data(img)
        
        if not self.recording:
            angle = calculate_angle(vector[0,3:5], vector[0,9:11], vector[0,15:17])

            if angle < 70 and vector[0,16] < vector[0,4]:
                self.recording = True   # ready for recording
                time.sleep(3)
                self.start_time = time.time()

        elif self.recording and time.time() - self.start_time < 2:
            # records for 2 seconds
            self.matrix = np.concatenate((self.matrix,vector),0)

        else:
            self.recording = False  # stops recoding
            self.matrix = np.concatenate((self.matrix[1:, :],vector),0)
            self.resp = self.classify_video(self.matrix)

        return img

    def record (self, video_path:str = 0):
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)  # video inicialization
        if not cap.isOpened():
            return

        # video reading
        success, img = cap.read()
        fps = FPS().start()

        # image processing
        while success:

            image = self.process_frame(img)

            cv2.imshow('Output', image)

            # resets matrix
            if not self.recording:
                self.matrix = np.zeros((1,18))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            fps.update()
            success, img = cap.read()

        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        return

    def print_data (self, image):
        # print parameters
        font = cv2.FONT_HERSHEY_DUPLEX
        size = 0.7
        color_text = (0, 255, 0)
        color_1 = (145, 45, 30)
        color_2 = (0, 0, 245)
        thickness = 1
        circular_radius = 2

         # RENDER DETECTIONS
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            self.detection.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(
                color=color_1,
                thickness=thickness,
                circle_radius=circular_radius),
            mp.solutions.drawing_utils.DrawingSpec(
                color=color_2,
                thickness=thickness,
                circle_radius=circular_radius)
        )

        # render results
        cv2.putText(
            image,
            f'Predic.: {self.resp}',
            (0, round(size*45)),
            font, size, color_text, thickness
        )
        return image
    
    def train_xlsx (self, trainData_path: str):
        subfiles = [f.path for f in os.scandir(trainData_path) if f.is_dir()]
        X = []  # train data
        Y = []  # target values

        for sub in subfiles:
            all_files = get_allFiles(sub)
            self.file_counter[str(sub.split('\\')[1])] = len(all_files)     # creates file counter

            for file in all_files:
                # colect the data
                dados = pd.read_excel(os.path.join(sub, file)).to_numpy()
                
                # saves in array-like
                X.append(extract_features(dados))
                Y.append(file.split('_')[0])    # awnser in the name of the file
        
        self.knn_classifier.fit(X, Y)
        return
    
    def classify_video (self, matrix, threshold:float=0.9):
            # makes prediction
            prob = self.knn_classifier.predict_proba([extract_features(matrix)])[0]

            # return prediction
            return self.knn_classifier.classes_[prob.argmax()] if max(prob) >= threshold else 'I'

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

    def saves_to_dataBase (self, resp):
        # Salvar o arquivo 
        file_name = f"Base_de_dados/{resp}/{resp}_{self.file_counter[resp]+1:02d}.xlsx"
        self.file_counter[resp] = self.file_counter[resp]+1     # adds to the file counter
        
        # organizes and saves
        df = pd.DataFrame(self.matrix, columns= self.name_order)
        df.to_excel(file_name, index=False, engine='openpyxl')
        
        print(f"Arquivo {file_name} salvo.")
        
#%%
if __name__ == "__main__":
    # data_directory = [
    #     'aleatorio/Base_de_dados_20_2',
    #     'validar_pa/validar_30_2'
    # ]
    data_directory = 'Base_de_dados'
    G = GestureDetector(['A', 'B', 'C', 'D', 'E'], data_directory)
    # G.classify_xlsx(data_directory[1])
    G.record()