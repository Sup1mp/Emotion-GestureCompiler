import os

import pandas as pd

from utils.Mconfusao import Mconfusao
from sklearn.neighbors import KNeighborsClassifier

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
    def __init__(self, gestures:list, train_path:str):

        self.gesture_name = gestures    # all gestures and their names
        self.gesture_counter = {i : 0 for i in self.gesture_name}

        # labels
        self.predicted_label = []
        self.real_label = []

        self.MC = Mconfusao(self.gesture_name, True)    # confusion matrix

        # Criar classificador KNN com k = 5 (numero de vizinhos)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
        self.train_xlsx(train_path)
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
    G.classify_xlsx(data_directory[1])
