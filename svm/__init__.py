import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self, group_name=None):
        
        self.group_name = group_name
        self.out_encoder = LabelEncoder()
        self.in_encoder = Normalizer(norm='l2')  
        

    def train(self, data, group_name):
        
        self.load(group_name)
        
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        
        trainX = self.in_encoder.transform(trainX)
        testX = self.in_encoder.transform(testX)

        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        testy = self.out_encoder.transform(testy)

        # fit model
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(trainX, trainy)

        # predict
        yhat_train = self.model.predict(trainX)
        yhat_test = self.model.predict(testX)

        # score
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)

        # summarize
        print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
        self.save()
        
        return  score_train*100, score_test*100


        
    def predict(self, encode_image, group_name):
        self.load(group_name)
        encode_image = self.in_encoder.transform(encode_image)
        yhat_class = self.model.predict(encode_image)
        yhat_prob = self.model.predict_proba(encode_image)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = self.out_encoder.inverse_transform(yhat_class)
        return class_probability, predict_names

    
    def load(self, group_name):
        if group_name == self.group_name:
            return
        self.group_name = group_name

        trained_file_path = os.path.join("./svm/models/", f"{self.group_name}.sav")
        out_encoder_path = os.path.join("./svm/models/", f"{self.group_name}.npy")

        if os.path.isfile(trained_file_path):
            self.model = pickle.load(open(trained_file_path, 'rb'))

        if os.path.isfile(out_encoder_path):
            self.out_encoder.classes_ = np.load(out_encoder_path)
    
    def save(self):

        trained_file_path = os.path.join("./svm/models/", f"{self.group_name}.sav")
        out_encoder_path = os.path.join("./svm/models/", f"{self.group_name}.npy")
        pickle.dump(self.model, open(trained_file_path, 'wb'))
        np.save(out_encoder_path, self.out_encoder.classes_)
        print("Save Success")