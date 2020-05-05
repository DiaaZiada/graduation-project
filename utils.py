import os
import numpy as np
from PIL import Image
# from mtcnn.mtcnn import MTCNN
from svm import Classifier
from encoder import Encoder
from face_detection import FaceDetection
from sklearn.utils import shuffle
import cv2
import pickle

classifier = Classifier()
encoder = Encoder()
fd = FaceDetection("./face_detection/face_detection/")

def box_area(box):
    sx, sy, ex, ey = box
    return (ex-sx) * (ey-sy)

def max_area(lis):
    areas = []
    for i in range(len(lis)):
        areas.append(box_area(lis[i]))
    return np.argmax(areas)
def extract_face(image, required_size=(160, 160)):
   
    pixels = np.asarray(image)

    results = fd(pixels)
    i = 0
    if len(results) == 0:
        return 0
    if len(results) > 1:
        i = max_area(results)

    sx, sy, ex, ey = results[i]
    

    face = pixels[sy:ey, sx:ex]
    
    image = cv2.resize(face,required_size)
    face_array = np.asarray(image)
    return face_array

def train(ids, group_name):
    # knownX, knownY = data
    # coded_knownX = encoder(knownX)
    print("aa\n\n\n")
    coded_unknownX = np.load("./latent_variables/unknown.npy")
    unknownY = ["unknown" for _ in range(coded_unknownX.shape[0])]
    print(coded_unknownX.shape,len(unknownY),'\n\n\n')
    trainX = []
    trainY = []
    testX = []
    testY = []

    for id in ids:
        latent_variable = np.load(f'./latent_variables/{id}.npy')
        labels = [id for _ in range(len(latent_variable))]


        trainX.extend(latent_variable[int(len(latent_variable)*0.2):])
        testX.extend(latent_variable[:int(len(latent_variable)*0.2)])

        trainY.extend(labels[int(len(labels)*0.2):])
        testY.extend(labels[:int(len(labels)*0.2)])
    
    trainX = np.asarray(trainX)
    testX = np.asarray(testX)
    trainY = np.asarray(trainY)
    testY = np.asarray(testY)
    # return trainX, coded_unknownX

    print(trainX.shape ,trainY.shape,testX.shape,testY.shape,coded_unknownX.shape,len(unknownY) )
    
    trainX = np.append(trainX, coded_unknownX[int(len(coded_unknownX)*.2):], axis=0)
    testX = np.append(testX, coded_unknownX[:int(len(coded_unknownX)*.2)], axis=0)
    trainY = np.append(trainY, unknownY[int(len(unknownY)*.2):], axis=0)
    testY = np.append(testY, unknownY[:int(len(unknownY)*.2)], axis=0)

    print(trainX.shape ,trainY.shape,testX.shape,testY.shape,coded_unknownX.shape,len(unknownY) )


    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    testX, testY = shuffle(testX, testY, random_state=0)
    print(trainX.shape,trainY.shape,testX.shape,testY.shape)

    data = {}
    data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'] = trainX, trainY, testX, testY

    score_train, score_test = classifier.train(data, group_name)
    print("done")
    return score_train, score_test
        
def predict(image_path, group_name):
    image = cv2.imread(image_path)
    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = extract_face(image)
    if type(image) == 0:
        return 0, "not Face detected"
    encoded_image = encoder(image[None])
    return classifier.predict(encoded_image, group_name)

def video_to_emb(video_path, id):

    print("aaa")
    camera = cv2.VideoCapture(video_path)
    faces = []
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        face = extract_face(frame)
        if type(face) == int:
                continue
        face =cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        faces.append(face)
    
    faces = np.asarray(faces)
    encode_faces = encoder(faces)
    encode_faces = np.asarray(encode_faces)
    np.random.shuffle(encode_faces)
    np.save(f'./latent_variables/{id}.npy', encode_faces)
    return True
