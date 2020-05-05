from numpy import expand_dims
import tensorflow as tf
from keras.models import load_model

model = load_model('./encoder/model/facenet_keras.h5')
class Encoder:
    def __init__(self):
        self.model = load_model('./encoder/model/facenet_keras.h5')
        self.graph = tf.get_default_graph()
    def __call__(self, face_pixels):

        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        with self.graph.as_default():   
            yhat = model.predict(face_pixels)
        return yhat
 
