"""
######################################### vgg face model on vox1 dataset#########################################

This is a Keras model based on VGG16 architecture for vox1 dataset.
it can be used with pretrained weights file generated from Vgg face model https://www.robots.ox.ac.uk/~vgg/software/vgg_face/.

@author: Vikram Abrol (UMKC) and Wenqing Hu (Missouri S&T)

References:

[1] O. M. Parkhi, A. Vedaldi, A. Zisserman
Deep Face Recognition
British Machine Vision Conference, 2015
https://www.robots.ox.ac.uk/~vgg/software/vgg_face/
Mat file with pretrained model downloaded from the above link

[2] To convert the mat file of pretrained model into a h5 weights file
Reference https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/

[3] Face detection code help from

https://www.kaggle.com/saidakbarp/face-recognition-part-1

[4] Face cropping help from

https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5

"""
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
#from tensorflow.keras.preprocessing.image import load_img,img_to_array
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import scipy.io
#import cv2
import matplotlib.pyplot as plt

# Files used for face detection, download from https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models
modelFile ="res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
#net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

class vggFace:
    def __init__(self):
        self.num_classes = 207
        self.model = self.build_model()
        self.model.load_weights('vgg_face_weights.h5')

    def build_model(self):
        # Build the network of vgg for 207 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))
        return model

    def generate_weights(self, model):
        mat = scipy.io.loadmat('vgg_face_matconvnet/vgg_face_matconvnet/data/vgg_face.mat', matlab_compatible=False, struct_as_record=False)
        net = mat['net'][0][0]
        ref_model_layers = net.layers
        print(ref_model_layers.shape)
        ref_model_layers = ref_model_layers[0]
        for layer in ref_model_layers:
            print(layer[0][0].name)
        num_of_ref_model_layers = ref_model_layers.shape[0]
        base_model_layer_names = [layer.name for layer in model.layers]
        for i in range(num_of_ref_model_layers):
            ref_model_layer = ref_model_layers[i][0, 0].name[0]
            if ref_model_layer in base_model_layer_names:
                # we just need to set convolution and fully connected weights
                if ref_model_layer.find("conv") == 0 or ref_model_layer.find("fc") == 0:
                    print(i, ". ", ref_model_layer)
                    base_model_index = base_model_layer_names.index(ref_model_layer)

                    weights = ref_model_layers[i][0, 0].weights[0, 0]
                    bias = ref_model_layers[i][0, 0].weights[0, 1]
                    model.layers[base_model_index].set_weights([weights, bias[:, 0]])
                    model.save_weights('vgg_face_weights.h5')

    # function to extract box dimensions
    def face_dnn(img, coord=False):
        blob = cv2.dnn.blobFromImage(img, 1, (224, 224), [104, 117, 123], False, False)  #
        # params: source, scale=1, size=300,300, mean RGB values (r,g,b), rgb swapping=false, crop = false
        conf_threshold = 0.8  # confidence at least 60%
        frameWidth = img.shape[1]  # get image width
        frameHeight = img.shape[0]  # get image height
        max_confidence = 0
        net.setInput(blob)
        detections = net.forward()
        detection_index = 0
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:

                if max_confidence < confidence:  # only show maximum confidence face
                    max_confidence = confidence
                    detection_index = i
        i = detection_index
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if coord == True:
            return x1, y1, x2, y2
        return cv_rgb

    def load_data(self, Vgg_Embedded_Matfile):
        vgg_faces = scipy.io.loadmat(Vgg_Embedded_Matfile,
                                     matlab_compatible=False, struct_as_record=False, squeeze_me=True)
        keys = list(k for k, v in vgg_faces.items() if k not in ['__header__', '__version__', '__globals__'])
        num_classes = len(keys)
        data_original = {"x": [], "y": []}
        for i in range(num_classes):
            num_faces = len(vgg_faces[keys[i]])
            for j in range(num_faces):
                data_original["x"].append(list(vgg_faces[keys[i]][j]))
                data_original["y"].append(keys[i])

        from sklearn.preprocessing import LabelEncoder
        # creating instance of labelencoder
        le = LabelEncoder()
        le.fit(data_original["y"])
        le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        print(le_name_mapping)
        data_original["y"] = le.transform(data_original["y"])

        num_total_faces = len(data_original["y"])
        # split into the training and testing data sets
        data_original_train = {"x": [], "y": []}
        data_original_test = {"x": [], "y": []}
        # extract the training and testing data sets
        indexes = np.random.permutation(num_total_faces)
        train_size = int(0.9 * num_total_faces)

        train_indexes = [indexes[_] for _ in range(train_size)]
        test_indexes = [indexes[_] for _ in range(train_size, num_total_faces)]
        data_original_train["x"] = [data_original["x"][_] for _ in train_indexes]
        data_original_train["y"] = [data_original["y"][_] for _ in train_indexes]
        data_original_test["x"] = [data_original["x"][_] for _ in test_indexes]
        data_original_test["y"] = [data_original["y"][_] for _ in test_indexes]

        x_train = data_original_train["x"]
        y_train = data_original_train["y"]
        x_test = data_original_test["x"]
        y_test = data_original_test["y"]

        x_train = np.array(x_train, dtype=np.float)
        y_train = np.array(y_train)
        x_test = np.array(x_test, dtype=np.float)
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test, le_name_mapping

    def classifier(self, x_train, y_train, x_test, y_test, epochs=10, train=0):
        # Softmax regressor to classify images based on encoding
        classifier_model = Sequential()
        classifier_model.add(Dense(units=4096, input_dim=x_train.shape[1], kernel_initializer='glorot_uniform'))
        classifier_model.add(BatchNormalization())
        classifier_model.add(Activation('tanh'))
        classifier_model.add(Dropout(0.3))
        classifier_model.add(Dense(units=1024, kernel_initializer='glorot_uniform'))
        classifier_model.add(BatchNormalization())
        classifier_model.add(Activation('tanh'))
        classifier_model.add(Dropout(0.2))
        classifier_model.add(Dense(units=self.num_classes, kernel_initializer='he_uniform'))
        classifier_model.add(Activation('softmax'))
        if train:
            classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                                     optimizer='nadam',
                                     metrics=['accuracy'])
            classifier_model.summary()
            history = classifier_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
            classifier_model.save_weights('vgg_classifier.h5')
        else:
            classifier_model.load_weights('vgg_classifier.h5')
        return classifier_model

    def predict_label(self, test_img, classifier_model, le_name_mapping):
        model = self.build_model()
        model.load_weights('vgg_face_weights.h5')
        # Remove last Softmax layer and get model upto last flatten layer #with outputs 2622 units
        vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        img = cv2.imread(test_img)
        x1, y1, x2, y2 = self.face_dnn(img, True)
        plt.imshow(img)
        plt.show()
        # Crop image
        left = x1
        top = y1
        right = x2
        bottom = y2
        width = right - left
        height = bottom - top
        img_crop = img[top:top + height, left:left + width]
        # print coordinates of the detected face
        print(x1, y1, x2, y2)
        plt.imshow(img_crop)
        plt.show()
        cv2.imwrite(os.getcwd() + '/crop_img.jpg', img_crop)

        # Find vgg face embeddings of this image
        crop_img = load_img(os.getcwd() + '/crop_img.jpg', target_size=(224, 224))
        crop_img = img_to_array(crop_img)
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = preprocess_input(crop_img)
        img_encode = vgg_face(crop_img)

        # Make Predictions
        embed = K.eval(img_encode)
        person = classifier_model.predict(embed)
        person_index = np.argmax(person)
        name = le_name_mapping[np.argmax(person)]
        return person_index, name

    def predict_label_embedded(self, embed, classifier_model):
        person = classifier_model.predict(embed)
        person_index = np.argmax(person, 1)
        return person_index

    def predict_label_name_embedded(self, embed, classifier_model, le_name_mapping):
        person = classifier_model.predict(embed)
        person_index = np.argmax(person, 1)
        name = [le_name_mapping[person_index[_]] for _ in range(len(person_index))]
        return person_index, name


if __name__ == '__main__':

    vgg_model = vggFace()

    x_train, y_train, x_test, y_test, le_name_mapping = vgg_model.load_data('data\\vgg_f_onefile.mat')
    classifier_model = vgg_model.classifier(x_train, y_train, x_test, y_test)
    # test prediction on a test image
    index, name = vgg_model.predict_label_name_embedded(x_test, classifier_model, le_name_mapping)
    index2 = vgg_model.predict_label_embedded(x_test, classifier_model)
    for i in range(len(index)):
        print(index2[i], "(", index[i], name[i], ") , (", y_test[i], le_name_mapping[y_test[i]],")")

    test_size = len(y_test)
    correct_number = sum(index==y_test)
    loss = 1 - correct_number/test_size
    print("accuracy is: ", (1-loss)*100, "%")
    print("the validation 0/1 loss is: ",loss)