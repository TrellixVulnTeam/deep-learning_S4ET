import numpy as numpy
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_util import *
from keras.models import Sequential
from keras.utils import to_categorical

import keras.backend as K 
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255

# Reshape
Y_train = Y_train_orig.T 
Y_test = Y_test_orig.T

print("number of traning examples = " + str(X_train.shape(0)))
print("number of test examples = " + str(X_test.shape(0)))
print("X_train shape = " + str(X_train.shape))
print("Y_train shape = " + str(Y_train.shape))
print("X_test shape = " + str(X_test.shape))
print("Y_test shape=" + str(Y_test.shape))

def HappySequenceModel(input_shape):
	"""
	Implementation of the HappyModle using Sequence model.
	Arguments:
	input_shape -- shape of the images of the dataset
	Returns:
	model -- a Model() instance in Keras
	"""

	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(2, activation='softmax'))

	return model

def HappyModel(input_shape):
	X_input = Input(input_shape)
	X = Conv2D(32, (3, 3), activation='relu') (X_input)
	X = Conv2D(32, (3, 3), activation='relu')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Dropout(0.25)(X)

	X = Conv2D(64, (3, 3), activation='relu') (X)
	X = Conv2D(64, (3, 3), activation='relu')(X)
	X = MaxPooling2D(pool_size=(2,2))(X)
	X = Dropout(0.25)(X)

	X = Flatten()(X)
	X = Dense(256, activation='relu')(X)
	X = Dropout()(X)
	X = Dense(2, activation='softmax')(X)

	model = Model(input=X_input, outputs=X, name='HappyModel')

	return model

from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
happyModel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train
happyModel.fit(X_train, to_categorical(Y_train), batch_size=32, epochs=10)
# evaluate
preds = happyModel.evaluate(X_test, to_categorical(Y_test), batch_size=32)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# Test with my own image
img_path = 'images/IMG_4087.JPG'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))

happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))