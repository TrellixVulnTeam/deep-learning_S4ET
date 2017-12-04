import keras.backend as K 
import math
import numpy as np 
import h5py # 一个为HDF5二进制文件格式提供访问api的python包
import matplotlib.pyplot as plt

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

def load_dataset():
	train_dataset = h5py.File('datasets/train_happy.h5', 'r')
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])

	test_dataset = h5py.File('datasets/test_happy.h5', 'r')
	test_set_x_orig = np.array(test_dataset['test_set_x'][:])
	test_set_y_orig = np.array(test_dataset['test_set_y'][:])

	classes = np.array(test_dataset['list_classes'][:])

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def peek_dataset():
	def printname(name):
		print(name)
	train_dataset = h5py.File('datasets/train_happy.h5', 'r')
	train_dataset.visit(printname)
	list_classes = train_dataset['list_classes']
	print(list_classes.shape)
	print(list_classes[:10])
	train_set_y_orig = np.array(train_dataset["train_set_y"][:])
	train_set_x_orig = np.array(train_dataset["train_set_x"][:])
	print(train_set_y_orig.shape)
	print(train_set_x_orig.shape)


if __name__ == '__main__':
	peek_dataset()