import os
import numpy as np 
import tensorflow as tf 
import h5py
import math

def load_dataset():
	train_dataset = h5py.File('datasets/train_signs.h5', 'r')
	train_set_x_orig = np.array(train_dataset['train_set_x'][:])
	train_set_y_orig = np.array(train_dataset['train_set_y'][:])

	test_dataset = h5py.File('datasets/test_signs.h5', 'r')
	test_set_x_orig = np.array(test_dataset['test_set_x'][:])
	test_set_y_orig = np.array(test_dataset['test_set_y'][:])

	classes = np.array(test_dataset['list_classes'][:]) # list of classes

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
 
def random_mini_batch(X, Y, mini_batch_size = 64, seed = 0):
	m = X.shape[0] # number of training examples
	mini_batches = []
	
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	batch_number = math.floor(m / mini_batch_size)
	for i in range(batch_number):
		mini_X = shuffled_X[i * mini_batch_size: (i+1) * mini_batch_size, :, :, :]
		mini_Y = shuffled_Y[i * mini_batch_size: (i+1) * mini_batch_size, :]
		mini_batches.append((mini_X, mini_Y))
	if m % batch_number * mini_batch_size:
		mini_batches.append((
			shuffled_X[batch_number * mini_batch_size:m, :, :, :], 
			shuffled_Y[batch_number * mini_batch_size:m, :]))
	return mini_batches

def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T
	return Y

def forward_propagation_for_predict(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)

	return Z3

def predict(X, parameters):
	W1 = tf.convert_to_tensor(parameters['W1'])
	b1 = tf.convert_to_tensor(parameters['b1'])
	W2 = tf.convert_to_tensor(parameters['W2'])
	b2 = tf.convert_to_tensor(parameters['b2'])
	W3 = tf.convert_to_tensor(parameters['W3'])
	b3 = tf.convert_to_tensor(parameters['b3'])

	params = {
		'W1':W1,
		'b1':b1,
		'W2':W2,
		'b2':b2,
		'W3':W3,
		'b3':b3
		}

	x = tf.placeholder('float', [12288, 1])
	z3 = forward_propagation_for_predict(x, params)
	p = tf.argmax(z3)

	sess = tf.Session()
	prediction = sess.run(p, feed_dict = {x: X})
	return prediction

if __name__ == '__main__':
	trainX, trainY, testX,testY, classes = load_dataset()
	print(trainX[:1])
	print(trainY[:1])
	# print(testX[:1])
	# print(testY[:1])
	print(trainX.shape)
	print(trainY.shape)
	print(testX.shape)
	print(testY.shape)
	print(classes)