
# coding: utf-8

# In[ ]:

import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from helper import batches

# reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as pfile:
    pickle_data = pickle.load(pfile)
    train_features = pickle_data['train_dataset']
    train_labels = pickle_data['train_labels']
    valid_features = pickle_data['valid_dataset']
    valid_labels = pickle_data['valid_labels']
    test_features = pickle_data['test_dataset']
    test_labels = pickle_data['test_labels']
    del pickle_data

print('Data module loaded')


# In[ ]:

n_classes = (train_labels[0].shape)[0]
print(n_classes)


# In[34]:

learning_rate = 0.1
batch_size = 128
training_epochs = 5
display_step = 50

n_records, n_features = train_features.shape
n_classes = (train_labels[0].shape)[0]
n_hidden_layer = 256

# weights and bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_features, n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}

biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_classes]))
}

# define inputs
x = tf.placeholder("float", [None, n_features])
y = tf.placeholder("float", [None, n_classes])

layer_1 = tf.add(tf.matmul(x, weights['hidden_layer']), biases['hidden_layer']) # (n_records, n_hidden_layer)
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights['output_layer']), biases['output_layer'])


# In[35]:

# Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_2, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.nn.softmax(layer_2)
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print("Accuracy function created.")


# In[36]:

mbatches = batches(batch_size, train_features, train_labels)
print(len(mbatches))
batch_x, batch_y = batch
print(batch_x[0])
print(batch_y[0])


# In[45]:

import sys
# Session

loss_batch = []
train_acc_batch = []
valid_acc_batch = []
plot_batches = []

# Initializing the variables
init = tf.global_variables_initializer()

save_file = './train_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    batch_count = int(math.ceil(n_records/batch_size))
    for epoch in range(training_epochs):
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch+1, training_epochs), unit='batches')
        
        mbatches = batches(batch_size, train_features, train_labels)
        assert len(mbatches) == batch_count
        for batch_i in batches_pbar:
            batch = mbatches[batch_i]
            batch_x, batch_y = batch
            _, l = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            
            if not batch_i % display_step:
                training_accuracy = sess.run(accuracy, feed_dict={x: train_features, y: train_labels})
                validation_accuracy = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels})
                
                previous_batch = plot_batches[-1] if plot_batches else 0
                plot_batches.append(display_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)
                sys.stdout.write("\rProgress: {:2.1f}".format(100 * batch_i/float(batch_count))                      + "% ... Training loss: " + str(l)[:5]                      + " ... Validation accuracy: " + str(validation_accuracy)[:5])
                sys.stdout.flush()
        # Check accuracy against Validation data
        validation_accuracy = sess.run(accuracy, feed_dict={x: valid_features, y: valid_labels})
    saver.save(sess, save_file)
    print("Trained model saved!")

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(plot_batches, loss_batch, 'g')
loss_plot.set_xlim([plot_batches[0], plot_batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(plot_batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(plot_batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([plot_batches[0], plot_batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))


# In[44]:

print('Validation accuracy at {}'.format(validation_accuracy))


# In[48]:

with tf.Session() as sess:
    saver.restore(sess, save_file)
    test_accuracy = sess.run(
        accuracy,
        feed_dict={x: test_features, y: test_labels})
    
print('Test Accuracy: {}'.format(test_accuracy))


# In[ ]:



