
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

training_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test

print(training_data.images.shape)
# print(valid_data.output_shapes)
# print(test_data.output_shapes)

import tensorflow as tf


# In[24]:

import numpy as np
image = np.array([
        [[[1],[1],[1],[1]], [[2],[2],[2],[2]], [[1],[1],[1],[1]], [[2],[2],[2],[2]]]
        ])
print(image.shape)


# In[8]:

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv_net(x, weights, biases, dropout):
#     pass
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    
    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    
    # Fully connected layer - 7*7*64 to 1024
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def run():
    # Parameters
    learning_rate = 0.0001
    epochs = 10
    batch_size = 128

    test_valid_size = 256

    # Network Parameters
    n_classes = 10
    dropout = 0.75

    # Store layers weight & bias
    # Original image: 32*32*1
    weights = {
        'wc1': tf.Variable(tf.random_normal([5,5,1,32])), # (28,28,1) --'same' padding-->(28,28,32)--max pooling--> (14,14,32)
        'wc2': tf.Variable(tf.random_normal([5,5,32,64])), # (14,14,32) --'same' padding-->(14,14,64)--max pooling--> (7,7,64)
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # (7,7,64) --fully connected layer-->(7*7*64, 1024)
        'out': tf.Variable(tf.random_normal([1024, n_classes])) # fully connected layer with output of width 'n_classes'
    }
    bias = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    # Model
    logits = conv_net(x, weights, bias, keep_prob)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Init
    init = tf.global_variables_initializer()
    
    # Launch the Graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            for batch in range(mnist.train.num_examples//batch_size):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={
                    x: batch_x, 
                    y: batch_y,
                    keep_prob: dropout
                })
                loss = sess.run(cost, {
                    x: batch_x,
                    y: batch_y,
                    keep_prob: 1
                })
                valid_accu = sess.run(accuracy, {
                    x: mnist.validation.images[:test_valid_size],
                    y: mnist.validation.labels[:test_valid_size],
                    keep_prob: 1
                })
                print('Epoch {:>2}, Batch {:>3} -'
                      'Loss {:>10.4f} Validation Accuracy {:.6f}'.format(
                      epoch+1, batch+1, loss, valid_accu))
        test_accu = sess.run(accuracy, {
            x: mnist.test.images[:test_valid_size],
            y: mnist.test.labels[:test_valid_size],
            keep_prob: dropout
        })
        print('')
    
def test():
    # Test
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        image = np.array([
            [[[1.0],[1],[1],[1]], [[2],[2],[2],[2]], [[1],[1],[1],[1]], [[2],[2],[2],[2]]]
        ], dtype='f')
        x = tf.placeholder(tf.float32, shape=[None, 4, 4, 1])
        w = tf.Variable(tf.random_normal([2,2,1,1]))
        b = tf.Variable(tf.random_normal([1]))
        conv = conv2d(image, w, b)
        print(conv)
        mconv = maxpool2d(conv)
        print(mconv)

run()


# In[ ]:



