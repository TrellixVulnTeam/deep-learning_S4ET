import time
from collections import namedtuple
import numpy as np 
import tensorflow as tf 

with open('anna.txt') as f:
	text = f.read() # reads some quantity of data and return it as a string or bytes object
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

print(encoded[:100])

def get_batches(arr, n_seqs, n_steps):
	num_characters = n_seqs * n_seqs
	n_batches = len(arr) // num_characters
	arr = arr[:n_batches * num_characters]

	arr = arr.reshape((n_seqs, -1))
	for n in range(0, arr.shape[1], n_steps):
		x = arr[:, n:n+n_steps]
		y = np.zeros_like(x)
		y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
		yield x, y

batches = get_batches(encoded, 10, 50)
x, y = next(batches)
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])

# Building the model

def build_inputs(batch_size, num_steps):
	inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
	targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	return inputs, targets, keep_prob

# Build single LSTM cell and stack them together
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

	def build_cell(num_units, keep_prob):
		lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
		drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
		return drop

	cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
	initial_state = cell.zero_state(batch_size, tf.float32)
	return cell, initial_state

def build_output(lstm_output, in_size, out_size):
	seq_out = tf.concat(lstm_output, axis=1)
	x = tf.reshape(seq_out, [-1, in_size])
	
	with tf.variable_scope('softmax'):
		softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
		softmax_b = tf.Variable(tf.zeros(out_size))

	logits = tf.matmul(x, softmax_w) + softmax_b
	out = tf.nn.softmax(logits)

	return out, logits


def build_loss(logits, targets, num_classes):
	# on-hot-encode targets
	y_one_hot = tf.one_hot(targets, num_classes)
	y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

	loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
	loss = tf.reduce_mean(loss)

	return loss


def build_optimizer(loss, learning_rate, grad_clip):
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
	train_op = tf.train.AdamOptimizer(learning_rate)
	optimizer = train_op.apply_gradients(zip(grads, tvars))

	return optimizer

class CharRNN:
	def __init__(self, num_classes, batch_size=64, num_steps=50, 
		lstm_size=128, num_layers=2, learning_rate=0.001, 
		grad_clip=5, sampling=False):

		# When we're using this network for sampling later, we'll be passing in
		# one character at a time, so providing an option for that
		if sampling == True:
			batch_size, num_steps = 1, 1
		else:
			batch_size, num_steps = batch_size, num_steps

		tf.reset_default_graph()

		# Build the input placeholder tensors
		self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

		# Build the LSTM cell
		cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

		### Run the data through the RNN layers
		# First, one-hot encode the input tokens
		x_one_hot = tf.one_hot(self.inputs, num_classes)

		# Run each sequence step through the RNN and collect the outputs
		outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
		self.final_state = state

		# Get softmax predictions and logits
		self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

		# Loss and optimizer (with gradient clipping)
		self.loss = build_loss(self.logits, self.targets, num_classes)
		self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability

epochs = 20
# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
	lstm_size=lstm_size, num_layers=num_layers, 
	learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Use the line below to load a checkpoint and resume training
	#saver.restore(sess, 'checkpoints/______.ckpt')
	counter = 0
	for e in range(epochs):
		# Train network
		new_state = sess.run(model.initial_state)
		loss = 0
		for x, y in get_batches(encoded, batch_size, num_steps):
			counter += 1
			start = time.time()
			feed = {model.inputs: x,model.targets: y,model.keep_prob: keep_prob,model.initial_state: new_state}
			batch_loss, new_state, _ = sess.run([model.loss, 
				model.final_state, 
				model.optimizer], 
				feed_dict=feed)

			end = time.time()
			print('Epoch: {}/{}... '.format(e+1, epochs),
				'Training Step: {}... '.format(counter),
				'Training loss: {:.4f}... '.format(batch_loss),
				'{:.4f} sec/batch'.format((end-start)))

			if (counter % save_every_n == 0):
				saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

	saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))