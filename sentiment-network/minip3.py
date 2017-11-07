from collections import Counter
import numpy as np
import pdb

class NeuralNetwork:
    def __init__(self, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)
        self.hidden_nodes = hidden_nodes
        self.output_nodes = 1
        self.learning_rate = learning_rate
        pdb.set_trace()
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.activation_function = sigmoid
        
    def MSE(y, Y):
        return np.mean((y-Y)**2)
    
    def train(self, features, targets):
        m_features = features.shape[1]
        m_records = 128
        pdb.set_trace()
        self.weight_input_to_hidden = np.zeros((m_features, self.hidden_nodes))
        self.weight_hidden_to_output = np.zeros(self.hidden_nodes, self.output_nodes)
        
        epochs = 1000
        for i in np.arrange(epochs):
            batch = np.random.choice(features.index, size=m_records)
            X, y = features.ix[batch].values, targets.ix[batch]
            # train data, test data, validation data
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#             X_val, y_val = train_test_split(X_train, y_train, test_size=0.25)
            
            # forwardprop
            hidden_layer_input = np.dot(X, weight_input_to_hidden) # hidden_layer_input(m_records, hidden_nodes)
            hidden_layer_output = self.activation_function(hidden_layer_input) # hidden_layer_output(m_records, hidden_nodes)

            output_layer_input = np.dot(hidden_layer_output, weight_hidden_to_output) # output_layer_input(m_records, self.output_nodes)
            output_layer_output = self.activation_function(output_layer_input)

            # backprop
            output_error_term = (np.sum(y - output_layer_output) / m_records) * output_layer_output * (1 - output_layer_output)
            del_w_hidden_to_output = output_error_term * hidden_layer_output

            hidden_error_term = output_error_term * weight_hidden_to_output * hidden_layer_output * (1 - hidden_layer_output)
            del_w_input_to_hidden = hidden_error_term * X
            
            weight_input_to_hidden += self.learning_rate * del_w_input_to_hidden
            weight_hidden_to_output += self.learning_rate * del_w_hidden_to_output
        
        

    def predict(self, feautures, targets):
        hidden_layer_input = np.dot(features, weight_input_to_hidden) # hidden_layer_input(m_records, hidden_nodes)
        hidden_layer_output = self.activation_function(hidden_layer_input) # hidden_layer_output(m_records, hidden_nodes)

        output_layer_input = np.dot(hidden_layer_output, weight_hidden_to_output) # output_layer_input(m_records, self.output_nodes)
        output_layer_output = self.activation_function(output_layer_input)
        
        error = self.MSE(output_layer_output, targets)
        return error


def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()


# a = np.array(labels)
# b = np.array(reviews)
# negativeIndicies = np.argwhere(a == 'NEGATIVE').flatten()
# negativeReviews = b[negativeIndicies]
# negativeWords = []
# for review in negativeReviews:
#     negativeWords.extend(review.split())

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for index, review in enumerate(reviews):
    words = review.split(' ')
    if labels[index] == 'POSITIVE':
        for word in words:
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in words:
            negative_counts[word] += 1
            total_counts[word] += 1

vocab = set(list(total_counts))
vocab_size = len(vocab)
print(vocab_size)
features = np.zeros((len(reviews), vocab_size))
print(features.shape)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
word2index['']

def update_input_layer(review, feature):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    # clear out previous state by resetting the layer to be all 0s
    feature *= 0

    # TODO: count how many times each word is used in the given review and store the results in layer_0 
    global word2index
    for word in review.split(' '):
        feature[word2index[word]] += 1

def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here
    if label == 'NEGATIVE':
        return 0
    else:
        return 1


classifier = NeuralNetwork()
labels = np.zeros((len(reviews), 1))
features = np.zeros((len(reviews), vocab_size))
for i,review in enumerate(reviews):
    update_input_layer(review, features[i])
labels = [get_target_for_label(label) for label in labels]
print(features.shape)
print(labels.shape)
pdb.set_trace()
classifier.train(features, labels)

# Create Counter object to store positive/negative ratios
# pos_neg_ratios = Counter()

# # TODO: Calculate the ratios of positive and negative uses of the most common words
# #       Consider words to be "common" if they've been used at least 100 times
# for word in list(total_counts):
#     if total_counts[word] >= 100:
#         pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word]+1)


# print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
# print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
# print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))


