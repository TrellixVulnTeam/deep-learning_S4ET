
# coding: utf-8

# In[11]:

import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print("All modules loaded.")


# In[12]:

def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)


# In[13]:

datasets_folder = '/Users/iratao/Documents/project/datasets/'
# Get the features and labels from the zip files
train_features, train_labels = uncompress_features_labels(datasets_folder + 'notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels(datasets_folder + 'notMNIST_test.zip')

# check the shape of the data
print('train_features.shape = {0}'.format(train_features.shape))
print('test_labels.shape = {0}'.format(test_labels.shape))
print('train_features[0] = {0}'.format(train_features[0]))


# In[14]:

# Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)


# Min-Max Scaling:
# $
# X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
# $

# In[15]:

# normalize data
is_features_normal = False
is_labels_encod = False
amin = 0.1
bmax = 0.9

def normalize_data(image_data, amin, bmax):
    image_data = np.array(image_data)
    return amin + (image_data - np.min(image_data))*(bmax - amin) / (np.max(image_data) - np.min(image_data))

# Test Cases
np.testing.assert_array_almost_equal(
    normalize_data(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]), 0.1, 0.9),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],
    decimal=3)
print('Tests Passed!')


# In[17]:

if not is_features_normal:
    train_features = normalize_data(train_features, amin, bmax)
    test_features = normalize_data(test_features, amin, bmax)
    is_features_normal = True

if not is_labels_encod:
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    print(encoder.classes_)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)
    
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_label_encod = True
print('Labels One-Hot Encoded')


# In[18]:

# split train and validate data set
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features, train_labels, test_size=0.05, random_state=832289)

print('Training features and labels randomized and split.')


# In[20]:

# save the data for easy access
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump({
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                }, pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to ', pickle_file, ':', e )
        raise
print('Data cached in pickle file')


# In[ ]:



