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

# print('This {food} is {adjective}.'.format(food='spam', adjective='absolutely horrible'))
datasets_folder = '/Users/iratao/Documents/project/datasets/'
# Get the features and labels from the zip files
train_features, train_labels = uncompress_features_labels(datasets_folder + 'notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels(datasets_folder + 'notMNIST_test.zip')

# Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

is_features_normal = False
is_labels_encod = False

# normalize data
print('train_features.shape = {0}'.format(train_features.shape))
print('test_labels.shape = {0}'.format(test_labels.shape))
print('train_features[0] = {0}'.format(train_features[0]))

# train validate split

# Save the data for easy access - use pickle