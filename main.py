from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np


## these are the flowers attributes. the goal of the model is to predict what species the flower is based on this information.
## headers of the columns 
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
## these are the species of flowers
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

## tells the model how to get the training dataset
train_path = tf.keras.utils.get_file(
    ## saves the file as 'iris_training.csv' onto the users the computer
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

## loads the data into dataframes. header = 0 just means the names will be put at row 0. 
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

## when this code is run, the model will automatically convert the species into numerical values. 'setosa' = 0, 'versicolor' = 1, 'virginica' = 2

## '.astype(np.float32)' isn't needed here since the species are already stored as integers
## this leaves only the numerical placeholder for the species and the number it is in the dataset. the first set of data and species 'setosa' would be --> 0, 0
y_train = train.pop('Species')
y_test = test.pop('Species')

## 'labels' = y_train, y_test. 'training=True' just means that the training dataset is being used. 256 samples per batch. 
def input_fn(features, labels, training=True, batch_size=256):
    ## converts the input data into a TensorFlow 'dataset'. '((dict(data_df), label_df))' passes the pandas dataframe along with the labels being used into the new 'tf.data.Dataset' TensorFlow dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        ## shuffles and repeats the data
        dataset = dataset.shuffle(1000).repeat()
    ## groups the data into the batch size defined above
    return dataset.batch(batch_size)

## initializes an empty list to hold the feature definitions for each feature in the dataset
my_feature_columns = []

## loops through each of the features (SepalLength, SepalWidth, etc) and assigns each a key (0, 1, 2 --> used to identify the feature numerically)
for key in train.keys():
    ## it then stores these values in a tensorflow 'feature_column'. it stores the feature and it's corresponding key
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


## defines a Sequential model in Keras
## a sequential model is a linear stack of layers where each layer has 1 input and output tensor
model = tf.keras.Sequential([
    ## input layer takes the features as the input
    tf.keras.layers.Input(shape=(len(train.keys()),)),
    ## creates a hidden layer of 30 neurons, using the relu activation function 
    tf.keras.layers.Dense(30, activation='relu'), 
    ## creates a hidden layer of 10 neurons using the relu activation function
    tf.keras.layers.Dense(10, activation='relu'), 
    ## output layer with 3 options (the species of flower). 'softmax' is 
    tf.keras.layers.Dense(3, activation='softmax')
])

## compiles the model with an optimizer, loss, and metric
model.compile(
    optimizer='adam',
    ## used for multi-class classification
    loss='sparse_categorical_crossentropy',
    ## tells the model to track accuracy
    metrics=['accuracy']
)

## convert pandas DataFrame to NumPy arrays
## features of the training dataset
x_train = train.values
## features of the testing dataset
x_test = test.values 
## labels of the training dataset
y_train = y_train.values
## labels of the testing dataset
y_test = y_test.values

# Train the model using the NumPy arrays
model.fit(
    ## input features
    x=x_train,
    ## target labels
    y=y_train,
    ## number of training epochs
    epochs=5000,
    ## batch size
    batch_size=256
)

## evaluate the model on the testing dataset
loss, accuracy = model.evaluate(
    ## testing features
    x=x_test,   
    ## true labels for the testing set 
    y=y_test,
    ## batch size
    batch_size=256
)

## print the results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
