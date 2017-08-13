# Copyright 2017 BlyNotes. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###############################################
# Author: Original code provided by Professor for CS 6375
#
# Modified by: Stephen Blystone
#
# Purpose: Modify hyperparameters using Adam Optimizer
#               and modified MNIST data to achieve highest
#               test accuracy that you can.
###############################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from our_mnist import *

import tensorflow as tf
import sys
import csv
import numpy as np
import timeit
mnist = read_data_sets('MNIST_data', one_hot=True)


# Variables initialized below for automation
# X = tf.placeholder(tf.float32, shape=[None, 784])
# Y = tf.placeholder(tf.float32, shape=[None, 10])
# keep_prob = tf.placeholder(tf.float32)

# *******************************************************************
# DO NOT MODIFY THE CODE ABOVE THIS LINE
# MAKE CHANGES TO THE CODE BELOW:

# Flag to disable printing every 100 batches
DISABLEBATCHPRINT = False


def weight_variable(shape):
    """a method for initializing weights. Initialize to small random values."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """a method for initializing bias. Initialize to 0.1."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def buildSingleLayer(input, numInputs, layer_dict, layerNum, keep_prob_label):
    """Build a single layer of the network (either Hidden or Output).
    This is a fully connected Layer of layer_dict['layerSize'] nodes

    Input:
       -input:   input values to this layer (output from the previous layer, or X for first layer)
       -numInputs:  number of nodes in the input layer (i.e. size of previous layer)
       -layer_dict: python dictionary holding layer-specific parameters
       -layerNum:   index of which layer is being built
       -keep_prob_label:    python list holding tf.placeholder(tf.float32) for the keep_prob for that layer

    Output:
       -v_fc:   output values for the layer
       -W_fc:   weights calculated for this layer; used for regularization calculations later
    """

    # Print layer-specific information
    print("Building Layer {}. numInputs = {} and layerSize = {}".format(
        layerNum + 1, numInputs, layer_dict['layerSize']))

    # Create Weight and Bias variables using layer-specific values in layer_dict
    W_fc = weight_variable([numInputs, layer_dict['layerSize']])
    b_fc = bias_variable([layer_dict['layerSize']])

    # Every node type calculates the matrix multiplication
    matmul = tf.matmul(input, W_fc) + b_fc

    # Apply specific node type
    if (layer_dict['nodeType'] == "ReLU"):
        v_fc = tf.nn.relu(matmul)
    elif (layer_dict['nodeType'] == "Sigmoid"):
        v_fc = tf.nn.sigmoid(matmul)
    elif (layer_dict['nodeType'] == "Linear"):
        # Linear, so no need to apply anything
        v_fc = matmul
    else:
        print("ERROR: Unknown nodeType in layer_dict: {}".format(layer_dict['nodeType']))
        sys.exit(1)

    # Add drop-out layer if specified
    if (layer_dict['dropoutFlag']):
        print("Dropout Layer")
        v_fc = tf.nn.dropout(v_fc, keep_prob_label[layerNum])

    return (v_fc, W_fc)


def buildLayers(layers, keep_prob_label, X):
    """Builds all of the layers specified by layerSizes list.
    It prints message when it builds input layer, Hidden layers or output layer

    Inputs:
       -layers: list of all of the layer_dicts
       -keep_prob_label:    python list holding tf.placeholder(tf.float32) for the keep_prob for each layer
       -X:  initial input values

    Outputs:
        -tmpVal:    This is the predicted_Y value
        -weights:   This is the list of weights for each layer; used later in regularization calculations
    """
    print("Building Layers")

    # Set tmpVal equal to output of current layer built
    # Use tmpVal to build next layer
    tmpVal = X

    # Set tmpSize equal to size of X initially
    tmpSize = X.get_shape().dims[1].value

    # initialize weights list
    weights = []

    # Iterate over each layer_dict
    # enumerate returns an index along with the data
    # index (idx) used to print whether we are building a Hidden Layer or Output Layer
    for idx, layer_dict in enumerate(layers):
        if (idx < len(layers) - 1):
            print("Building Hidden Layer {}".format(idx + 1))
        else:
            print("Building Output Layer")

        # Build layer and get output values
        tmpVal, tmpWeight = buildSingleLayer(
            tmpVal, tmpSize, layer_dict, idx, keep_prob_label)
        # Store the size of the current layer to be used as the input size for the next layer
        tmpSize = layer_dict['layerSize']  # Use this for next iteration
        # Append the weights from the current layer to the weights list
        weights.append(tmpWeight)

    # tmpVal is the predicted_Y value
    # weights is the list of weights used in each layer
    return (tmpVal, weights)


def runAlgorithm(param, outputText=""):
    """Function that runs the algorithm.

    Inputs:
       -param:  a dict containing all the variables I can adjust
       -outputText: used for writing to the output file
    """

    # Reset the default graph to free memory
    # If you were to run a for loop over runAlgorithm trying different configurations,
    # Tensorflow places each node into a graph (the default graph if no other is specified).
    # Eventually this will cause an Out Of Memory error to occur, because the graph size
    # continues to grow.
    # To fix this issue, I reset the default graph everytime I call runAlgorithm.
    # This causes Tensorflow to remove the old graph and create a brand new graph each time.
    tf.reset_default_graph()

    # As a result of reseting the default graph, I have to recreate the X and Y placeholder variables.
    # Placing them after reset_default_graph causes Tensorflow to add these first to the new graph.
    X = tf.placeholder(tf.float32, shape=[None, 784])
    Y = tf.placeholder(tf.float32, shape=[None, 10])

    # Start interactive session
    sess = tf.InteractiveSession()

    # Start timer.  Used to time how long it takes runAlgorithm to complete
    start_time = timeit.default_timer()

    # Initialize keep_prob list
    # keep_prob holds the actual probability values specified in the layers list.
    # keep_prob_label holds the tf.placeholder values for each layer
    keep_prob = []
    keep_prob_label = []
    # enumerate over the layers to get the keep_prob values
    for idx, layer in enumerate(param['layers']):
        # print("{}: {}".format(idx, layer))
        # create placeholder value for keep_prob for each layer
        tmpProbHolder = tf.placeholder(tf.float32)
        # Append placeholder value to keep_prob_label list
        keep_prob_label.append(tmpProbHolder)
        # Append actual probability for each layer to end of keep_prob list
        keep_prob.append(layer['keep_prob'])

    # Create feed_dict for printing the output and testing
    feed_dict_Print_Train = {keep_prob_label[i]: 1 for i in range(0, len(param['layers']))}
    feed_dict_Print_Train[X] = mnist.train.images
    feed_dict_Print_Train[Y] = mnist.train.labels

    # Create feed_dict for printing the output and testing
    feed_dict_Test = {keep_prob_label[i]: 1 for i in range(0, len(param['layers']))}
    feed_dict_Test[X] = mnist.test.images
    feed_dict_Test[Y] = mnist.test.labels

    # Build layers and get predicted_Y and layerWeights back
    predicted_Y, layerWeights = buildLayers(param['layers'], keep_prob_label, X)

    # Calculate cross_entropy
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predicted_Y))

    # Calculate regularization value
    # Assert that the number of layers equals to the number of layers for which we have weights
    # This is required for zip to work properly below
    assert len(param['layers']) == len(layerWeights)
    # Use Python's zip function, along with list comprehension to calculate the regularization values
    # Zip takes 2 data structures of equal lengths and returns the i'th value of the first structure along
    # with the i'th value of the second structure.
    # For example:
    #       d1 = [1, 2, 3]
    #       d2 = [10, 100, 1000]
    #       for elementFrom_d1, elementFrom_d2 in zip(d1, d2):
    #           print(elementFrom_d1 * elementFrom_d2)
    #
    # The above example outputs:
    #       10
    #       200
    #       3000
    #
    # The list comprehension is basically a shorthand way of doing the following:
    #       tmp = []
    #       for a, b in zip(param['layers'], layerWeights):
    #           tmp.append(a['regLambda'] * tf.nn.l2_loss(b))
    #
    # Using the above example, the corresponding list comprehension would be:
    #       d1 = [1, 2, 3]
    #       d2 = [10, 100, 1000]
    #       tmp = []
    #       for elementFrom_d1, elementFrom_d2 in zip(d1, d2):
    #           tmp.append(elementFrom_d1 * elementFrom_d2)
    #
    # Printing the tmp value out, we would get:
    #       [10, 200, 3000]
    #
    # I then wrap the list comprehension using sum(), which adds up each of the values.
    # Performing this on our tmp list:
    #       sum(tmp) = 3210
    #
    #
    # If you look at the code the Professor originally provided, the below code performs the same
    # function, but with a variable number of layers and lambda values.
    regularizer = sum([a['regLambda'] * tf.nn.l2_loss(b)
                       for a, b in zip(param['layers'], layerWeights)])

    # calculate the loss
    loss = tf.reduce_mean(cross_entropy + regularizer)

    # Use Adam to minimize the loss
    train_step = tf.train.AdamOptimizer(
        learning_rate=param['learning_rate'], beta1=param['beta1'], beta2=param['beta2'], epsilon=param['epsilon']).minimize(loss)

    # Run the session and initialize the variables
    sess.run(tf.global_variables_initializer())

    print("Starting Training...")
    # Only print if DISABLEBATCHPRINT is not set
    if (not DISABLEBATCHPRINT):
        print("epoch\ttrain_accuracy\ttest_accuracy")

    for i in range(3000):
        batch = mnist.train.next_batch(param['batch_size'])

        # Only print if DISABLEBATCHPRINT is not set
        if (not DISABLEBATCHPRINT):
            if i % 100 == 0:
                correct_prediction = tf.equal(tf.argmax(predicted_Y, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                test_accuracy = accuracy.eval(feed_dict=feed_dict_Test)
                train_accuracy = accuracy.eval(feed_dict=feed_dict_Print_Train)

                print("%d \t %.3f \t\t %.3f" % (i, train_accuracy, test_accuracy))

        # Create feed_dict for Training
        feed_dict_Train = {keep_prob_label[i]: keep_prob[i] for i in range(0, len(param['layers']))}
        feed_dict_Train[X] = batch[0]
        feed_dict_Train[Y] = batch[1]

        # TRAIN STEP
        train_step.run(feed_dict=feed_dict_Train)
    # end for loop

    correct_prediction = tf.equal(tf.argmax(predicted_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = accuracy.eval(feed_dict=feed_dict_Test)

    # End timer
    time_duration = timeit.default_timer() - start_time

    print("test accuracy: ", test_accuracy)
    print("test duration: ", time_duration)

    # Write to output file
    # Opening a file using the "with open() as f" is considered good practice when dealing with files.
    # "with open" ensures that the file is closed properly even if an exception is raised.
    # This is much shorter than writing equivalent try-catch-finally blocks
    with open('OutputScript2.xls', 'a', newline='') as outfile:
        # Use Python's csv module to write to an output file with a tab-delimiter
        writer = csv.writer(outfile, delimiter='\t')
        # The csv.writer.writerow function takes a single list as input,
        # and each column is an item in the list
        writer.writerow([test_accuracy, param['layers'], param['learning_rate'], param['beta1'],
                         param['beta2'], param['epsilon'], param['batch_size'], time_duration, outputText])

    # Free up memory by closing the session
    sess.close()


def main():
    # Only use 1 iteration for the file to submit
    num_iterations = 1

    # Specify final values
    layer1 = 1176
    layer2 = 588
    layer3 = 1323
    dropout1 = True
    dropout2 = True
    dropout3 = False  # or True
    keep_prob1 = 0.5
    keep_prob2 = 0.7
    keep_prob3 = 0.6
    lambdaVal1 = 0
    lambdaVal2 = 0
    lambdaVal3 = 0.0008
    lambdaValOut = 0.005
    learnRate = 0.0015
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08
    batchSize = 100

    # Create list of parameters.
    # runParameters is a list of dictionaries.
    # If wanted to have multiple runs, then you can have multiple dictionaries as items in the list.
    # Dictionaries are a data structure that use a key/value pair.
    #
    # The following are global parameters for the run:
    #       learning_rate
    #       batch_size
    #       beta1
    #       beta2
    #       epsilon
    #
    # The other parameters are layer specific.
    # The 'layers' key has a list as its value.
    # The layers list consists of multiple dictionaries, one for each layer.
    # Each layer must specify the following:
    #       layerSize: number of nodes in that layer
    #       dropoutFlag: True, False
    #       nodeType: "ReLU", "Sigmoid", "Linear"
    #       regLambda: Lambda values for that layer
    #       keep_prob: dropout keep probability for that layer
    runParameters = [{'learning_rate': learnRate, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon, 'batch_size': batchSize,
                      'layers': [
                          {'layerSize': layer1, 'dropoutFlag': dropout1,
                           'nodeType': "ReLU", 'regLambda': lambdaVal1, 'keep_prob': keep_prob1},
                          {'layerSize': layer2, 'dropoutFlag': dropout2,
                              'nodeType': "ReLU", 'regLambda': lambdaVal2, 'keep_prob': keep_prob2},
                          {'layerSize': layer3, 'dropoutFlag': dropout3,
                           'nodeType': "ReLU", 'regLambda': lambdaVal3, 'keep_prob': keep_prob3},
                          {'layerSize': 10, 'dropoutFlag': False,
                           'nodeType': "Linear", 'regLambda': lambdaValOut, 'keep_prob': 1}
                      ]}]

    # enumerate over the runParameters
    # If there is only one set of parameters (like there is above), then it just runs once
    for idx, params in enumerate(runParameters):
        # Run over num_iterations to get a good average for these parameters
        for i in range(0, num_iterations):
            # Run algorithm
            runAlgorithm(
                params, "iteration {} of {}".format(i + 1, num_iterations))


# Call the main function
if __name__ == '__main__':
    main()
