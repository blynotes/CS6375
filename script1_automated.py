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
# Purpose: Modify hyperparameters using Gradient Descent
#               Optimizer and modified MNIST data to achieve
#               highest test accuracy that you can.
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
DISABLEBATCHPRINT = True


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

    Outputs:
        -test_accuracy:    The test accuracy for the run.
        -time_duration:   The time it took to complete the run.
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

    # Perform Gradient Descent minimizing the loss
    train_step = tf.train.GradientDescentOptimizer(param['learning_rate']).minimize(loss)
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

    # Write to output file information about every run
    # Opening a file using the "with open() as f" is considered good practice when dealing with files.
    # "with open" ensures that the file is closed properly even if an exception is raised.
    # This is much shorter than writing equivalent try-catch-finally blocks
    with open('OutputRunsScript1AutomatedRuns.xls', 'a', newline='') as outfile:
        # Use Python's csv module to write to an output file with a tab-delimiter
        writer = csv.writer(outfile, delimiter='\t')
        # The csv.writer.writerow function takes a single list as input,
        # and each column is an item in the list
        writer.writerow([test_accuracy, param['layers'], param['learning_rate'],
                         param['batch_size'], time_duration, outputText])

    # Free up memory by closing the session
    sess.close()

    return (test_accuracy, time_duration)


def main():
    # Use 3 iterations for each set of parameters to get an average % accuracy
    num_iterations = 3

    # Define ranges for each of the values you can change.
    # numLayerRange specifies the range on the number of Hidden Layers.
    # I determined from script1_checkLayerShapes.py that 2 hidden layers is
    # the optimal number of layers for this project.
    # You can specify a range of values using the following syntax:
    #       numLayerRange = range(2, 5, 1)  # 3 iterations
    # The above example would use 2, 3, and 4 layers.
    # Note that range is non-inclusive of the maximum value specified. [2, 5)
    # I am going to just use the optimal number of layers, since I want to determine
    # what the other parameters should be.
    numLayerRange = [2]

    # layerSizeRange contains how many nodes can go in any layer
    # I want the range to be [20, 196, 392, 588, 784, 980, 1176, 1372, 1568]
    # I cannot only set it to a range() value and have it give me what I want.
    # Instead I have to first initialize the list to 20,
    # and then extend with the range values.
    layerSizeRange = [20]  # 1 iterations
    layerSizeRange.extend(range(196, 1569, 196))  # 8 iterations

    # dropoutRange is True if use dropout and False otherwise
    dropoutRange = [True, False]

    # probRange is the dropout keep probability range
    # Have to use NumPy's arange function since these are decimal values
    probRange = np.arange(0.3, 1, 0.05)  # 14 iterations

    # lambdaRange is the range for the lambda values for regularization
    # Have to use NumPy's arange function since these are decimal values
    lambdaRange = np.arange(0.0001, 0.001, 0.0001)  # 9 iterations

    # learnRange will use 2 ranges put together
    # Some configurations work well with larger learning rates.
    # Other configurations end up stuck in the "0" range of ReLU and
    # end up dying.  These require smaller learning rates
    #
    # Have to use NumPy's arange function since these are decimal values
    learnRange = []
    learnRange.extend(np.arange(0.001, 0.1, 0.005))  # 19 iterations
    # Some configurations work better with larger learning rates
    learnRange.extend(np.arange(0.1, 1, 0.05))  # 20 iterations

    # batchsizeRange holds the range for batch size
    batchsizeRange = range(1, 500, 50)  # 9 iterations

    # paramRanges is a dictionary that holds the ranges for each key
    paramRanges = {
        "numLayers": numLayerRange,
        "layer1": layerSizeRange,
        "layer2": layerSizeRange,  # I later refined this to range(20, 400, 20)
        # "layer3": layerSizeRange,  # don't use layer 3; optimal layer size is 2
        # "layer4": layerSizeRange,  # don't use layer 4; optimal layer size is 2
        "dropout1": dropoutRange,
        "dropout2": dropoutRange,
        # "dropout3": dropoutRange,
        # "dropout4": dropoutRange,
        "keep_prob1": probRange,
        "keep_prob2": probRange,
        # "keep_prob3": probRange,
        # "keep_prob4": probRange,
        'lambdaVal1': lambdaRange,
        'lambdaVal2': lambdaRange,
        # 'lambdaVal3': lambdaRange,
        # 'lambdaVal4': lambdaRange,
        'lambdaValOut': lambdaRange,
        'learnRate': learnRange,
        'batchsize': batchsizeRange
    }

    # initial guesses for best parameter values based on prior tests
    bestNumLayers = 2
    bestLayer1 = 980
    bestLayer2 = 100
    # bestLayer3 = 784
    # bestLayer4 = 20
    bestDropout1 = True
    bestDropout2 = True
    # bestDropout3 = True
    # bestDropout4 = True
    bestKeep_prob1 = 0.35
    bestKeep_prob2 = 0.75
    # bestKeep_prob3 = 0.6
    # bestKeep_prob4 = 0.6
    bestLambdaVal1 = 0
    bestLambdaVal2 = 0.0001
    # bestLambdaVal3 = 0
    # bestLambdaVal4 = 0
    bestLambdaValOut = 0.0005
    bestLearnRate = 0.75
    bestBatchSize = 100

    # values to store after each run
    percent = 0  # Initialize to 0 and take the largest; iterate up from there

    # dictionary used to hold the best parameters for comparison
    bestParams = None

    # For output printing
    runCounter = 0

    # infinite loop
    while(True):
        runCounter += 1

        # Iterate through each parameter and range
        # Capture the values that give the highest % and use that as the new best value
        for key, rangeVal in paramRanges.items():
            # This iterates over each key/value pair in the paramRanges dictionary
            # For example, the first run might iterate over the lambdaVal2 range.
            # Next time through the for loop might iterate over the keep_prob1 range.

            # initialize all values for this run:
            numLayers = bestNumLayers
            layer1 = bestLayer1
            layer2 = bestLayer2
            # layer3 = bestLayer
            # layer4 = bestLayer
            dropout1 = bestDropout1
            dropout2 = bestDropout2
            # dropout3 = bestDropout3
            # dropout4 = bestDropout4
            keep_prob1 = bestKeep_prob1
            keep_prob2 = bestKeep_prob2
            # keep_prob3 = bestKeep_prob3
            # keep_prob4 = bestKeep_prob4
            lambdaVal1 = bestLambdaVal1
            lambdaVal2 = bestLambdaVal2
            # lambdaVal3 = bestLambdaVal3
            # lambdaVal4 = bestLambdaVal4
            lambdaValOut = bestLambdaValOut
            learnRate = bestLearnRate
            batchSize = bestBatchSize

            # Initialize list containing parameters for each run
            multipleRuns = []

            # Go through each value in the range we are looking at, and append to multipleRuns list
            for val in rangeVal:
                # Update numLayers before checking if we can skip the parameter if N/A for numLayers
                if key == "numLayers":
                    # This means we are testing number of layers
                    break  # Not testing number Layers at this time
                    # numLayers = val
                    # print("Key is numLayers: {}".format(val))

                # Check if parameter is N/A based on numLayers
                # If we are currently testing with 2 layers and we are trying to iterate over a range
                # for a parameter that ends with 3 or 4, then we can skip this parameter and move on.
                if numLayers < 3:
                    if key[-1:] in ['3', '4']:
                        # No need to run change to parameter ending in 3 or 4 because don't have that many layers
                        break

                # If we are currently testing with 2 or 3 layers and we are trying to iterate over a range
                # for a parameter that ends with 4, then we can skip this parameter and move on.
                if numLayers < 4:
                    if key[-1:] in ['4']:
                        # No need to run change to parameter ending in 4 because don't have that many layers
                        break

                # Set appropriate variable to be the new value "val" from the range we are iterating over.
                if key == "layer1":
                    layer1 = val
                    print("Key is layer1: {}".format(val))

                elif key == "layer2":
                    layer2 = val
                    print("Key is layer2: {}".format(val))

                # elif key == "layer3":
                #     layer3 = val
                #     print("Key is layer3: {}".format(val))
                #
                # elif key == "layer4":
                #     layer4 = val
                #     print("Key is layer4: {}".format(val))

                elif key == "dropout1":
                    dropout1 = val
                    print("Key is dropout1: {}".format(val))

                elif key == "dropout2":
                    dropout2 = val
                    print("Key is dropout2: {}".format(val))

                # elif key == "dropout3":
                #     dropout3 = val
                #     print("Key is dropout3: {}".format(val))
                #
                # elif key == "dropout4":
                #     dropout4 = val
                #     print("Key is dropout4: {}".format(val))

                elif key == "keep_prob1":
                    keep_prob1 = val
                    print("Key is keep_prob1: {}".format(val))

                elif key == "keep_prob2":
                    keep_prob2 = val
                    print("Key is keep_prob2: {}".format(val))

                # elif key == "keep_prob3":
                #     keep_prob3 = val
                #     print("Key is keep_prob3: {}".format(val))
                #
                # elif key == "keep_prob4":
                #     keep_prob4 = val
                #     print("Key is keep_prob4: {}".format(val))

                elif key == "lambdaVal1":
                    lambdaVal1 = val
                    print("Key is lambdaVal1: {}".format(val))

                elif key == "lambdaVal2":
                    lambdaVal2 = val
                    print("Key is lambdaVal2: {}".format(val))

                # elif key == "lambdaVal3":
                #     lambdaVal3 = val
                #     print("Key is lambdaVal3: {}".format(val))
                #
                # elif key == "lambdaVal4":
                #     lambdaVal4 = val
                #     print("Key is lambdaVal4: {}".format(val))

                elif key == "lambdaValOut":
                    lambdaValOut = val
                    print("Key is lambdaValOut: {}".format(val))

                elif key == "learnRate":
                    learnRate = val
                    print("Key is learnRate: {}".format(val))

                elif key == "batchSize":
                    batchSize = val
                    print("Key is batchSize: {}".format(val))

                elif key == "numLayers":
                    pass  # Do nothing; already handled this at the top of for loop

                else:
                    print("WARNING: Unknown Key: {}".format(key))
                    break

                # Append values to multipleRuns based on numLayers
                #
                # multipleRuns is a list of dictionaries.
                # Each dictionary is for a different run of the algorithm.
                # Dictionaries are a data structure that use a key/value pair.
                #
                # The following are global parameters for the run:
                #       learning_rate
                #       batch_size
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
                if numLayers == 2:
                    print("Number Layers is 2")
                    multipleRuns.append({'learning_rate': learnRate, 'batch_size': batchSize,
                                         'layers': [
                                             {'layerSize': layer1, 'dropoutFlag': dropout1,
                                                 'nodeType': "ReLU", 'regLambda': lambdaVal1, 'keep_prob': keep_prob1},
                                             {'layerSize': layer2, 'dropoutFlag': dropout2,
                                              'nodeType': "ReLU", 'regLambda': lambdaVal2, 'keep_prob': keep_prob2},
                                             {'layerSize': 10, 'dropoutFlag': False,
                                                 'nodeType': "Linear", 'regLambda': lambdaValOut, 'keep_prob': 1}
                                         ]})
                # elif numLayers == 3:
                #     print("Number Layers is 3")
                #     multipleRuns.append({'learning_rate': learnRate, 'batch_size': batchSize,
                #                          'layers': [
                #                              {'layerSize': layer1, 'dropoutFlag': dropout1,
                #                                  'nodeType': "ReLU", 'regLambda': lambdaVal1, 'keep_prob': keep_prob1},
                #                              {'layerSize': layer2, 'dropoutFlag': dropout2,
                #                               'nodeType': "ReLU", 'regLambda': lambdaVal2, 'keep_prob': keep_prob2},
                #                              {'layerSize': layer3, 'dropoutFlag': dropout3,
                #                                  'nodeType': "ReLU", 'regLambda': lambdaVal3, 'keep_prob': keep_prob3},
                #                              {'layerSize': 10, 'dropoutFlag': False,
                #                                  'nodeType': "Linear", 'regLambda': lambdaValOut, 'keep_prob': 1}
                #                          ]})
                # elif numLayers == 4:
                #     print("Number Layers is 4")
                #     multipleRuns.append({'learning_rate': learnRate, 'batch_size': batchSize,
                #                          'layers': [
                #                              {'layerSize': layer1, 'dropoutFlag': dropout1,
                #                                  'nodeType': "ReLU", 'regLambda': lambdaVal1, 'keep_prob': keep_prob1},
                #                              {'layerSize': layer2, 'dropoutFlag': dropout2,
                #                               'nodeType': "ReLU", 'regLambda': lambdaVal2, 'keep_prob': keep_prob2},
                #                              {'layerSize': layer3, 'dropoutFlag': dropout3,
                #                                  'nodeType': "ReLU", 'regLambda': lambdaVal3, 'keep_prob': keep_prob3},
                #                              {'layerSize': layer4, 'dropoutFlag': dropout4,
                #                               'nodeType': "ReLU", 'regLambda': lambdaVal4, 'keep_prob': keep_prob4},
                #                              {'layerSize': 10, 'dropoutFlag': False,
                #                                  'nodeType': "Linear", 'regLambda': lambdaValOut, 'keep_prob': 1}
                #                          ]})
                else:
                    print("INVALID # LAYERS: {}".format(numLayers))
                    sys.exit(1)

            # At this point in the code, All values in the current range have been appended to multipleRuns

            # tmpMaxPercent holds the current best accuracy
            tmpMaxPercent = percent

            # tmpParams holds the best parameters if I find a better accuracy
            tmpParams = None

            # tmpTime holds the time value for the best parameters if I find a better accuracy.
            # This is only used in the output file.
            tmpTime = None

            # Flag to exit early if we are testing Learning Rate and encounter percentages < 0.1
            # This is when we get stuck in the "0" region of ReLU
            exitDueToLearningRate = False

            # Enumerate over all of the runs
            for idx, params in enumerate(multipleRuns):
                # Zero out the average percent.
                # This is used to get the average percentage over num_iterations of the same parameters.
                # I want to find parameters that on average give a high percentage for my project presentation.
                avgPercent = 0

                # Run over num_iterations to get a good average for these parameters
                for i in range(0, num_iterations):
                    # Print information about the current run
                    print("On iteration {} of {} using parameters for run {} of {} for {}".format(
                        i + 1, num_iterations, idx + 1, len(multipleRuns), key))

                    # Run algorithm
                    tmpPercent, tmpTimeVal = runAlgorithm(
                        params, "Automated individual run on iteration {} of {} for {}".format(i + 1, num_iterations, key))

                    # increase avgPercent with the percent just found
                    avgPercent += tmpPercent

                    # If the percent we just found is greater than the previous max percent:
                    #       - Set tmpMaxPercent = percent we just found
                    #       - Set tmpParams = params we just ran with
                    #       - Set tmpTime = time it just took
                    if tmpPercent > tmpMaxPercent:
                        print("Found larger percent: {} > {}".format(tmpPercent, tmpMaxPercent))
                        tmpMaxPercent = tmpPercent
                        tmpParams = None  # Clear it out before copying over
                        tmpParams = params
                        tmpTime = tmpTimeVal  # used just for output file

                    # Check if we are in the "0" region of ReLU AND we are testing on learning rate
                    if (tmpPercent < 0.1) and (key == "learnRate"):
                        # Reached 0 region in the ReLU, and it will not escape
                        # Continuing to take the learning rate higher will not improve
                        exitDueToLearningRate = True
                        print(
                            "Stopping iteration over learning rate values since percent is < 0.1: {}".format(tmpPercent))

                # Print out the average percent accuracy and current parameters.
                # Opening a file using the "with open() as f" is considered good practice when dealing with files.
                # "with open" ensures that the file is closed properly even if an exception is raised.
                # This is much shorter than writing equivalent try-catch-finally blocks
                with open('OutputRunsScript1AverageParams.xls', 'a', newline='') as outfile:
                    # Use Python's csv module to write to an output file with a tab-delimiter
                    writer = csv.writer(outfile, delimiter='\t')
                    # The csv.writer.writerow function takes a single list as input,
                    # and each column is an item in the list
                    writer.writerow(
                        [avgPercent / num_iterations, params, tmpTime, "Average Parameters on run {} for {}".format(runCounter, key)])

                # If in "0" region of ReLU, break out of loop and move onto another parameter
                if exitDueToLearningRate:
                    break

            # If found a better accuracy, then update values accordingly
            if tmpParams is not None:
                # Update max percent
                percent = tmpMaxPercent

                # Update best parameters
                bestParams = tmpParams

                # Update Best values for the parameter we are testing
                if key == "layer1":
                    # extract layerSize for layer1 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestLayer1 = bestParams['layers'][0]['layerSize']

                if key == "layer2":
                    # extract layerSize for layer2 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestLayer2 = bestParams['layers'][1]['layerSize']

                # if key == "layer3":
                #     # extract layerSize for layer3 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestLayer3 = bestParams['layers'][2]['layerSize']
                #
                # if key == "layer4":
                #     # extract layerSize for layer4 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestLayer4 = bestParams['layers'][3]['layerSize']

                if key == "dropout1":
                    # extract dropoutFlag for dropout1 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestDropout1 = bestParams['layers'][0]['dropoutFlag']

                if key == "dropout2":
                    # extract dropoutFlag for dropout2 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestDropout2 = bestParams['layers'][1]['dropoutFlag']

                # if key == "dropout3":
                #     # extract dropoutFlag for dropout3 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestDropout3 = bestParams['layers'][2]['dropoutFlag']
                #
                # if key == "dropout4":
                #     # extract dropoutFlag for dropout4 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestDropout4 = bestParams['layers'][3]['dropoutFlag']

                if key == "keep_prob1":
                    # extract keep_prob for keep_prob1 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestKeep_prob1 = bestParams['layers'][0]['keep_prob']

                if key == "keep_prob2":
                    # extract keep_prob for keep_prob2 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestKeep_prob2 = bestParams['layers'][1]['keep_prob']

                # if key == "keep_prob3":
                #     # extract keep_prob for keep_prob3 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestKeep_prob3 = bestParams['layers'][2]['keep_prob']
                #
                # if key == "keep_prob4":
                #     # extract keep_prob for keep_prob4 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestKeep_prob4 = bestParams['layers'][3]['keep_prob']

                if key == "lambdaVal1":
                    # extract regLambda for lambdaVal1 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestLambdaVal1 = bestParams['layers'][0]['regLambda']

                if key == "lambdaVal2":
                    # extract regLambda for lambdaVal2 from bestParams
                    # convoluted syntax is due to nested data structures
                    bestLambdaVal2 = bestParams['layers'][1]['regLambda']

                # if key == "lambdaVal3":
                #     # extract regLambda for lambdaVal3 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestLambdaVal3 = bestParams['layers'][2]['regLambda']
                #
                # if key == "lambdaVal4":
                #     # extract regLambda for lambdaVal4 from bestParams
                #     # convoluted syntax is due to nested data structures
                #     bestLambdaVal4 = bestParams['layers'][3]['regLambda']

                if key == "lambdaValOut":
                    # extract regLambda from the last layer of bestParams
                    # convoluted syntax is due to nested data structures
                    bestLambdaValOut = bestParams['layers'][len(
                        bestParams['layers']) - 1]['regLambda']

                if key == "learnRate":
                    # extract learning_rate from bestParams
                    bestLearnRate = bestParams['learning_rate']

                if key == "batchSize":
                    # extract batch_size from bestParams
                    bestBatchSize = bestParams['batch_size']

                # Write the best parameters to output file
                # Opening a file using the "with open() as f" is considered good practice when dealing with files.
                # "with open" ensures that the file is closed properly even if an exception is raised.
                # This is much shorter than writing equivalent try-catch-finally blocks
                with open('OutputRunsScript1BestParams.xls', 'a', newline='') as outfile:
                    # Use Python's csv module to write to an output file with a tab-delimiter
                    writer = csv.writer(outfile, delimiter='\t')
                    # The csv.writer.writerow function takes a single list as input,
                    # and each column is an item in the list
                    writer.writerow(
                        [percent, bestParams, tmpTime, "Best Parameters on run {} for {}".format(runCounter, key)])


# Call the main function
if __name__ == '__main__':
    main()
