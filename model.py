import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
# from matplotlib import pyplot as plt

# Create Tensorflow placeholders for the input and output
def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float64, shape = (n_x, None))
    Y = tf.placeholder(tf.float64, shape = (n_y, None))

    return X, Y


# Initializes the weights for a given shape and sets up L2
def initialize_parameters(layer_dims, l2_scale):

    parameters = {}
    layers = len(layer_dims)

    for l in range(1, layers):
        parameters["W" + str(l)] = tf.get_variable(
            shape = [layer_dims[l], layer_dims[l-1]],
            initializer = tf.contrib.layers.variance_scaling_initializer(
                factor = 2.0,
                mode = 'FAN_IN',
                uniform = False
            ),
            name = "W" + str(l),
            dtype = tf.float64,
            regularizer = tf.contrib.layers.l2_regularizer(l2_scale)
        )
        parameters["b" + str(l)] = tf.get_variable(
            "b" + str(l),
            [layer_dims[l], 1],
            initializer = tf.zeros_initializer(),
            dtype = tf.float64
        )

    return parameters


# Defines the forward propagation step: ReLU(W*X + B)
def forward_propagation(X, parameters):

    layers = len(parameters) // 2 # The number of layers in the NN
    A = X

    for l in range(1, layers + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = tf.add(tf.matmul(W, A), b)
        A = tf.maximum(Z, Z*0.05) # Leaky ReLU

    Y = A

    return Y


# Defines the cost function
def compute_cost(A, Y, parameters):

    logits = tf.transpose(A)
    labels = tf.transpose(Y)

    weights = [parameters[name] for name in parameters.keys() if "W" in name]
    # entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    entropy = (labels - logits)**2
    l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    return (tf.reduce_mean(entropy) + sum(l2_losses)), tf.reduce_mean(entropy), l2_losses


# Aggregates all steps and trains
def train(train_data, dev_data, test_data, batch_size, hidden_layers, lr = 0.005, epochs = 1000, l2_scale = .01, silent = False):

    input_size = len(train_data[0][0])  # Size of first input set
    output_size = len(train_data[0][1]) # Size of first output set

    num_batches = int(len(train_data) / batch_size)

    tf.reset_default_graph()

    X, Y = create_placeholders(input_size,output_size)
    parameters = initialize_parameters([input_size]+hidden_layers+[output_size], l2_scale)
    forward_prop = forward_propagation(X, parameters)
    cost, entropy, l2_losses = compute_cost(forward_prop, Y, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
    accuracy = entropy # Implement accuracy at some point

    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        costs = []

        (train_x, train_y) = format_data(train_data)
        (dev_x, dev_y) = format_data(dev_data)

        for epoch in range(epochs):

            minibatches = random_mini_batches(train_data, batch_size)

            epoch_cost = 0
            epoch_accuracy = 0
            for minibatch in tqdm(minibatches, ascii = True, desc = "{0}/{1}".format(epoch+1, epochs), disable = silent):

                (minibatch_x, minibatch_y) = format_data(minibatch)

                mini_accuracy, mini_cost, _ = sess.run([accuracy, cost, optimizer], feed_dict = {X: minibatch_x, Y: minibatch_y})
                epoch_cost += mini_cost / len(minibatches)
                epoch_accuracy += mini_accuracy / len(minibatches)

            if (not silent):

                dev_cost, dev_accuracy = sess.run([cost, accuracy], feed_dict = {X : dev_x, Y : dev_y})
                losses = sess.run(l2_losses, feed_dict = {X : train_x, Y : train_y})

                print("Average training cost: ", epoch_cost)
                print("Dev cost: ", dev_cost)
                print("l2 losses", losses)

        (t_results) = sess.run([forward_prop], feed_dict = { X : format_data(test_data)[0]})
        
        if not silent:
            print("Test cost: ", 0)

        trained_parameters = sess.run(parameters)

        return {"costs" : costs, "results" : t_results, "parameters" : trained_parameters}


# Creates randomized mini batches from a dataset
def random_mini_batches(data, batch_size):

    deck = random.shuffle(data)

    num_batches = len(data) // batch_size

    minibatches = [data[x * batch_size : (x+1) * batch_size] for x in range(num_batches)]

    return minibatches


# Turns an list of input-output lists into a matrix of input vectors and a matrix of output vectors
def format_data(data):

    inputs, outputs = list(zip(*data))
    return (np.array(inputs).T, np.array(outputs).T)


# Computes softmax over an output vector
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis = 0)


# Rounds a onehot to be the most likely class
def round_onehot(a):

    return [1 if e == max(a) else 0 for e in a]
