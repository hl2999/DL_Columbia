import tensorflow as tf
import numpy as np

# Conv layer
def conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

# Average pooling
def pool_layer(input):
    return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def np_gram_matrix(input):
    f = np.reshape(input, (-1, input.shape[3]))
    gram = np.matmul(f.T, f) / f.size
    return gram

def tf_gram_matrix(input):
    shape = input.get_shape()
    F = tf.reshape(input, (-1, input.get_shape()[3].value))                   
    G = tf.matmul(tf.transpose(F), F) / tf.cast(tf.size(F), tf.float32)
    return G

# create vgg19 NN
def vgg_net(input_image, vggmat):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    weight = vggmat['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        layer = name[:4]
        if layer == 'conv':
            kernels, bias = weight[i][0][0][2][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = conv_layer(current, kernels, bias)
        elif layer == 'relu':
            current = tf.nn.relu(current)
        elif layer == 'pool':
            current = pool_layer(current)
        net[name] = current
    return net