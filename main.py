import os.path
import sys
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from math import ceil
from tensorflow.python.framework import dtypes
from time import gmtime, strftime

import project_tests as tests

slim = tf.contrib.slim

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'


    '''with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(processed_images, num_classes=3,
                                        is_training=True)
        

        vgg_input_tensor = end_points[vgg_input_tensor_name]
        vgg_keep_prob_tensor = end_points[vgg_keep_prob_tensor_name]
        vgg_layer3_out_tensor = end_points[vgg_layer3_out_tensor_name]
        vgg_layer4_out_tensor = end_points[vgg_layer4_out_tensor_name]
        vgg_layer7_out_tensor = end_points[vgg_layer7_out_tensor_name]
    '''
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (input_tensor, keep_prob_tensor, layer3_out_tensor,
            layer4_out_tensor, layer7_out_tensor)

tests.test_load_vgg(load_vgg, tf)


def get_bilinear_initializer(name):
    def _initializer(kernel_shape, dtype=dtypes.float32, partition_info=None):
        print("{} bilinear shape {}".format(name, kernel_shape))
        width = kernel_shape[0]
        heigh = kernel_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([kernel_shape[0], kernel_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(kernel_shape)
        for i in range(kernel_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name=name + "_up_filter", initializer=init,
                              shape=weights.shape)
        return var
    return _initializer

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # weight decay
    wd = 5e-4
    num_layer7_features = vgg_layer7_out.get_shape()[3].value 
    kernel_stddev = (2 / num_layer7_features) ** 0.5
    print("vgg_layer7_out {} features using stddev {:.5f}".format(
        num_layer7_features, kernel_stddev))
    pred_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_stddev),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    output = tf.layers.conv2d_transpose(pred_7, num_classes, 4, 2, padding='same',
                                        kernel_initializer=get_bilinear_initializer("pred_7"),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    pred_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    output = tf.add(output, pred_4)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same',
                                        kernel_initializer=get_bilinear_initializer("pred_4"),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    pred_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    output = tf.add(output, pred_3)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                        kernel_initializer=get_bilinear_initializer("pred_3"),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(wd))
    # output = tf.Print(output, [tf.shape(output)])
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=correct_label, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images, correct_label: labels,
                                          keep_prob: 0.8, learning_rate: 1.5e-4})
        print("Epoch {} loss: {:.5f}".format(epoch, loss))
        sys.stdout.flush()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    batch_size = 16
    epochs = 200

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        images, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layers_out = layers(layer3_out, layer4_out, layer7_out, num_classes)
        gt_labels = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, loss_op = optimize(layers_out, gt_labels, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss_op, images, gt_labels, keep_prob,
                 learning_rate)

        time_str = strftime("%H_%M_%S", gmtime())
        model_version = 'checkpoints/model_{}_{}'.format(epochs, time_str)
        saver = tf.train.Saver()
        saver.save(sess, model_version + '.ckpt')
        saver.export_meta_graph(model_version + '.meta')

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, images)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
