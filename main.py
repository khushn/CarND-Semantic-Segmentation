import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


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
    #Note the loader.load should be called exactly as expected in the test_load_vgg
    # function. For a while, I was not putting vgg_tag in square brackets!
    vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)    

    vgg_input_tensor_name = 'image_input:0'
    vgg_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    vgg_layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #1st step 1x1 convolution instead of dense layer 
    #regularizer is to be added as suggested in the project walkthrought video

    output1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #Decoder 1st level    



    #num_filters = 512 # should be the same dimension as skip layer we are going to add after 
                     # the transpose
    dec1 = tf.layers.conv2d_transpose(output1x1, num_classes, 4, strides=(2, 2), padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

     #This post provided the missing piece
    #https://discussions.udacity.com/t/what-is-the-output-layer-of-the-pre-trained-vgg16-to-be-fed-to-layers-project/327033/25
    #As otherwise I was getting shape mistmatch
    #as vgg_layer4_out is 512 plates (i.e. no. of filters)
    # we need to get itto just 2 plates
    vgg_layer4_out_thinned = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #Add the skip layer 4 of the encoder part to it
    skip1 = tf.add(dec1, vgg_layer4_out_thinned)

    #Decoder 2nd level
    dec2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2), padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #Again reduce the number of plates of layer 3
    vgg_layer3_out_thinned = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #Add the skip layer 3 of the encoder part to it
    skip2 = tf.add(dec2, vgg_layer3_out_thinned)

    
    output = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding='same', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

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

    #logits
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #tensor flow can take care of working on loss even in case of any dimensions 
    # so no need to to reshape it to 2-d
    #Apparently we do so commenting it out
    #logits = nn_last_layer 

    #cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    #train_op
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
        loss=cross_entropy_loss,
        global_step=tf.train.get_global_step())


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
    #image_input1 = tf.placeholder(tf.float32, [None, None, None, 3])

    init = tf.global_variables_initializer()
    #this is run only once
    sess.run(init)

    for epoch_i in range(epochs):
        print("epoch_i: ", epoch_i)
        images_generator = get_batches_fn(batch_size) 
        for images, gt_images in images_generator:                                    
            sess.run([train_op, cross_entropy_loss], 
                feed_dict={input_image: images, correct_label: gt_images, 
                            keep_prob: 0.5})
        print('cross_entropy_loss: ', cross_entropy_loss)


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs=5
    batch_size=2
    learning_rate=.05

    #image_ph = tf.placeholder(tf.float32, [None, None, None, 3])
    gt_image_ph = tf.placeholder(tf.float32, [None, None, None, num_classes])
    #keep_prob = tf.placeholder(tf.float32, [])

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training_sample'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layers_output, gt_image_ph, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             gt_image_ph, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
