import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    """
    A trainable version VGG16.
    """

    def __init__(self, vgg16_npy_path=None, trainable=True):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("Model Loaded")
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.wd = 5e-4
        
    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.conv_layer2(self.pool5, 512, 4096,7, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        if train_mode is not None:
            self.fc6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.fc6, 0.5), lambda: self.fc6)
        elif self.trainable:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self.conv_layer2(self.fc6, 4096, 4096,1, "fc7")
        if train_mode is not None:
            self.fc7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.fc7, 0.5), lambda: self.fc7)
        elif self.trainable:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)
        self.gap=tf.reduce_mean(self.fc7,[1,2])
        self.fc8 = self.fc_layer(self.gap, 4096, 142, "fc8")

        #self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
        
    def conv_layer2(self, bottom, in_channels, out_channels,filter_size, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        #initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        var_shape=[filter_size, filter_size, in_channels, out_channels]
        filters = self.get_var(var_shape, name, 0, name + "_filters_W")
        #filters = tf.get_variable(name + "_filters_W", shape=[filter_size, filter_size, in_channels, out_channels],initializer=tf.contrib.layers.xavier_initializer())
        weight_decay = tf.mul(tf.nn.l2_loss(filters), self.wd,
                                  name='weight_loss')
        print(name, "L2 Loss Added")
        tf.add_to_collection('losses', weight_decay)
            
        #initial_value = tf.zeros([out_channels])
        var_shape=[out_channels]
        biases = self.get_bias(var_shape, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        var_shape=[in_size, out_size]
        weights = self.get_var(var_shape, name, 0, name + "_weights_W")
        #weights = tf.get_variable(name + "_weights_W", shape=[in_size, out_size],
        #                          initializer=tf.contrib.layers.xavier_initializer())
        weight_decay = tf.mul(tf.nn.l2_loss(weights), self.wd,
                                      name='weight_loss')
        print(name, "L2 Loss Added")
        tf.add_to_collection('losses', weight_decay)
            
        #initial_value = tf.constant(0.1, shape=[out_size])
        var_shape=[out_size]
        biases = self.get_bias(var_shape, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, var_shape, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            var = tf.Variable(value, name=var_name)
            print (name, "Value Loaded")
        else:
            if(name=="fc6" or name=="fc7" or name=="fc8"):
                var=tf.get_variable(var_name,shape=var_shape,initializer=tf.contrib.layers.xavier_initializer())
                print (name, "Value Initialized")
            else:
                var=tf.get_variable(var_name,shape=var_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
                print (name, "2d Value Initialized")
        self.var_dict[(name, idx)] = var

        print var_name, var.get_shape().as_list()
        assert var.get_shape() == var_shape

        return var
    
    def get_bias(self, var_shape, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            var = tf.Variable(value, name=var_name)
            print (name, "Value Loaded")
        else:
            if(name=="fc6" or name=="fc7" or name=="fc8"):
                init_value=tf.constant(0.1, shape=var_shape)
                var=tf.get_variable(var_name,shape=var_shape,initializer=tf.constant_initializer(value=0.1,
                                       dtype=tf.float32))
                print (name, "Value Initialized 0.1")
            else:
                #init_value=tf.zeros(var_shape,dtype=tf.float32)
                var=tf.get_variable(var_name,initializer=tf.zeros_initializer(shape=var_shape,dtype=tf.float32))
                print (name, "Value Initialized 0")
        self.var_dict[(name, idx)] = var

        print var_name, var.get_shape().as_list()
        assert var.get_shape() == var_shape

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count