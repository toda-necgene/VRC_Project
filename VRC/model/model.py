
import tensorflow as tf
import math


def discriminator(inp, reuse):
    # setting paramater
    chs = [64, 128]
    current = inp
    for i in range(len(chs)):
        ten = current

        ten = tf.layers.conv2d(ten, chs[i], kernel_size=[3, 5], strides=[1, 4], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(
                                   stddev=math.sqrt(2.0 / 3 / 5 / chs[i])), use_bias=False, data_format="channels_last",
                               name="disc_" + str(i), reuse=reuse)
        ten = tf.layers.batch_normalization(ten, axis=-1, training=True, trainable=True, reuse=reuse,
                                            name="disc_bn_" + str(i))
        current = tf.nn.leaky_relu(ten)
    # dense
    current = tf.reshape(current, [current.shape[0], current.shape[1], current.shape[2] * current.shape[3]])
    ten = tf.layers.dense(current, 1, name="dence", reuse=reuse,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(0.0002)))

    return ten


def generator(ten, reuse, train):
    ten = tf.transpose(ten, [0, 1, 3, 2])
    ten = tf.layers.conv2d(ten, 513, kernel_size=[3, 1], strides=[1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.002), use_bias=False,
                           data_format="channels_last", reuse=reuse, name="conv2d-last")

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="enc_bn_1_1")

    ten = tf.transpose(ten, [0, 1, 3, 2])

    ten = tf.layers.conv2d(ten, 16, [1, 3], [1, 3], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 9 / 32)),
                           use_bias=False,
                           data_format="channels_last", reuse=reuse, name="res_conv_A_9")

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="enc_bn_1_2")
    ten = tf.nn.leaky_relu(ten)

    ten = tf.layers.conv2d(ten, 64, [1, 3], [1, 3], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3 / 128)),
                           use_bias=False,
                           data_format="channels_last", reuse=reuse, name="res_conv_A_3")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="enc_bn_1_3")
    ten = tf.nn.leaky_relu(ten)
    # resnet 6blocks
    for i in range(6):
        # rate=(i/8/2)+0.5
        # tenA=ShakeDrop(ten,rate,train)
        tenA = ten
        tenA = tf.layers.conv2d(tenA, 64, [3, 3], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 9 / 128)),
                                use_bias=False,
                                data_format="channels_last", reuse=reuse, name="res_conv_C_3x4_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_0_" + str(i))
        ten = tf.nn.leaky_relu(tenA + ten)

    # decodeing

    ten = deconve_with_ps(ten, [1, 3], 8, reuse, "dec_31", False)

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="dec_bn_1_11")
    ten = tf.nn.leaky_relu(ten)
    ten = deconve_with_ps(ten, [1, 3], 1, reuse, "dec_32", True)

    return tf.tanh(ten)


def deconve_with_ps(inp, r, otp_shape, reuses=None, name="", b=True):
    # pixcel shuffler layer

    # calculating output channels
    ch_r = r[0] * r[1] * otp_shape
    in_h = inp.get_shape()[1]
    in_w = inp.get_shape()[2]

    # convolution
    ten = tf.layers.conv2d(inp, ch_r, kernel_size=[1, 1], strides=[1, 1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / ch_r)), use_bias=b,
                           data_format="channels_last", reuse=reuses, name=name)
    # reshaping
    ten = tf.reshape(ten, [-1, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [-1, in_h * r[0], in_w * r[1], otp_shape])

    return ten


