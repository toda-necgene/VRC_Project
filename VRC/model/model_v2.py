
import tensorflow as tf
import math

def discriminator(inp,reuse):
    # setting paramater
    chs=[64,64,128]
    current = inp
    # convolution(3*5,stride 1*4)
    for i in range(len(chs)):
        ten=current
        ten = tf.layers.conv2d(ten, chs[i], kernel_size=[3,6], strides=[1,4], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/7/chs[i])),use_bias=False, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        ten = tf.layers.batch_normalization(ten, axis=-1, training=True, trainable=True, reuse=reuse,
                                            name="disc_bn_" + str(i))
        current = tf.nn.leaky_relu(ten)
    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(0.002)))

    return ten

def generator(ten,reuse,train):
    ten = tf.transpose(ten, [0, 1, 3, 2])
    ten = tf.layers.conv2d(ten, 128, kernel_size=[3, 1], strides=[1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/128)), use_bias=False,
                           data_format="channels_last", reuse=reuse, name="encode_conv_3x1_1")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="encode_bn_1")
    ten = tf.nn.leaky_relu(ten)

    ten=tf.layers.conv2d(ten, 64, [3, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/64)), use_bias=False,
                           data_format="channels_last", reuse=reuse, name="encode_conv_3x1_3")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="encode_bn_3")

    ten = tf.nn.leaky_relu(ten)
    # resnet 6blocks
    for i in range(16):
        # rate=(i/8/2)+0.5
        # tenA=ShakeDrop(ten,rate,train)
        tenA=ten
        tenA = tf.layers.conv2d(tenA, 64, [3, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3 / 64)),
                                use_bias=False,
                                data_format="channels_last", reuse=reuse, name="residual_conv_C_3x1_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="residual_bn_" + str(i))

        ten = tf.nn.leaky_relu(tenA + ten)

    # decodeing
    ten = tf.layers.conv2d(ten, 128, [3, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3 / 128)),
                           use_bias=False,
                           data_format="channels_last", reuse=reuse, name="decode_conv_3x1_1")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                        name="decode_bn_1")

    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d(ten, 513, [3, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3 / 513)),
                           use_bias=True,
                           data_format="channels_last", reuse=reuse, name="decode_conv_3x1_3")

    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)

