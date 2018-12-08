
import tensorflow as tf

def discriminator(inp,reuse):
    # setting paramater
    chs=[256,128,1]
    ten = inp
    ten = tf.transpose(ten, [0, 1, 3, 2])
    # convolution(3*5,stride 1*4)
    for i in range(len(chs)):
        ten = tf.layers.conv2d(ten,chs[i],[3,1],[1,1],"SAME",kernel_initializer=tf.initializers.he_normal(),use_bias=False,name="disc_"+str(i),reuse=reuse)
        if i !=len(chs)-1:
            ten = tf.layers.batch_normalization(ten, training=True, reuse=reuse,name="disc_bn_" + str(i))
            ten = tf.nn.leaky_relu(ten)
    # dense
    ten = tf.transpose(ten, [0, 1, 3, 2])
    return ten

def generator(ten,reuse,train):
    ten = tf.transpose(ten, [0, 1, 3, 2])

    tenR=ten
    ten=tf.layers.conv2d_transpose(ten, 171, [3, 3], [1, 3], padding="SAME",
                           kernel_initializer=tf.initializers.he_normal(), use_bias=False,reuse=reuse, name="encode_conv_1")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name="encode_bn_1")

    ten = tf.nn.leaky_relu(ten)
    ten+=ten+tf.reshape(tenR,ten.shape)
    tenR=ten
    ten = tf.layers.conv2d_transpose(ten, 57, [3, 5], [1, 3], padding="SAME",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                                     name="encode_conv_2")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name="encode_bn_2")

    ten = tf.nn.leaky_relu(ten)

    ten=ten+tf.reshape(tenR,ten.shape)
    tenR=ten

    ten = tf.layers.conv2d_transpose(ten, 19, [3, 5], [1, 3], padding="SAME",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                                     name="encode_conv_3")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name="encode_bn_3")

    ten = tf.nn.leaky_relu(ten)

    ten=ten+tf.reshape(tenR,ten.shape)
    tenR=ten
    ten = tf.layers.conv2d_transpose(ten, 6, [1, 7], [1, 3], padding="VALID",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                                     name="encode_conv_4")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name="encode_bn_4")

    ten = tf.nn.leaky_relu(ten)

    ten = tf.layers.conv2d_transpose(ten, 1, [1, 9], [1, 6], padding="VALID",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                                     name="encode_conv_3x1_5")

    ten=ten+tf.reshape(tenR,ten.shape)
    return tf.tanh(ten)