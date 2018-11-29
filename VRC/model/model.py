
import tensorflow as tf

def discriminator(inp,reuse):
    # setting paramater
    chs=[32,64,128,1]
    ten = inp
    # convolution(3*5,stride 1*4)
    for i in range(len(chs)):
        ten = tf.layers.conv2d(ten,chs[i],[2,6],[1,4],kernel_initializer=tf.initializers.he_normal(),use_bias=False,name="disc_"+str(i),reuse=reuse)
        if i !=len(chs)-1:
            ten = tf.layers.batch_normalization(ten, axis=-1, training=True, trainable=True, reuse=reuse,
                                                name="disc_bn_" + str(i))
            ten = tf.nn.leaky_relu(ten)
    # dense
    return ten

def generator(ten,reuse,train):
    ten = tf.transpose(ten, [0, 1, 3, 2])

    ten=tf.layers.conv2d(ten, 64, [3, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.initializers.he_normal(), use_bias=False,reuse=reuse, name="encode_conv_3x1_3")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="encode_bn_3")

    ten = tf.nn.leaky_relu(ten)
    # resnet 16blocks
    for i in range(16):
        tenA=ten
        tenA = tf.layers.conv2d(tenA, 64, [3, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.initializers.he_normal(),
                                use_bias=False, reuse=reuse, name="residual_conv_A_3x1_" + str(i))
        rate=1-(i/32)
        tenA=ShakeDrop(tenA,rate,train)

        tenB=ten
        tenB = tf.layers.conv2d(tenB, 64, [1, 1], [1, 1],
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                reuse=reuse, name="residual_conv_B_3x1_" + str(i))

        ten=tenA*tf.sigmoid(tenB)
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="residual_bn_" + str(i))

    # decodeing
    ten = tf.layers.conv2d(ten, 513, [3, 1], [1, 1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                           use_bias=True, reuse=reuse, name="decode_conv_3x1_3")

    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)

def ShakeDrop(ten,rate,train):
    # shakedrop layer
    # s=ten.get_shape()
    if train:
        s = [int(ten.get_shape()[0]), int(ten.get_shape()[1]), 1, 1]
        # random noise
        f_rand = tf.random_uniform(s, -1.0, 1.0)
        b_rand = tf.random_uniform(s, 0.0, 1.0)
        # droping
        prop=tf.random_uniform(s,0.0,1.0)+rate
        prop=tf.floor(prop)
        tenA= ten*prop
        tenB = ten*(1-prop)

        # shaking
        ten=tenA+tenB*b_rand+tf.stop_gradient(tenB*(f_rand-b_rand))

        return ten
    else:

        return ten*rate