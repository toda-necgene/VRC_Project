
import tensorflow as tf

def discriminator(inp,reuse):
    # setting paramater
    chs=[256,128,64,32,1]
    ten = inp
    ten = tf.transpose(ten, [0, 1, 3, 2])
    # convolution(3*5,stride 1*4)
    for i in range(len(chs)):
        ten = tf.layers.conv2d(ten,chs[i],[3,1],[1,1],"SAME",kernel_initializer=tf.initializers.he_normal(),use_bias=True,name="disc_"+str(i),reuse=reuse)
        if i !=len(chs)-1:
            # ten = tf.layers.batch_normalization(ten, training=True, reuse=reuse,name="disc_bn_" + str(i))
            ten = tf.nn.leaky_relu(ten)
    return ten

def generator(ten,reuse,train):
    ten = tf.transpose(ten, [0, 1, 3, 2])
    ten = tf.layers.conv2d(ten, 64, [3, 1], [1, 1], padding="SAME",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                                     name="encode_conv")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name="encode_bn")
    ten = tf.nn.leaky_relu(ten)
    for i in range(4):
        tenA = tf.layers.conv2d(ten, 64, [3, 1], [1, 1], padding="SAME",
                               kernel_initializer=tf.initializers.he_normal(), use_bias=False, reuse=reuse,
                               name="res_conv_A_"+str(i))
        tenA = tf.layers.batch_normalization(tenA, training=train, reuse=reuse, name="res_bn_A_"+str(i))

        tenB = tf.layers.conv2d(ten, 64, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.initializers.he_uniform(), use_bias=False, reuse=reuse,
                                name="res_conv_B_"+str(i))
        tenB = tf.layers.batch_normalization(tenB, training=train, reuse=reuse, name="res_bn_B_" + str(i))

        ten=tenA*tf.sigmoid(tenB)+ten

    ten = tf.layers.conv2d(ten, 513, [1, 1], [1, 1], padding="VALID",
                                     kernel_initializer=tf.initializers.he_normal(), use_bias=True, reuse=reuse,
                                     name="decode_conv")
    ten = tf.transpose(ten, [0, 1, 3, 2])

    return tf.tanh(ten)