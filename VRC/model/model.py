
import tensorflow as tf
import math

def discriminator(inp,reuse):
    # setting paramater
    depth=3
    chs=[64,128,256]
    current = LineDrop(inp,0.75)
    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,7], strides=[1,6], padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/5/chs[i])),use_bias=False, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True, reuse=reuse,
                                             name="disc_bn_" + str(i))
        current = tf.nn.leaky_relu(ten)
    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(0.0002)))

    return ten

def generator(ten,reuse,train):
    # encoding
    ten = tf.layers.conv2d(ten, 32, [1, 9], [1, 9], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 9 / 32)),
                            use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_conv_A_9")

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="enc_bn_1_1")
    ten = tf.nn.relu(ten)
    ten=tf.layers.conv2d(ten, 64, [1, 3], [1, 3], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/64)), use_bias=False,
                           data_format="channels_last", reuse=reuse, name="res_conv_A_3")

    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="enc_bn_1_2")
    ten = tf.nn.relu(ten)
    # resnet 6blocks
    for i in range(6):
        tenA=tf.layers.conv2d(ten, 32, [3, 5], [1, 2], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/12/32)), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_C_3x4_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_0_" + str(i))
        tenA = tf.nn.leaky_relu(tenA)
        tenA=tf.transpose(tenA,[0,1,3,2])
        out_size=int(tenA.shape[3])
        tenA=tf.layers.dense(tenA,out_size,name="dence"+str(i),reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(0.0002)))
        tenA = tf.transpose(tenA, [0, 1, 3, 2])
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_1_" + str(i))
        tenA = tf.nn.leaky_relu(tenA)
        tenA = tf.layers.conv2d_transpose(tenA, 64, [3, 5], [1, 2], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 12 / 64)),
                                use_bias=False,
                                data_format="channels_last", reuse=reuse, name="res_deconv_C_3x4_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_2_" + str(i))

        # rate=1-(i/6.0)
        # tenA = ShakeDrop(tenA, rate, train)
        ten = tf.nn.leaky_relu(tenA + ten)

    # decodeing

    ten = deconve_with_ps(ten, [1, 3], 32, reuse, "dec_3", False)
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="dec_bn_1_1")
    ten = tf.nn.relu(ten)

    ten = deconve_with_ps(ten, [1, 9], 1, reuse, "dec_9", True)
    return tf.sigmoid(ten)

def deconve_with_ps(inp,r,otp_shape,reuses=None,name="",b=True):
    # pixcel shuffler layer

    # calculating output channels
    ch_r=r[0]*r[1]*otp_shape
    in_h = inp.get_shape()[1]
    in_w = inp.get_shape()[2]

    # convolution
    ten = tf.layers.conv2d(inp, ch_r, kernel_size=[1,1], strides=[1,1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/ch_r)), use_bias=b,
                           data_format="channels_last", reuse=reuses, name=name )
    # reshaping
    ten = tf.reshape(ten, [-1, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [-1, in_h * r[0], in_w * r[1], otp_shape])

    return ten
def LineDrop(ten,rate):
    # linedrop layer
    s = [int(ten.get_shape()[0]), int(ten.get_shape()[1]),1,1]
    # droping
    prop=tf.random_uniform(s,0.0,1.0)+rate
    prop=tf.floor(prop)
    tenA= ten*prop
    return tenA
def ShakeDrop(ten,rate,train):
    # shakedrop layer
    # s=ten.get_shape()
    if train:
        s = [int(ten.get_shape()[0]), int(ten.get_shape()[1]), int(ten.get_shape()[2]), int(ten.get_shape()[3])]
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

