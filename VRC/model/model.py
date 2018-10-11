
import tensorflow as tf
import math

def discriminator(inp,reuse):
    current=inp
    # setting paramater
    depth=4
    chs=[64,64,128,256]

    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,5], strides=[1,4], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2/(3*7*chs[i]))),use_bias=False, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten = tf.layers.batch_normalization(ten, axis=3, training=True, trainable=True, reuse=reuse,
        #                                      name="disc_bn_" + str(i))
        # ten=tf.nn.dropout(ten,0.75)
        current = tf.nn.leaky_relu(ten)
    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/int(current.shape[2]))))

    return ten

def generator(ten,reuse,train):
    # setting paramater
    times=4
    chs_enc=[16,32,32,64]
    chs_dec=[32,32,16,16]
    ten = tf.layers.conv2d(ten, 16, [4, 2], [2, 1], padding="VALID",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3/9/16)),
                            use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_conv_A_pre_")
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="enc_bn0_")
    ten = tf.nn.leaky_relu(ten)
    tens=list()
    for i in range(times):
        tens.append(ten)

        ten=tf.layers.conv2d(ten, chs_enc[i], [1, 2], [1, 2], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/4/chs_enc[i])), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))

        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="enc_bn1_" + str(i))
        ten = tf.nn.leaky_relu(ten)

    ten=tf.transpose(ten,[0,1,3,2])
    rs=int(ten.shape[3])
    ten=tf.layers.dense(ten,rs,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/rs)),use_bias=False,reuse=reuse,name="dense"+str(i))
    ten = tf.transpose(ten, [0, 1, 3, 2])
    ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="res_bn_" + str(i))
    ten = tf.nn.leaky_relu(ten)

    # decodeing
    for i in range(times):
        ten=deconve_with_ps(ten,[1,2],chs_dec[i],reuse,"dec_"+str(i),False)
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="dec_bn1_" + str(i))
        ten = tf.nn.leaky_relu(ten)
        ten+=tens[times-i-1]

    ten= tf.layers.conv2d_transpose(ten ,1, kernel_size=[4, 2], strides=[2, 1], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9/2/32)),use_bias=True,
                                data_format="channels_last", reuse=reuse, name="last_conv1")
    return ten

def deconve_with_ps(inp,r,otp_shape,reuses=None,name="",b=True):
    # pixcel shuffler layer

    # calculating output channels
    ch_r=r[0]*r[1]*otp_shape
    in_h = inp.get_shape()[1]
    in_w = inp.get_shape()[2]

    # convolution
    ten = tf.layers.conv2d(inp, ch_r, kernel_size=[1,1], strides=[1,1], padding="SAME",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/ch_r)), use_bias=b,
                           data_format="channels_last", reuse=reuses, name=name )
    # reshaping
    ten = tf.reshape(ten, [-1, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [-1, in_h * r[0], in_w * r[1], otp_shape])

    return ten
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

