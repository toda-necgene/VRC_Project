
import tensorflow as tf
import math

def discriminator(inp,reuse):
    current=inp
    # setting paramater
    depth=4
    chs=[16,32,64,128]

    # convolution(2*4,stride 1*4)
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[3,7], strides=[1,4], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2/(3*7*chs[i]))),use_bias=True, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten=tf.layers.dropout(ten,0.4)
        current = tf.nn.leaky_relu(ten)

    # dense
    current=tf.reshape(current,[current.shape[0],current.shape[1],current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(current,1,name="dence",reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2)))

    return ten
def generator(ten,reuse,train):

    ten = tf.layers.conv2d(ten, 16, [3, 3], [1, 1], padding="SAME",
                              kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / 3 / 3)),
                              use_bias=True,
                              data_format="channels_last", reuse=reuse, name="res_conv_PRE")
    # setting paramater
    res=3
    for i in range(res):
        #inception resblock
        tenA =tf.layers.conv2d(ten, 16, [3, 1], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/1/16)), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="res_conv_A_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_A_" + str(i))
        tenA=tf.nn.relu(tenA)
        tenB = tf.transpose(tenA, [0, 1, 3, 2])
        hs=int(tenB.shape[3])//4
        tenC=list()
        for j in range(4):
            tenC.append(tf.layers.dense(tenB[:,:,:,hs*j:hs*(j+1)],hs,use_bias=True,reuse=reuse,name="res_dense"+str(j) + str(i)))
        tenE = tf.add_n(tenC)
        tenE = tf.layers.dense(tenE,hs,use_bias=False,reuse=reuse,name="res_denseF" + str(i))
        for j in range(4):
            tenC[j]=tenC[j]*tf.tanh(tenE)
        tenD = tf.concat(tenC, axis=3)
        tenF = tf.transpose(tenD, [0, 1, 3, 2])
        # adding noise(shakedrop)
        prop = (1 - i / (res * 2))
        tenG = ShakeDrop(tenF, prop, train)

        tenG = tf.layers.batch_normalization(tenG, axis=3, training=train, trainable=True, reuse=reuse,
                                             name="res_bn_B" + str(i))
        ten=tenG+ten

    tenPOW=tf.layers.conv2d(ten, 1, [3, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/3)), use_bias=True,
                               data_format="channels_last", reuse=reuse, name="res_conv_POW_")

    tenPHA=tf.layers.conv2d(ten, 1, [3, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/3/3)), use_bias=True,
                               data_format="channels_last", reuse=reuse, name="res_conv_PHA_")
    # concating and activation
    ten=tf.concat([tenPOW,tenPHA],axis=3)
    ten=tf.tanh(ten)
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

