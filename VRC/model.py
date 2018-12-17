
import tensorflow as tf

def discriminator(inp,reuse):
    # setting paramater
    chs=[256,128,64,32,16]
    ten = inp
    ten = tf.transpose(ten, [0, 1, 3, 2])
    # convolution(3*5,stride 1*4)
    for i in range(len(chs)):
        ten = tf.layers.conv2d(ten,chs[i],[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.he_normal(),use_bias=False,name="disc_"+str(i),reuse=reuse)
        ten = tf.layers.batch_normalization(ten, training=True, reuse=reuse,name="disc_bn_" + str(i))
        ten = tf.nn.leaky_relu(ten)

    ten = tf.layers.conv2d(ten, 3, [1, 1], [1, 1], "SAME", kernel_initializer=tf.initializers.random_normal(stddev=0.1),
                           use_bias=True, name="disc_last", reuse=reuse)
    return tf.reshape(ten,[ten.shape[0],ten.shape[1],ten.shape[3]])

def generator(ten,reuse,train):
    # U-net++ depth3
    ten = tf.transpose(ten, [0, 1, 3, 2])

    ten_top_0=ten
    re_ten0=resample(ten_top_0,train,reuse,"re_t0_t1")
    ten_mid_0_0=downsample(ten_top_0,64,4,train,reuse,"down_t0_m00")
    up_ten0=upsample(ten_mid_0_0,513,4,train,reuse,"up_m00_t1")
    ten_top_1=re_ten0*tf.sigmoid(up_ten0)
    ten_bottom=downsample(ten_mid_0_0,8,4,train,reuse,"down_m00_b")
    up_ten1=upsample(ten_bottom,64,4,train,reuse,"up_b_m01")
    re_ten1 = resample(ten_mid_0_0, train, reuse, "re_m00_m01")
    ten_mid_0_1 = re_ten1*tf.tanh(up_ten1)
    re_ten2=resample(ten_top_1,train,reuse,"re_t1_t2")
    up_ten2=upsample(ten_mid_0_1,513,4,train,reuse,"up_m01_t2")
    ten_top_2=re_ten2*tf.sigmoid(up_ten2)
    ten=tf.tanh(ten_top_2)

    ten = tf.transpose(ten, [0, 1, 3, 2])

    return ten

def resample(ten,train,reuse,name):
    ten = tf.layers.conv2d(ten, ten.shape[-1], [1, 1], [1, 1], padding="VALID",
                             kernel_initializer=tf.initializers.random_normal(stddev=0.2), use_bias=True,
                             reuse=reuse,
                             name=name+"_conv")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name=name+"_bn")
    return ten

def upsample(ten,out_ch,size,train,reuse,name):
    ten = tf.layers.conv2d_transpose(ten, out_ch, [size, 1], [1, 1], padding="VALID",
                             kernel_initializer=tf.initializers.random_normal(stddev=0.2), use_bias=True,
                             reuse=reuse,
                             name=name+"_conv")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name=name+"_bn")
    return ten

def downsample(ten,out_ch,size,train,reuse,name):
    ten = tf.layers.conv2d(ten, out_ch, [size, 1], [1, 1], padding="VALID",
                             kernel_initializer=tf.initializers.random_normal(stddev=0.2), use_bias=True,
                             reuse=reuse,
                             name=name+"_conv")
    ten = tf.layers.batch_normalization(ten, training=train, reuse=reuse, name=name+"_bn")
    return ten