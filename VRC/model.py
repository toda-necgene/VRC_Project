
import tensorflow as tf

def discriminator(inp,reuse):
    ten = inp
    ten = tf.layers.conv2d(ten,128,[4,10],[4,8],kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True,name="disc_0",reuse=reuse)
    # ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_0")
    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d(ten,64,[3,10],[1,8],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True,name="disc_1",reuse=reuse)
    # ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_1")
    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d(ten,32,[2,10],[1,8],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True,name="disc_2",reuse=reuse)
    # ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_2")
    ten = tf.nn.leaky_relu(ten)
    ten=tf.reshape(ten,[ten.shape[0],ten.shape[1],-1])
    ten = tf.layers.dense(ten, 1, kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True, name="disc_dense", reuse=reuse)
    return tf.sigmoid(tf.reshape(ten,[ten.shape[0],ten.shape[1]]))

def generator(ten,reuse):
    ten=tf.transpose(ten,[0,1,3,2])
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 9],kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False,
                            reuse=reuse,name="encode_conv_")
    for i in range(3):
        ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="mid_bn" + str(i))
        ten = tf.nn.leaky_relu(ten)
        tenA = tf.layers.conv2d(ten,64,[3,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_CONV_" + str(i))
        tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="mid_bnA" + str(i))
        tenA = tf.nn.leaky_relu(tenA)
        tenA = tf.transpose(tenA,[0,1,3,2])
        tenA = tf.layers.conv2d(tenA,9,[1,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_CONV2_" + str(i))
        tenA = tf.transpose(tenA,[0,1,3,2])
        tenB = tf.layers.conv2d(tenA,64,[1,1],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=False,reuse=reuse, name="mid_GATE_" + str(i))
        ten = tenA*tf.nn.sigmoid(tenB)
        
    ten = tf.contrib.layers.instance_norm(ten,reuse=reuse, scope="bn_last")
    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d(ten,513,[1,9],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse,name="decode_conv")
    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)