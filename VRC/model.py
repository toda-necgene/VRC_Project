
import tensorflow as tf

def discriminator(inp,reuse):
    ten = tf.transpose(inp,[0,1,3,2])
    ten = tf.layers.conv2d(ten,64,[4,1],[2,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=False,name="disc_0",reuse=reuse)
    ten = tf.nn.leaky_relu(ten)
    ten=tf.nn.dropout(ten,0.85)
    ten = tf.layers.conv2d(ten,64,[3,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=False,name="disc_1",reuse=reuse)
    ten = tf.nn.leaky_relu(ten)
    ten=tf.nn.dropout(ten,0.85)
    ten=tf.reshape(ten,[ten.shape[0],ten.shape[1],-1])
    ten = tf.layers.dense(ten, 1, kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True, name="disc_dense", reuse=reuse)
    return tf.sigmoid(tf.reshape(ten,[ten.shape[0],ten.shape[1]]))

def generator(ten,reuse):
    ten=tf.transpose(ten,[0,1,3,2])
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 9],kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False,
                            reuse=reuse,name="encode_conv_")
    ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_00")
    ten = tf.nn.relu(ten)
    for i in range(4):
        tenA = tf.layers.conv2d(ten,64,[4,4],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse, name="mid_CONV_" + str(i))
        tenB = tf.layers.conv2d(ten,64,[1,9],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse, name="mid_GATE_" + str(i))
        tenD = tf.nn.relu(tenA)*tf.nn.tanh(tenB)
        tenD = tf.contrib.layers.instance_norm(tenD, reuse=reuse, scope="mid_bn" + str(i))
        ten += tenD
    ten = tf.layers.conv2d(ten,513,[1,9],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse,name="decode_conv")
    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)