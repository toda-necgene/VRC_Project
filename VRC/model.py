
import tensorflow as tf

def discriminator(inp,reuse):
    ten = tf.transpose(inp,[0,1,3,2])
    w=tf.get_variable("kernel_e",[4,1,513,3],initializer=tf.initializers.random_normal(stddev=0.02))
    b=tf.get_variable("bias_e",[3],initializer=tf.initializers.random_normal())
    w=tf.tanh(w)
    tenE=tf.nn.conv2d(ten,w,[1,1,1,1],"SAME")
    tenE=tf.reshape(tf.exp(tf.nn.bias_add(tenE,b)),[ten.shape[0],ten.shape[1],-1])
    tenA = tf.layers.conv2d(ten,512,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),name="disc_0",reuse=reuse)
    tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn_0")
    ten = tf.nn.leaky_relu(tenA+ten[:,:,:,:512])
    tenA = tf.layers.conv2d(ten,256,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),name="disc_1",reuse=reuse)
    tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn_1")
    ten = tf.nn.leaky_relu(tenA+ten[:,:,:,:256]+ten[:,:,:,256:])
    tenA = tf.layers.conv2d(ten,128,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),name="disc_2",reuse=reuse)
    tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn_2")
    ten = tf.nn.leaky_relu(tenA+ten[:,:,:,:128]+ten[:,:,:,128:])
    tenA = tf.layers.conv2d(ten,64,[4,1],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),name="disc_3",reuse=reuse)
    tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="bn_3")
    ten = tf.nn.leaky_relu(tenA+ten[:,:,:,:64]+ten[:,:,:,64:])
    ten=tf.reshape(ten,[ten.shape[0],ten.shape[1],-1])
    ten = tf.layers.dense(ten, 3, kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True, name="disc_dense", reuse=reuse)
    ten=ten+tenE
    return tf.sigmoid(tf.reshape(ten,[ten.shape[0],ten.shape[1],3]))

def generator(ten,reuse):
    ten=tf.transpose(ten,[0,1,3,2])
    ten = tf.layers.conv2d_transpose(ten, 64, [1, 9],kernel_initializer=tf.initializers.random_normal(stddev=0.002), use_bias=False,
                            reuse=reuse,name="encode_conv_")
    ten = tf.contrib.layers.instance_norm(ten, reuse=reuse, scope="bn_00")
    ten = tf.nn.relu(ten)
    for i in range(4):
        tenA = tf.layers.conv2d(ten,64,[4,4],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse, name="mid_CONV_" + str(i))
        tenA = tf.contrib.layers.instance_norm(tenA, reuse=reuse, scope="CONV_bn" + str(i))
        tenB = tf.layers.conv2d(ten,64,[1,1],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse, name="mid_GATE_" + str(i))
        tenB = tf.contrib.layers.instance_norm(tenB, reuse=reuse, scope="GATE_bn" + str(i))
        tenD = tf.nn.relu(tenA)*tf.nn.tanh(tenB)
        tenD = tf.contrib.layers.instance_norm(tenD, reuse=reuse, scope="mid_bn" + str(i))
        ten += tenD
    ten = tf.layers.conv2d(ten,513,[1,9],kernel_initializer=tf.initializers.random_normal(stddev=0.002),use_bias=True,reuse=reuse,name="decode_conv")
    ten = tf.transpose(ten, [0, 1, 3, 2])
    return tf.tanh(ten)