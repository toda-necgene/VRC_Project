
import tensorflow as tf

def discriminator(inp,reuse):
    chs=[64,128,256,512]
    s=[2,2,2,1]
    ten = inp
    for i in range(len(chs)):
        ten = tf.layers.conv2d(ten,chs[i],[4,8],[s[i],4],padding="SAME",kernel_initializer=tf.initializers.he_normal(),use_bias=True,name="disc_"+str(i),reuse=reuse)
        ten = tf.nn.leaky_relu(ten)

    ten = tf.layers.conv2d(ten, 1, [1, 3], kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=True, name="disc_last", reuse=reuse)
    return tf.reshape(ten,[ten.shape[0],ten.shape[1]])

def generator(ten,reuse,training):
    tenA = tf.layers.conv2d(ten, 64, [4, 9], [2,8], padding="VALID",kernel_initializer=tf.initializers.he_normal(), use_bias=False,
                            reuse=reuse,name="guru_conv_A_")
    tenA = tf.layers.batch_normalization(tenA,training=training, reuse=reuse, name="guru_A_bn")

    ten = tf.nn.leaky_relu(tenA)
    ten=tf.transpose(ten,[0,1,3,2])
    for i in range(3):
        tenA = tf.layers.conv2d(ten,64,[4,4],[1,1],"SAME",kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=False,reuse=reuse, name="guru_mid_A_" + str(i))
        tenB = tf.layers.conv2d(tenA,64,[1,1],kernel_initializer=tf.initializers.random_normal(stddev=0.02),use_bias=False,reuse=reuse, name="guru_mid_B_" + str(i))
        tenA = tf.layers.batch_normalization(tenA, training=training, reuse=reuse, name="mid_A_bn" + str(i))
        tenB = tf.layers.batch_normalization(tenB, training=training, reuse=reuse, name="mid_B_bn" + str(i))
        tenD = tf.nn.leaky_relu(tenA)*tf.nn.sigmoid(tenB)
        tenC = tf.layers.conv2d(tenD, 1, [1, 1],kernel_initializer=tf.initializers.random_normal(stddev=0.02), use_bias=False,reuse=reuse, name="guru_mid_C_" + str(i))
        tenC = tf.layers.batch_normalization(tenC,training=training, reuse=reuse, name="mid_C_bn"+str(i))
        ten = tenD*tf.sigmoid(tenC)+ten
    ten = tf.transpose(ten, [0, 1, 3, 2])
    ten = tf.layers.conv2d(ten, 8*8, [1, 3], [1, 1],  padding="SAME",kernel_initializer=tf.initializers.he_normal(), use_bias=False,reuse=reuse,
                           name="guru_deconv_A_")
    ten = tf.reshape(ten,[ten.shape[0],ten.shape[1],ten.shape[2],8,8])
    ten = tf.transpose(ten, [0,1,2,4,3])
    ten = tf.reshape(ten, [ten.shape[0], ten.shape[1], ten.shape[2]*8,8])

    ten = tf.layers.batch_normalization(ten,training=training, reuse=reuse, name="guru_A_debn")
    ten = tf.nn.leaky_relu(ten)
    ten = tf.layers.conv2d_transpose(ten,1,[4,2],[2,1],kernel_initializer=tf.initializers.random_normal(stddev=0.2),use_bias=True,reuse=reuse,name="decode_fc")
    return tf.tanh(ten)