
import tensorflow as tf


def discriminator(inp,reuse,depth,chs,train=True):
    current=inp
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[2,5], strides=[1,2], padding="VALID",kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),use_bias=False, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
        #                                     name="bnD" + str(i))
        ten=tf.layers.dropout(ten,0.25)
        current = tf.nn.leaky_relu(ten)
    print(" [*] bottom shape:"+str(current.shape))
    h4=tf.reshape(current, [-1,current.shape[1]*current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return ten
def generator(current_outputs,reuse,depth,chs,d,train,r):
    ten=current_outputs
    for i in range(len(d)):
        ten = tf.layers.conv2d(ten, chs[i], kernel_size=[2, 1], strides=[1, 1], padding="VALID",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                               data_format="channels_last", reuse=reuse, name="conv_p" + str(i),
                               dilation_rate=(d[i], 1))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuse,
                                            name="bn_p" + str(i))

        ten = tf.nn.leaky_relu(ten)
    rs=list()
    rms=ten
    for l in range(r):
        ten = block_res(ten, chs, l, depth, reuse, d, train)
        if l!=r-1:
            rs.append(ten)
            ten+=rms
    tenA = ten
    tenA = tf.layers.conv2d(tenA, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                            data_format="channels_last", reuse=reuse, name="res_last2A")
    tenA = tenA * 10
    tenB = tf.concat([tf.stop_gradient(tenA),ten],axis=3)
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="bnBLA")
    tenB = tf.layers.conv2d(tenB, 4, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_last1B")
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="bnBLB")
    tenB = tf.nn.leaky_relu(tenB)
    tenB = tf.layers.conv2d(tenB, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_last2B")
    tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuse,
                                         name="bnBLC")
    tenB = tf.nn.leaky_relu(tenB)
    tenB = tf.layers.conv2d(tenB, 1, [1, 1], [1, 1], padding="SAME",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                            data_format="channels_last", reuse=reuse, name="res_last3B")
    tenB=tenB*3.1416
    ten = tf.concat([tenA, tenB], 3)
    tent=list()
    for f in rs:
        tenA = f
        tenA = tf.layers.conv2d(tenA, 1, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                                data_format="channels_last", reuse=True, name="res_last2A")
        tenA = tenA * 10
        tenB = tf.concat([tf.stop_gradient(tenA),f],axis=3)
        tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=True,
                                             name="bnBLA")
        tenB = tf.layers.conv2d(tenB, 4, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                                data_format="channels_last", reuse=True, name="res_last1B")
        tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=True,
                                             name="bnBLB")
        tenB = tf.nn.leaky_relu(tenB)
        tenB = tf.layers.conv2d(tenB, 1, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                                data_format="channels_last", reuse=True, name="res_last2B")
        tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=True,
                                             name="bnBLC")
        tenB = tf.nn.leaky_relu(tenB)
        tenB = tf.layers.conv2d(tenB, 1, [1, 1], [1, 1], padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                                data_format="channels_last", reuse=True, name="res_last3B")
        tenB *= 3.1416
        tens = tf.concat([tenA, tenB], 3)
        tent.append(tens)
    return ten,tent

def block_res(current,chs,rep_pos,depth,reuses,d,train=True):
    ten = current
    times=depth[0]
    res=depth[1]
    tenM=list()
    tms=len(d)
    for i in range(times):
        tenA = tf.layers.conv2d(ten, chs[i + tms], kernel_size=[1, 4], strides=[1, 4], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),use_bias=False,
                                data_format="channels_last", reuse=reuses, name="convSmaller"+str(i) + str(rep_pos),
                                dilation_rate=(1, 1))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,name="bnA_en"+str(i) + str(rep_pos))
        ten = tf.nn.leaky_relu(tenA)
        tenM.append(ten)

    tms=times+len(d)
    for i in range(res):

        tenA=ten
        ten = tf.layers.conv2d(tenA, chs[tms + i]//2, [1, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_conv1" + str(i) + str(rep_pos))

        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bnA1"+str(tms+i) + str(rep_pos))
        ten = tf.nn.relu(ten)
        ten=tf.layers.conv2d(ten, chs[tms + i]//2, [3, 5], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_conv2" + str(i) + str(rep_pos))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnA2" + str(tms + i) + str(rep_pos))
        ten = tf.nn.relu(ten)
        ten = tf.layers.conv2d(ten, chs[tms + i], [1, 3], [1, 1], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                               data_format="channels_last", reuse=reuses, name="res_conv3" + str(i) + str(rep_pos))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bnA3"+str(tms+i) + str(rep_pos))
        prop=(1-i/(res*2))
        ten=ShakeShake(ten,prop,train)
        if i!=res-1 :
            ten=ten+tenA
        ten = tf.nn.relu(ten)
    tms+=res
    for i in range(times):
        ten += tenM[times-i-1][:, -8:, :, :int(ten.shape[3])]
        ten = deconve_with_ps(ten, [1, 4], chs[tms+i], rep_pos, reuses=reuses, name="00"+str(i))
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                             name="bn"+str(times+res+i) + str(rep_pos))
        ten = tf.nn.relu(ten)
    return ten
def deconve_with_ps(inp,r,otp_shape,depth,reuses=None,name=""):
    chs_r=r[0]*r[1]*otp_shape
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=[1,1], strides=[1,1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=False,
                           data_format="channels_last", reuse=reuses, name="deconv_ps1"+name + str(depth))
    b_size = -1
    in_h = ten.shape[1]
    in_w = ten.shape[2]
    ten = tf.reshape(ten, [b_size, r[0], r[1], in_h, in_w, otp_shape])
    ten = tf.transpose(ten, [0, 2, 3, 4, 1, 5])
    ten = tf.reshape(ten, [b_size, in_h * r[0], in_w * r[1], otp_shape])
    return ten[:,:,:,:]
def ShakeShake(ten,rate,train):
    s=[int(ten.shape[1]),int(ten.shape[2]),int(ten.shape[3])]
    f_rand=tf.random_uniform(s,-1.0,1.0)
    # f_rand = 0.0
    b_rand=tf.random_uniform(s,0.0,1.0)
    # b_rand = 0.5

    if train:
        prop=tf.random_uniform(s,0.0,1.0)+rate
        prop=tf.floor(prop)
        tenA=ten*prop
        tenB = ten*(1-prop)
        return tenA+tenB*b_rand+tf.stop_gradient(tenB*(f_rand-b_rand))
    else:
        return ten*(rate+(1-rate)*0.0)

