
import tensorflow as tf


def discriminator(inp,reuse,depth,chs,train=True):
    current=inp
    for i in range(depth):
        ten = tf.layers.conv2d(current, chs[i], kernel_size=[2,5], strides=[1,2], padding="SAME",kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),use_bias=True, data_format="channels_last",name="disc_"+str(i),reuse=reuse)
        # ten= tf.layers.batch_normalization(ten, trainable=True,training=train,name="bnS"+str(i),reuse=reuse )
        # ten=tf.layers.dropout(ten,0.4,training=True)
        current = tf.nn.leaky_relu(ten)
    print(" [*] bottom shape:"+str(current.shape))
    h4=tf.reshape(current, [-1,current.shape[1]*current.shape[2]*current.shape[3]])
    ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
    return ten
def generator(current_outputs,reuse,depth,chs,d,train,r):
    ten=current_outputs
    rs=list()
    rms=ten
    for l in range(r):
        ten = block_res(ten, chs, l, depth, reuse, d, train)
        if l!=r-1:
            rs.append(ten)
            ten+=rms
    return ten,rs

def block_res(current,chs,rep_pos,depth,reuses,d,train=True):
    ten = current
    times=depth[0]
    res=depth[1]
    tenM=list()
    tms=len(d)
    for i in range(times):
        tenA = tf.layers.conv2d(ten, chs[i + tms], kernel_size=[1, 2], strides=[1, 2], padding="VALID",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),use_bias=True,
                                data_format="channels_last", reuse=reuses, name="convSmaller"+str(i) + str(rep_pos),
                                dilation_rate=(1, 1))
        tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,name="bnA_en"+str(i) + str(rep_pos))
        ten = tf.nn.leaky_relu(tenA)
        tenM.append(ten)

    tms=times+len(d)
    for i in range(res):

        tenA=ten
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnA1" + str(tms + i) + str(rep_pos))

        ten = tf.layers.conv2d(ten, chs[tms + i]//2, [3, 5], [1, 2], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                               data_format="channels_last", reuse=reuses, name="res_conv1" + str(i) + str(rep_pos))

        ten = tf.nn.leaky_relu(ten)
        ten=tf.transpose(ten,[0,1,3,2])
        r=ten.get_shape()[-1]
        ten=tf.layers.dense(ten, r,kernel_initializer=tf.random_normal_initializer(stddev=0.02),  reuse=reuses, name="res_dense" + str(i) + str(rep_pos))
        ten = tf.transpose(ten, [0, 1, 3, 2])
        ten = tf.layers.batch_normalization(ten, axis=3, training=train, trainable=True, reuse=reuses,
                                            name="bnA2" + str(tms + i) + str(rep_pos))
        ten = tf.nn.leaky_relu(ten)
        ten = tf.layers.conv2d_transpose(ten, chs[tms + i], [3, 5], [1, 2], padding="SAME",
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
                               data_format="channels_last", reuse=reuses, name="res_conv3" + str(i) + str(rep_pos))
        prop=(1-i/(res*2))
        ten=ShakeShake(ten,prop,train)
        # ten=tf.layers.dropout(ten,0.2,training=train)
        if i!=res-1 :
            ten=ten+tenA
        ten = tf.nn.leaky_relu(ten)
    tms+=res
    tenA=ten
    for i in range(times):
        tenA += tenM[times-i-1]
        tenA = deconve_with_ps(tenA, [1, 2], chs[tms+i], rep_pos, reuses=reuses, name="00"+str(i))
        if i!=times-1:

            tenA = tf.layers.batch_normalization(tenA, axis=3, training=train, trainable=True, reuse=reuses,
                                                 name="bnAD"+str(times+res+i) + str(rep_pos))
            tenA = tf.nn.leaky_relu(tenA)
    tenB=ten
    for i in range(times):
        tenB += tenM[times-i-1]
        tenB = deconve_with_ps(tenB, [1, 2], chs[tms+i], rep_pos, reuses=reuses, name="01"+str(i))
        if i != times - 1:
            # tenB = tf.layers.dropout(tenB, 0.4, train)
            tenB = tf.layers.batch_normalization(tenB, axis=3, training=train, trainable=True, reuse=reuses,
                                                 name="bnBD"+str(times+res+i) + str(rep_pos))
            tenB = tf.nn.leaky_relu(tenB)
    ten=tf.concat([tenA,tenB],axis=3)
    return ten
def deconve_with_ps(inp,r,otp_shape,depth,reuses=None,name=""):
    chs_r=r[0]*r[1]*otp_shape
    ten = tf.layers.conv2d(inp, chs_r, kernel_size=[1,1], strides=[1,1], padding="VALID",
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), use_bias=True,
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

