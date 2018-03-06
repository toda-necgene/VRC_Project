'''
Created on 2018/03/02

@author: tadop
'''
import tensorflow as tf
import numpy as np
config = tf.ConfigProto(
            allow_soft_placement=True,gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.8
                )
            )
ints = tf.placeholder(tf.int32,shape=[1])
outs= tf.one_hot(ints,256)

s=tf.InteractiveSession(config=config)
print(s.run(outs, feed_dict={ints:np.array([1],dtype=np.int32)}))