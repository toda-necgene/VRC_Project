from glob import glob
import tensorflow as tf
import os
import time
from six.moves import xrange
import numpy as np
import wave
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pyaudio
from . import eve
from datetime import datetime
from tensorflow.python.ops import random_ops
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
class Model:
    def __init__(self,debug):
        self.rate=0.9
        self.p_scale_1=0.
        self.p_scale_2=0.
        self.down=128
        self.up=64
        self.input_ch=256
        self.out_channels=self.down
        self.width=4
        self.dataset_name="wave2wave_ver0.10.0"
        self.data_format=[1,1,80000]
        f=open("Data.txt",'w')
        f.write("Start:"+nowtime())
        f.close()

        self.gf_dim=64
        self.depth=4
        self.batch_size=16

        self.gauth=GoogleAuth()
        self.gauth.LocalWebserverAuth()
        self.drive=GoogleDrive(self.gauth)
        f=self.drive.CreateFile({'title':str(nowtime()+self.dataset_name+'.txt')})
        f.SetContentFile('Data.txt')
        f.Upload()
        self.id_of_result=f['id']

        self.dilations=[]
        self.f_dilations=[]
        self.hance=[]
        #in=25
        self.out_put_size=[self.batch_size,1,256]
        d=1
        self.dilations.append(d)
        for i in range(self.depth):
            d=self.width**(i+1)
            self.dilations.append(d)
        d=self.width**(self.depth)
        self.in_put_size=[self.batch_size,1,d+self.out_put_size[2]]
        for i in range(self.depth):
            self.hance.append((self.width)*self.dilations[i]+1)
        a_in= self.in_put_size[2]-self.out_put_size[2]
        for i in range(self.depth):
            a=a_in//(self.dilations[i+1])
            self.f_dilations.append(a)
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.85)))
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    def build_model(self):
        l1ls=[]
        #変数の予約

        self.var_pear=[]
        self.var=dict()
        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                w=tf.get_variable('w', [2,1,self.input_ch,self.down],initializer=tf.contrib.layers.xavier_initializer())
                l1ls.append(w)
                layer['filter'] = w
                self.var['causal_layer'] = layer

            self.var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i in range(self.depth):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        w= tf.get_variable('w1a', [self.width,1,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l1ls.append(w)
                        current['w-1'] =w
                        w= tf.get_variable('w2a', [self.width,1,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l1ls.append(w)
                        current['w-2'] =w
                        w = tf.get_variable('w3a', [1,1,self.up,self.down],initializer=tf.contrib.layers.xavier_initializer())
                        l1ls.append(w)
                        current['w-3'] =w
                        w = tf.get_variable('w4a', [self.f_dilations[i],1,self.up,self.out_channels],initializer=tf.contrib.layers.xavier_initializer())
                        l1ls.append(w)
                        current['w-4'] =w
                        self.var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                w=tf.get_variable('w1p', [1,1,self.down,127],initializer=tf.contrib.layers.xavier_initializer())
                l1ls.append(w)
                current['postprocess1'] =w
                w= tf.get_variable('w2p', [1,1,127,256],initializer=tf.contrib.layers.xavier_initializer())
                l1ls.append(w)
                current['postprocess2'] =w
                bias = tf.get_variable("bias", [127],initializer=tf.constant_initializer(0.0))
                current['bias']=bias
                l1ls.append(bias)
                bias = tf.get_variable("bias2", [256],initializer=tf.constant_initializer(0.0))
                l1ls.append(bias)
                current['bias2']=bias
                self.var['postprocessing'] = current
        self.var_pear.append(self.var)
        l2ls=[]
        self.var=dict()
        with tf.variable_scope('wavenet2'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                w=tf.get_variable('w', [2,1,self.input_ch,self.down],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                layer['filter'] = w
                self.var['causal_layer'] = layer

            self.var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i in range(self.depth):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        w= tf.get_variable('w1a', [self.width,1,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w-1'] =w
                        w= tf.get_variable('w2a', [self.width,1,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w-2'] =w
                        w = tf.get_variable('w3a', [1,1,self.up,self.down],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w-3'] =w
                        w = tf.get_variable('w4a', [self.f_dilations[i],1,self.up,self.out_channels],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w-4'] =w
                        self.var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                w=tf.get_variable('w1p', [1,1,self.down,127],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                current['postprocess1'] =w
                w= tf.get_variable('w2p', [1,1,127,256],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                current['postprocess2'] =w
                bias = tf.get_variable("bias", [127],initializer=tf.constant_initializer(0.0))
                current['bias']=bias
                l2ls.append(bias)
                bias = tf.get_variable("bias2", [256],initializer=tf.constant_initializer(0.0))
                l2ls.append(bias)
                current['bias2']=bias
                self.var['postprocessing'] = current
            self.var_pear.append(self.var)
        l3ls=[]
        self.var=dict()
        with tf.variable_scope('dis_layer'):
            layer = dict()
            w=tf.get_variable('w1', [2,1,2],initializer=tf.contrib.layers.xavier_initializer())
            l3ls.append(w)
            self.var['w-1'] = w
            w=tf.get_variable('w2', [2,2,4],initializer=tf.contrib.layers.xavier_initializer())
            l3ls.append(w)
            self.var['w-2'] = w
            self.var['causal_layer'] = layer
            w=tf.get_variable('w3', [2,4,8],initializer=tf.contrib.layers.xavier_initializer())
            l3ls.append(w)
            self.var['w-3'] = w
            self.var['causal_layer'] = layer
            w=tf.get_variable('w4', [2,8,1],initializer=tf.contrib.layers.xavier_initializer())
            l3ls.append(w)
            self.var['w-4'] = w
            self.var['causal_layer'] = layer
            self.var_pear.append(self.var)


        self.real_data = tf.placeholder(tf.float32,
                                        self.in_put_size,
                                        name='inputA')
        self.curs = tf.placeholder(tf.float32,
                                        self.in_put_size,
                                        name='current')
        self.inputs_result = tf.placeholder(tf.float32,
                                        [1, 1, self.data_format[2]],
                                        name='input_D')
        self.ans_result = tf.placeholder(tf.float32,
                                        [1, 1, self.data_format[2]],
                                        name='target_D')
        self.real_data_result = tf.placeholder(tf.float32,
                                        [1, 1, self.data_format[2]],
                                        name='target_D')
        self.ans= tf.placeholder(tf.float32,
                                        [self.in_put_size[0], 1, self.out_put_size[2]],
                                        name='target_b')
        self.inputs= tf.placeholder(tf.float32,
                                        [self.in_put_size[0], 1, self.out_put_size[2]],
                                        name='target_d')

        self.is_train= tf.placeholder(tf.bool,
                                        name='is_training')
        self.punish = tf.placeholder(tf.float32,
                                        name='punish')

        self.real_B = tf.reshape( self.encode(self.ans),[self.batch_size,1,-1])
        self.real_A = tf.reshape( self.encode(self.real_data),[self.batch_size,1,-1])
        self.cursa  = tf.reshape( self.encode(self.curs),[self.batch_size,1,-1])
        with tf.variable_scope("generator_1"):
            self.fake_B ,self.fake_B_logit= self.generator(self.one_hot((self.real_A+1.0)/2.*255.0),self.one_hot((self.real_A+1.0)/2.*255.0),False,self.var_pear[0],"1",True)
            self.fake_B_decoded=tf.stop_gradient(self.decode(self.un_oh(self.fake_B)),"asnyan")
        with tf.variable_scope("generator_2"):
            self.fake_B_2 ,self.fake_B_logit_2= self.generator(self.one_hot((self.real_A+1.0)/2.*255.0),self.one_hot((self.real_A+1.0)/2.*255.0),False,self.var_pear[1],"2",True)
            self.fake_B_decoded_2=tf.stop_gradient(self.decode(self.un_oh(self.fake_B_2)),"asnyan")
        self.res1=tf.concat([self.inputs_result,self.real_data_result], axis=2)
        self.res2=tf.concat([self.ans_result,self.real_data_result], axis=2)
        self.res1_s=tf.concat([tf.pad(self.inputs[0:0,:,:],[[0,0],[0,0],[0,self.data_format[2]-self.out_put_size[2]]]),tf.pad(self.real_data[0:0,:,self.in_put_size[2]-self.out_put_size[2]-1:-1],[[0,0],[0,0],[0,self.data_format[2]-self.out_put_size[2]]])], axis=2)
        self.res2_s=tf.concat([tf.pad(self.ans[0:0,:,:],[[0,0],[0,0],[0,self.data_format[2]-self.out_put_size[2]]]),tf.pad(self.real_data[0:0,:,self.in_put_size[2]-self.out_put_size[2]-1:-1],[[0,0],[0,0],[0,self.data_format[2]-self.out_put_size[2]]])], axis=2)
        self.d_judge_F1,self.d_judge_F1_logits=self.discriminator(self.res1,self.var_pear[2],False)
        self.d_judge_R,self.d_judge_R_logits=self.discriminator(self.res2,self.var_pear[2],True)
        self.d_judge_F1_s,self.d_judge_F1_logits_s=self.discriminator(self.res1_s,self.var_pear[2],True)
        self.d_judge_R_s,self.d_judge_R_logits_s=self.discriminator(self.res2_s,self.var_pear[2],True)
        self.d_scale = self.d_judge_R - self.d_judge_F1
        self.g_vars_1=l1ls
        self.g_vars_2=l2ls
        self.d_vars=l3ls
        target=tf.cast(tf.reshape((self.real_B+1.0)/2.0*255.0,[self.batch_size,-1]),dtype=tf.int32)

        logit_1=tf.transpose(self.fake_B_logit, perm=[0,2,1])
        logit_2=tf.transpose(self.fake_B_logit_2, perm=[0,2,1])
#         lo=-tf.reduce_sum(target*tf.log(logit+eps))/self.batch_size
        lo=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit_1)
        lo2=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit_2)

        #l1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.real_B, logits=self.fake_B))
        self.g_loss_1 = lo+0.01*tf.fill(lo.get_shape(), self.punish)
        self.g_loss_2 = lo2+0.01*tf.fill(lo2.get_shape(), self.punish)
        self.d_loss_R = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.d_judge_R), self.d_judge_R_logits )
        self.d_loss_F = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.d_judge_F1), self.d_judge_F1_logits )
        self.d_loss=tf.reduce_mean([self.d_loss_F,self.d_loss_R])
        self.d_loss_R_s = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.d_judge_R_s), self.d_judge_R_logits_s )
        self.d_loss_F_s = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.d_judge_F1_s), self.d_judge_F1_logits_s )
        self.d_loss_s=tf.reduce_mean([self.d_loss_F_s,self.d_loss_R_s])
        self.g_loss_sum = tf.summary.scalar("g_loss_1", tf.reduce_mean(self.g_loss_1))
        self.g_loss_sum_2 = tf.summary.scalar("g_loss_2", tf.reduce_mean(self.g_loss_2))
        self.d_loss_sum = tf.summary.scalar("d_loss", tf.reduce_mean(self.d_loss))

        self.exps=tf.placeholder(tf.float32, [1,1,160000], name="FB")
        self.exps_2=tf.placeholder(tf.float32, [1,1,160000], name="FB2")
        self.fake_B_sum = tf.summary.audio("fake_B_1", tf.reshape(self.exps,[1,160000,1]), 16000, 1)
        self.fake_B_sum_2 = tf.summary.audio("fake_B_2", tf.reshape(self.exps_2,[1,160000,1]), 16000, 1)

        self.g_loss_epo=tf.placeholder(tf.float32,name="g_l_epo")
        self.g_loss_epoch = tf.summary.scalar("g_loss_epoch", self.g_loss_epo)

        self.g_loss_epo_2=tf.placeholder(tf.float32,name="g_l_epo_2")
        self.g_loss_epoch_2 = tf.summary.scalar("g_loss_epoch_2", self.g_loss_epo_2)
        self.rrs=tf.summary.merge([self.fake_B_sum,self.fake_B_sum_2,self.g_loss_epoch,self.g_loss_epoch_2])
        self.saver = tf.train.Saver()

    def convert(self,in_put):
        tt=time.time()
        times=in_put.shape[2]//(self.out_put_size[2])+1
        if in_put.shape[2]%(self.out_put_size[2]*self.batch_size)==0:
            times-=1
        cur_res=np.zeros((self.in_put_size[0],self.in_put_size[1],self.in_put_size[2]),dtype=np.int16)
        cur_res_2=np.zeros((self.in_put_size[0],self.in_put_size[1],self.in_put_size[2]),dtype=np.int16)
        otp=np.array([],dtype=np.int16)
        otp2=np.array([],dtype=np.int16)
        for t in range(times):
            red=np.zeros((self.in_put_size[0]-1,self.in_put_size[1],self.in_put_size[2]))
            start_pos=self.out_put_size[2]*(t)+((in_put.shape[2])%self.out_put_size[2])
            resorce=np.reshape(in_put[0,0,max(0,start_pos-self.in_put_size[2]):min(start_pos,in_put.shape[2])],(1,1,-1))
            r=max(0,self.in_put_size[2]-resorce.shape[2])
            if r>0:
                resorce=np.pad(resorce,((0,0),(0,0),(r,0)),'constant')
            red=np.append(resorce,red,axis=0)
            red=red.reshape((self.in_put_size[0],self.in_put_size[1],self.in_put_size[2]))
            res=self.sess.run(self.fake_B_decoded,feed_dict={ self.real_data:red ,self.curs:cur_res ,self.is_train:False })
            res=res*32767
            cur_res=np.append(cur_res,res,axis=2)
            cur_res=cur_res[:,:,self.out_put_size[2]-1:-1]
            otp=np.append(otp,res[0,0])
            res2=self.sess.run(self.fake_B_decoded_2,feed_dict={ self.real_data:cur_res ,self.curs:cur_res_2 ,self.is_train:False })
            res2=res2*32767
            cur_res_2=np.append(cur_res_2,res2, axis=2)
            cur_res_2=cur_res_2[:,:,self.out_put_size[2]-1:-1]
            otp2=np.append(otp,res2[0,0])
        otp=otp[otp.shape[0]-in_put.shape[2]-1:-1]
        otp2=otp2[otp2.shape[0]-in_put.shape[2]-1:-1]
        return otp.reshape(1,in_put.shape[1],in_put.shape[2]),otp2.reshape(1,in_put.shape[1],in_put.shape[2]),time.time()-tt
    def train(self,args):
        self.checkpoint_dir=args.checkpoint_dir
        lr_g_opt=0.0001
        beta_g_opt=0.9
        lr_g_opt_2=0.0001
        beta_g_opt_2=0.9
        lr_d_opt=0.0001
        beta_d_opt=0.9
        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+"]"
        g_optim_1 = eve.EveOptimizer(lr_g_opt,beta_g_opt).minimize(self.g_loss_1, var_list=self.g_vars_1)
        g_optim_2 = eve.EveOptimizer(lr_g_opt_2,beta_g_opt_2).minimize(self.g_loss_2, var_list=self.g_vars_2)
        d_optim = eve.EveOptimizer(lr_d_opt,beta_d_opt).minimize(self.d_loss, var_list=self.d_vars)
        d_optim_s = eve.EveOptimizer(lr_d_opt,beta_d_opt).minimize(self.d_loss_s, var_list=self.d_vars)
        init_op = tf.global_variables_initializer()
        self.exp= np.zeros((self.batch_size,1,80000),dtype=np.int16)
        self.real_ds= np.zeros((self.batch_size,1,80000),dtype=np.int16)
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs/"+self.lod+self.dataset_name, self.sess.graph)
        counter = 1
        start_time = time.time()
        cv1=None
        cv2=None
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
#         self.experiment=Experiment(self.dataset_name)
#         self.experiment.param("lr_g_opt", lr_g_opt)
#         self.experiment.param("beta_g_opt", beta_g_opt)
        p1=0
        p2=0
        for epoch in range(0,100):
            data = glob('./Model/datasets/train/01/*')
            data2 = glob('./Model/datasets/train/02/*')
            np.random.shuffle(data)
            np.random.shuffle(data2)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            test=load_data('./Model/datasets/test/test.wav').reshape(1,1,160000)
            # test
#             out_puts=self.convert(test)
#             upload(out_puts,self.drive)
            #test end
            gps=0
            gps2=0
            dps=0
            counter=0
            print("Epoch %3d start" % (epoch))
            for idx in xrange(0, batch_idxs):
                cur_res=np.zeros((self.batch_size,1,self.in_put_size[2]),dtype=np.int16)
                cur_res_2=np.zeros((self.batch_size,1,self.in_put_size[2]),dtype=np.int16)
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32).reshape(self.batch_size,2,80000)
                test_train=load_data(data2[idx]).reshape(1,2,80000)

                self.real_ds=batch_images[:,:1,:]
                times=5*16000//self.out_put_size[2]+1
                times_added=0
                if (5*16000)%self.out_put_size[2]==0:
                    times-=1
                ti=(batch_idxs*times)//100+1
                g_score=0
                g_score2=0
                d_score=0
                for t in range(times):
                    start_pos=self.out_put_size[2]*t+((5*16000)%self.out_put_size[2])
                    target=np.reshape(batch_images[:,0,max(0,start_pos-self.out_put_size[2]):start_pos],(self.batch_size,1,-1))
                    resorce=np.reshape(batch_images[:,1,max(0,start_pos-self.in_put_size[2]):start_pos],(self.batch_size,1,-1))
                    r=max(0,self.in_put_size[2]-resorce.shape[2])

                    if r>0:
                        resorce=np.pad(resorce,((0,0),(0,0),(r,0)),'constant')
                    r=max(0,self.out_put_size[2]-target.shape[2])
                    if r>0:
                        target=np.pad(target,((0,0),(0,0),(r,0)),'constant')

                    # Update G1 network
                    _,hg=self.sess.run([g_optim_1,self.g_loss_sum],feed_dict={ self.real_data:resorce, self.curs:cur_res,self.ans:target ,self.is_train:True,self.punish:p1*self.p_scale_1 })
                    if counter % 100==0:
                        self.writer.add_summary(hg, counter//100+ti*epoch)
                    res=self.sess.run(self.fake_B_decoded,feed_dict={ self.real_data:resorce ,self.curs:cur_res, self.ans:target ,self.is_train:False })
                    res=res*32767
                    if counter % 100==0:
                        errG = self.g_loss_1.eval({ self.real_data:resorce, self.curs:cur_res,self.ans:target ,self.is_train:False,self.is_train:False,self.punish:p1*self.p_scale_1})
                        g_score += (np.mean(errG))
                        times_added+=1
                    cur_res=np.append(cur_res,res, axis=2)
                    cur_res=cur_res[:,:,self.out_put_size[2]-1:-1]


                    # Update G2 network
                    _,hg=self.sess.run([g_optim_2,self.g_loss_sum_2],feed_dict={ self.real_data:cur_res, self.curs:cur_res_2,self.ans:target ,self.is_train:True ,self.punish:p2*self.p_scale_2})
                    if counter % 100==0:
                        self.writer.add_summary(hg, counter//100+ti*epoch)
                    res2=self.sess.run(self.fake_B_decoded_2,feed_dict={ self.real_data:cur_res ,self.curs:cur_res_2, self.ans:target ,self.is_train:False })
                    res2=res2*32767
                    if counter % 100==0:
                        errG = self.g_loss_2.eval({ self.real_data:cur_res, self.curs:cur_res_2,self.ans:target ,self.is_train:False,self.is_train:False,self.punish:p2*self.p_scale_2})
                        g_score2 += (np.mean(errG))
                        times_added+=1
                    cur_res_2=np.append(cur_res_2,res2, axis=2)
                    cur_res_2=cur_res_2[:,:,self.out_put_size[2]-1:-1]

                    counter += 1
                    #Training D Netwok
                    if not( cv1 is None) and not( cv2 is None):
                        self.sess.run(d_optim_s,feed_dict={self.real_data:resorce,self.inputs:res,self.ans:target ,self.is_train:True })
                        self.sess.run(d_optim_s,feed_dict={self.real_data:resorce,self.inputs:res2,self.ans:target ,self.is_train:True })

                gps+=(g_score/times_added)
                gps2+=(g_score2/times_added)
                resorce=test_train[0,1,:].reshape(1,1,-1)
                cv1,cv2,_=self.convert(resorce)
                target=test_train[0,0,:].reshape(1,1,-1)
                #update_Punish_scale
                p1a,p1s=self.sess.run([self.d_judge_F1,self.d_scale],feed_dict={ self.real_data_result:resorce ,self.inputs_result:cv1 ,self.ans_result:target ,self.is_train:False })
                p2a,p2s=self.sess.run([self.d_judge_F1,self.d_scale],feed_dict={self.real_data_result:resorce , self.inputs_result:cv2 ,self.ans_result:target ,self.is_train:False })
                p1=np.mean(p1a)
                p2=np.mean(p2a)
                p1=-(np.log(p1+1e-8))
                p2=-(np.log(p2+1e-8))
                self.p_scale_1=max([0,np.mean(p1s)*5.])
                self.p_scale_2=max([0,np.mean(p2s)*5.])
                # Update D network
                self.sess.run(d_optim,feed_dict={self.real_data_result:resorce ,self.inputs_result:cv1,self.ans_result:target ,self.is_train:True })
                _,hd=self.sess.run([d_optim,self.d_loss_sum],feed_dict={self.real_data_result:resorce , self.inputs_result:cv2,self.ans_result:target ,self.is_train:True })
                self.writer.add_summary(hd, counter//100+ti*epoch)
                errD = self.d_loss.eval({ self.real_data_result:resorce ,self.inputs_result:cv1,self.ans_result:target ,self.is_train:False})
                errD_2 = self.d_loss.eval({ self.real_data_result:resorce ,self.inputs_result:cv2,self.ans_result:target ,self.is_train:False})
                d_score += (np.mean(errD))
                d_score += (np.mean(errD_2))
                dps+=(d_score)
            self.save(args.checkpoint_dir, epoch+1)
            ff='Data.txt'
            f=open(ff,'a')
            f.write("-------------------------------\n")
            f.write("TimeStamped:"+nowtime())
            f.write("\nEpoch: [%2d]  time: %3.1f, \n G-LOSS_1: %f \n G-LOSS_2: %f \n D-Loss : %f \n" % (epoch+1,time.time() - start_time,(gps/batch_idxs),(gps2/batch_idxs),(dps/batch_idxs)))
            f.write("-------------------------------\n")
            f.close()
            print("\nEpoch: [%2d]  time: %3.1f, \n G-LOSS_1: %f \n G-LOSS_2: %f \n D-Loss : %f \n" % (epoch+1,time.time() - start_time,(gps/batch_idxs),(gps2/batch_idxs),(dps/batch_idxs)))
#             self.experiment.metric("errG",gps/batch_idxs)
            out_puts,out_put_2,taken_time=self.convert(test)
            out_put=(out_puts.astype(np.float32)/32767.0)
            rs=self.sess.run(self.rrs,feed_dict={ self.exps:out_put,self.exps_2:out_put_2,self.g_loss_epo:(gps/batch_idxs),self.g_loss_epo_2:(gps2/batch_idxs)})
            self.writer.add_summary(rs, epoch+1)
            upload(out_puts,ff,self.drive,self.id_of_result)
            print("test taken: %f secs" % (taken_time))
            start_time = time.time()
#         self.experiment.end()

    def encode(self,in_puts):
        mu=2**8-1.0
        ten=in_puts/32767.0
        inputs = tf.sign(ten,"sign2")*(tf.log(1+mu*tf.abs(ten),name="encode_log_up")/(tf.log(1+mu,name="encode_log_down")))
        return tf.to_float(inputs)
    def one_hot(self,inp):
        inp=tf.cast(inp,tf.int32,name="integer")
        ten=tf.one_hot(inp, 256, axis=-1)
        ten=tf.reshape(ten, [self.batch_size,-1,256])
        ten=tf.transpose(ten, perm=[0,2,1])
        return tf.cast(ten,tf.float32)
    def un_oh(self,in_puts):
        inputs=tf.reshape(in_puts, [self.batch_size,256,-1])
        ten=tf.transpose(inputs, perm=[0,2,1])
        ten=tf.argmax(ten, axis=2,output_type=tf.int32)
        ten=tf.to_float(ten)
        ten=tf.reshape(ten, [self.batch_size,1,-1])
        ten=(ten/255-0.5)*2.0
        return ten
    def decode(self,in_puts):
        ten=in_puts
        mu=2**8-1.0
        inputs=  tf.sign(ten,"sign2")*(1/mu)*((1+mu)**tf.abs(ten)-1)
        return tf.to_float(inputs)

    def discriminator(self,inputs,var,reuse):
        with tf.variable_scope("dis",reuse=tf.AUTO_REUSE):
            w=var['w-1']
            h1 = tf.nn.leaky_relu(tf.nn.conv1d(inputs, w, stride=2, padding="VALID",data_format="NCW",name="dis_01"))
            w=var['w-2']
            h2 = tf.nn.leaky_relu(tf.nn.conv1d(tf.layers.batch_normalization(h1,training=self.is_train,name="dis_bn_02",reuse=reuse), w, stride=2, padding="VALID",data_format="NCW",name="dis_02"))
            w=var['w-3']
            h3 = tf.nn.leaky_relu(tf.nn.conv1d(tf.layers.batch_normalization(h2,training=self.is_train,name="dis_bn_03",reuse=reuse), w, stride=2, padding="VALID",data_format="NCW",name="dis_03"))
            w=var['w-4']
            h4 = tf.nn.leaky_relu(tf.nn.conv1d(tf.layers.batch_normalization(h3,training=self.is_train,name="dis_bn_04",reuse=reuse), w, stride=2, padding="VALID",data_format="NCW",name="dis_04"))

            ten=tf.layers.dense(h4,1,name="dence",reuse=reuse)
            ot=tf.nn.sigmoid(ten)
            return ot,ten

    def generator(self,in_put,current_outputs,reuse,var,name,sd):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        current_output=tf.reshape(current_outputs, [self.batch_size,256,self.in_put_size[2],1])
#         in_puts=tf.cast(in_put, tf.float32)
        #causual
        current = self.causal_layer(current_output,var,reuse,"causual_c"+name)
        #dilation
        outputs=[]
        self.receptive_field = (2 - 1) * sum(self.dilations) + 1
        self.receptive_field += 2 - 1
        for i in range(self.depth):
            otp,current=self.dilation_layer(reuse,current,i,var,name,sd)
            outputs.append(otp)
        current=tf.reshape(current,[self.batch_size,self.down,self.out_put_size[2]])
        outputs.append(current)
        total=sum(outputs)
        transformed=tf.reshape(tf.nn.leaky_relu(total),[self.batch_size,self.down,1,self.out_put_size[2]])
        with tf.variable_scope("posted"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            w=var['postprocessing']['postprocess1']
            w2=var['postprocessing']['postprocess2']
            conv = tf.nn.conv2d(transformed, w, [1,1,1,1], padding="VALID",data_format="NCHW",dilations=[1,1,1,1] ,name="post_01"+name)
            conv = tf.nn.bias_add(conv,var['postprocessing']['bias'],data_format="NCHW")
            transformed=tf.nn.leaky_relu(conv)
            conv = tf.nn.conv2d(transformed, w2, [1,1,1,1], padding="VALID",data_format="NCHW",dilations=[1,1,1,1] ,name="post_02"+name)
            conv = tf.nn.bias_add(conv,var['postprocessing']['bias2'],data_format="NCHW")
            conv=tf.reshape(conv, [self.batch_size,256,-1])
        sm=tf.transpose(conv, perm=[0,2,1])
        sm = tf.nn.softmax(sm,axis=2)
        sm=tf.transpose(sm, perm=[0,2,1])
        return sm , conv
    def causal_layer(self,current_otp,var,reuse,name="causual"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            w =var['causal_layer']['filter']
            res=  tf.nn.conv2d(current_otp, w, [1,1,1,1], padding="VALID",data_format="NCHW",dilations=[1,1,self.dilations[0],1] ,name=name)
            return tf.nn.leaky_relu(res)
    def dilation_layer(self,reuse,in_put,depth,var,name,sd):
        with tf.variable_scope("dil",reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            #前処理

            etan=tf.layers.batch_normalization(in_put,training=self.is_train,name="bn_"+str(depth)+"-"+str(1)+name)
            w=var['dilated_stack'][depth]['w-1']
            etan = dilation_conv(etan, w, "dil_01"+name,self.width,self.up)
            etan=tf.nn.tanh(etan)
            w=var['dilated_stack'][depth]['w-2']
            esig=tf.layers.batch_normalization(in_put,training=self.is_train,name="bn_"+str(depth)+"-"+str(2)+name)
            esig = dilation_conv(esig, w, "dil_02"+name,self.width,self.up)
            d8=tf.multiply(etan,esig)
            d8=tf.layers.batch_normalization(d8,training=self.is_train,name="bn_"+str(depth)+"-"+str(3)+name)
            w=var['dilated_stack'][depth]['w-3']
            otp=tf.nn.conv2d(d8, w, [1,1,1,1], padding="VALID",data_format="NCHW",dilations=[1,1,1,1] ,name="dil_03"+name)
            obs=tf.shape(otp)[2]*tf.shape(otp)[3]
            in_s=(in_put.get_shape())
            inp=tf.reshape(in_put,[self.batch_size,self.down,-1])
            inp=tf.slice(inp,[0,0,0],[-1,-1,obs])
            ten=tf.reshape(inp,[in_s[0],self.down,self.width,in_s[2]//self.width,in_s[3]])
            ten=tf.transpose(ten, [0,1,3,2,4])
            con=tf.reshape(ten,[in_s[0],self.down,in_s[2]//self.width,in_s[3]*self.width])

            w=var['dilated_stack'][depth]['w-4']
            skp=tf.nn.conv2d(d8, w, [1,1,1,1], padding="VALID",data_format="NCHW",dilations=[1,1,1,1] ,name="dil_04"+name)
            skp=tf.reshape(skp,[self.batch_size,self.down,self.out_put_size[2]])
            if sd:
                otp=shake_drop(otp, rate=self.rate**depth,training=self.is_train,name="do_"+str(depth)+"-"+str(1)+name)
            return skp,otp+con

    def save(self, checkpoint_dir, step):
        model_name = "wave2wave.model"
        model_dir = "%s_%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.lod,self.depth)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s_%s layers" % (self.dataset_name, self.batch_size,self.lod,self.depth)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.epoch=self.saver
            return True
        else:
            return False

def dilation_conv(inp,w,name,width,otc):

    ten=tf.nn.conv2d(inp, w, [1,1,1,1], padding="VALID",data_format="NCHW" ,name=name)
    in_s=(ten.get_shape())
    ten=tf.reshape(ten,[in_s[0],otc,width,in_s[2]//width,width,in_s[3]])
    ten=tf.transpose(ten, [0,1,3,2,4])
    ten=tf.reshape(ten,[in_s[0],otc,in_s[2]//width,in_s[3]*width])
    return ten
def shake_drop(in_p,rate,training,name):
    layer=Shake_Dropout(rate,name)
    return layer.apply(in_p,training=training)
class Shake_Dropout(base.Layer):
    def __init__(self,rate,name,**kwargs):
        super(Shake_Dropout, self).__init__(name=name, **kwargs)
        self.rate=1-rate


    def call(self,inputs,training=False):
        def dropped_inputs():
            b=self.rate+random_ops.random_uniform(inputs.shape, name=self.name+"SDR_1")
            bel=tf.floor(b)
            @tf.RegisterGradient(self.name+"Gradient_Ident")
            def _change_grad(op,grad):
                return tf.stop_gradient((1-bel)*grad*(random_ops.random_uniform(inputs.shape, name=self.name)))


            ten1=bel*inputs

            ten2=tf.stop_gradient((1-bel)*inputs*(random_ops.random_uniform(inputs.shape,name=self.name)*2-1))
            g=tf.get_default_graph()
            with g.gradient_override_map({self.name+"Ident": self.name+"Gradient_Ident"}):
                ten3=tf.identity(inputs,self.name+"Ident")
            ten=ten1+ten2+ten3
            return ten
        return utils.smart_cond(training,
                            dropped_inputs,
                            lambda: tf.identity(inputs*(self.rate)+inputs))

def nowtime():
    return datetime.now().strftime("%Y_%m_%d %H_%M_%S")
def upload(voice,res,drive,id_of_result):
    voiced=voice.astype(np.int16)
    p=pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    ww = wave.open("tmp.wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(16000)
    ww.writeframes(voiced.reshape(-1).tobytes())
    ww.close()
    p.terminate()
    f = drive.CreateFile({'id':'1jOONrLOutTRekKM_f23QEcd-FdfC2Pax'})
    f.SetContentFile("tmp.wav")
    f.Upload()
    f = drive.CreateFile({'id':id_of_result})
    f.SetContentFile(res)
    f.Upload()
def load_data(image_path):
    images = imread(image_path)
    images = images
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return images
def imread(path):
    wf = wave.open(path, 'rb')
    ans=[]
    bb=wf.readframes(1024)
    while bb != b'':
        ans.append(bb)
        bb=wf.readframes(1024)
    wf.close()
    return np.frombuffer(b''.join(ans),"int16").reshape([1,160000])