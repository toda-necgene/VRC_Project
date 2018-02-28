from glob import glob
import tensorflow as tf
import os
import time
from six.moves import xrange
import numpy as np
import wave
from . import eve
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib.debug_data import has_inf_or_nan
from hyperdash import Experiment
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pyaudio
from pydrive import drive
class Model:
    def __init__(self,debug):
        self.gauth=GoogleAuth()
        self.gauth.LocalWebserverAuth()
        self.drive=GoogleDrive(self.gauth)
        self.experiment=Experiment("Wave2Wave_train_test_ver_0.6")
        self.gf_dim=64
        self.depth=4
        self.batch_size=32
        self.dataset_name="wave2wave_ver0.6"
        #in=25
        self.out_put_size=[self.batch_size,1,512]
        a=self.out_put_size[2]
        self.dilations=[]
        for _ in range(self.depth):
            a = a * 2
        a*=2
        self.dilations.append(a)
        self.in_put_size=[self.batch_size,1,a]
        for _ in range(self.depth):
            a = a // 2
            self.dilations.append(a)
        self.dilations.append(1)
        self.skp_out_size=1024
        self.rate=0.4

        self.down=128
        self.up=64
        self.input_ch=256
        self.out_channels=self.down
        self.width=2

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.8
                )
            )
        self.sess=tf.InteractiveSession(config=config)
        if debug:
            self.sess=tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    def build_model(self):
        l2ls=[]
        #変数の予約
        self.var=dict()
        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                w=tf.get_variable('w', [self.width,self.input_ch,self.down],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                layer['filter'] = w
                self.var['causal_layer'] = layer

            self.var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i in range(self.depth):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        w= tf.get_variable('w1a', [self.width,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-1'] =w
                        w= tf.get_variable('w1ia', [2**(i+2),self.input_ch,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-1i'] =w
                        w= tf.get_variable('w2a', [self.width,self.down,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-2'] =w
                        w= tf.get_variable('w2ia', [2**(i+2),self.input_ch,self.up],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-2i'] =w
                        w = tf.get_variable('w3a', [1,self.up,self.down],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-3'] =w
                        w = tf.get_variable('w4a', [self.dilations[i+2],self.up,self.out_channels],initializer=tf.contrib.layers.xavier_initializer())
                        l2ls.append(w)
                        current['w'+str(i)+'-4'] =w
                        self.var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                w=tf.get_variable('w1p', [1,self.down,127],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                current['postprocess1'] =w
                w= tf.get_variable('w2p', [1,127,512],initializer=tf.contrib.layers.xavier_initializer())
                l2ls.append(w)
                current['postprocess2'] =w
                bias = tf.get_variable("bias", [self.out_put_size[2]],initializer=tf.constant_initializer(0.0))
                current['bias']=bias
                l2ls.append(bias)
                bias = tf.get_variable("bias2", [self.out_put_size[2]],initializer=tf.constant_initializer(0.0))
                l2ls.append(bias)
                current['bias2']=bias
                self.var['postprocessing'] = current

        self.real_data = tf.placeholder(tf.float32,
                                        self.in_put_size,
                                        name='inputA')
        self.curs = tf.placeholder(tf.float32,
                                        self.in_put_size,
                                        name='current')
        self.ans= tf.placeholder(tf.float32,
                                        [self.in_put_size[0], 1, self.out_put_size[2]],
                                        name='target_b')
        self.is_train= tf.placeholder(tf.bool,
                                        [1],
                                        name='is_training')
        self.real_A_same = tf.reshape( self.encode(tf.slice(self.real_data,[0,0,self.in_put_size[2]-self.out_put_size[2]],[-1,-1,-1])),[self.batch_size,1,-1])
        self.real_B = tf.reshape( self.encode(self.ans),[self.batch_size,1,-1])
        self.real_A = tf.reshape( self.encode(self.real_data),[self.batch_size,1,-1])
        self.cursa  = tf.reshape( self.encode(self.curs),[self.batch_size,1,-1])
        self.fake_B ,self.fake_B_logit= self.generator(self.one_hot((self.real_A+1.0)/2.*256.0),self.one_hot((self.cursa+1.0)/2.*256.0),False)
        self.fake_B_decoded=tf.stop_gradient(self.decode(tf.clip_by_value(self.un_oh(self.fake_B)+self.real_A_same,-1.0,1.0)),"asnyan")
        self.res_B=self.decode(self.real_B)
        self.g_vars=l2ls


        target=tf.cast(tf.reshape((self.real_B+1.)/2.*256,[self.batch_size,-1]),dtype=tf.int32)

        target_res=target-tf.cast(tf.reshape((self.real_A_same+1.)/2.*256,[self.batch_size,-1]),dtype=tf.int32)+256
        logit=tf.reshape(self.fake_B_logit,[self.batch_size,512,-1])
        logit=tf.transpose(logit, perm=[0,2,1])
#         lo=-tf.reduce_sum(target*tf.log(logit+eps))/self.batch_size
        lo=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_res, logits=logit)
        #l1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.real_B, logits=self.fake_B))
        self.g_loss = lo
        self.g_loss_sum = tf.summary.scalar("g_loss", tf.reduce_mean(self.g_loss))

        self.exps=tf.placeholder(tf.float32, [32,80000,1], name="FB")
        self.fake_B_sum = tf.summary.audio("fake_B", self.exps, 16000, 1)
        self.rrs=tf.summary.merge([self.fake_B_sum])
        self.saver = tf.train.Saver()

    def convert(self,in_put):
        times=5*16000//in_put.shape+1
        if (5*16000)%in_put.shape==0:
            times-=1
        cur_res=np.zeros((self.batch_size,1,self.in_put_size[2]),dtype=np.int16)
        otp=np.zeros((self.batch_size,1,self.in_put_size[2]),dtype=np.int16)
        for t in range(times):
            start_pos=self.out_put_size[2]*t+((5*16000)%self.out_put_size[2])
            resorce=np.reshape(input[max(0,start_pos-self.in_put_size[2]):start_pos],(self.batch_size,1,-1))
            r=max(0,self.in_put_size[2]-resorce.shape[2])
            if r>0:
                resorce=np.pad(resorce,((0,0),(0,0),(r,0)),'constant')
            # Update G network
            res=self.sess.run([self.fake_B_decoded],feed_dict={ self.real_data:resorce ,self.curs:cur_res ,self.is_train:False })
            cur_res=np.append(cur_res,res, axis=2)
            cur_res=cur_res[:,:,self.out_put_size[2]-1:-1]
            self.exp=np.append(self.exp,res, axis=2)
            otp=np.append(self.exp,res*32767, axis=2)
        return otp
    def train(self,args):
        """Train pix2pix"""
        self.checkpoint_dir=args.checkpoint_dir
        lr_g_opt=0.001
        beta_g_opt=0.9
        self.experiment.param("lr_g_opt", lr_g_opt)
        self.experiment.param("beta_g_opt", beta_g_opt)
        self.lod="[glr="+str(lr_g_opt)+",gb="+str(beta_g_opt)+"]"
        g_optim = eve.EveOptimizer(lr_g_opt,beta_g_opt).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.exp= np.zeros((self.batch_size,1,80000),dtype=np.int16)
        self.real_ds= np.zeros((self.batch_size,1,80000),dtype=np.int16)
        self.sess.run(init_op)
        self.g_sum = tf.summary.merge([ self.g_loss_sum])
        self.writer = tf.summary.FileWriter("./logs/"+self.lod, self.sess.graph)
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for epoch in range(0,100):
            data = glob('./Model/datasets/train/*')
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            test=load_data('./Model/datasets/test/test.wav')
            gps=0
            counter=0
            for idx in xrange(0, batch_idxs):
                cur_res=np.zeros((self.batch_size,1,self.in_put_size[2]),dtype=np.int16)
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32).reshape(self.batch_size,2,80000)
                self.real_ds=batch_images[:,:1,:]
                self.exp= np.zeros((self.batch_size,1,80000),dtype=np.int16)
                times=5*16000//self.out_put_size[2]+1
                times_added=0
                if (5*16000)%self.out_put_size[2]==0:
                    times-=1
                ti=(batch_idxs*times)//100+1
                g_score=0
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

                    # Update G network
                    _,hg=self.sess.run([g_optim,self.g_sum],feed_dict={ self.real_data:resorce, self.curs:cur_res,self.ans:target })
                    if counter % 100==0:
                        self.writer.add_summary(hg, counter//100+ti*epoch)
                    res=self.sess.run([self.fake_B_decoded],feed_dict={ self.real_data:resorce ,self.curs:cur_res, self.ans:target ,self.is_train:True })
                    cur_res=np.append(cur_res,res, axis=2)

                    cur_res=cur_res[:,:,self.out_put_size[2]-1:-1]
                    self.exp=np.append(self.exp,res, axis=2)
                    if counter % 100==0:
                        errG = self.g_loss.eval({ self.real_data:resorce, self.curs:cur_res,self.ans:target ,self.is_train:False})
                        g_score += (np.mean(errG))
                        times_added+=1
                    counter += 1
                gps+=(g_score/times_added)
                self.writer.add_summary(hg, counter//100+ti*epoch)
            self.save(args.checkpoint_dir, epoch+1)
            f=open('Z:/Data.txt','a')
            f.write("\nEpoch: [%2d]  time: %4.4f, G-LOSS: %f \n" % (epoch+1,time.time() - start_time,(gps/batch_idxs)))
            print("\nEpoch: [%2d]  time: %4.4f, G-LOSS: %f \n" % (epoch+1, time.time() - start_time,(gps/batch_idxs)))
            self.experiment.metric("errG",gps/batch_idxs)
            f.close()
            out_puts=self.convert(test)
            out_put=out_puts/32767.0
            self.exp=self.exp[:self.batch_size,:,80000:160000]
            rs=self.sess.run(self.rrs,feed_dict={ self.exps:self.exp.reshape([out_put,80000,1])})
            self.writer.add_summary(rs, epoch+1)
            upload(out_puts,self.drive)
            start_time = time.time()
        self.experiment.end()

    def encode(self,in_puts):
        mu=2**8-1.0
        ten=in_puts/32767.0
        inputs = tf.sign(ten,"sign2")*(tf.log(1+mu*tf.abs(ten),name="encode_log_up")/(tf.log(1+mu,name="encode_log_down")))
        return tf.to_float(inputs)
    def one_hot(self,inp):
        inp=tf.cast(inp,tf.int32)
        ten=tf.one_hot(inp, 256, axis=-1)
        ten=tf.reshape(ten, [self.batch_size,-1,256])
        ten=tf.transpose(ten, perm=[0,2,1])
        return tf.cast(ten,tf.float32)
    def un_oh(self,in_puts):
        inputs=tf.reshape(in_puts, [self.batch_size,512,-1])
        ten=tf.transpose(inputs, perm=[0,2,1])
        ten=tf.argmax(ten, axis=2,output_type=tf.int32)
        ten=tf.to_float(ten)
        ten=tf.reshape(ten, [self.batch_size,1,-1])
        ten=(ten/512.0-0.5)*2.0
        return ten
    def decode(self,in_puts):
        ten=in_puts
        mu=2**8-1.0
        inputs=  tf.sign(ten,"sign2")*(1/mu)*((1+mu)**tf.abs(ten)-1)
        return tf.to_float(inputs)

    def generator(self,in_put,current_outputs,reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

#         current_output=tf.cast(current_outputs, tf.float32)
#         in_puts=tf.cast(in_put, tf.float32)
        #causual
        current = self.causal_layer(current_outputs,reuse)
        #dilation
        outputs=[]
        self.receptive_field = (2 - 1) * sum(self.dilations) + 1
        self.receptive_field += 2 - 1
        for i in range(self.depth):
            otp,current=self.dilation_layer(reuse,current,in_put,i)
            outputs.append(otp)
        total=sum(outputs)
        transformed=tf.nn.leaky_relu(total)
        with tf.variable_scope("posted"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            w=self.var['postprocessing']['postprocess1']
            w2=self.var['postprocessing']['postprocess2']
            transformed=tf.layers.batch_normalization(transformed,training=self.is_train)
            conv = tf.nn.conv1d(transformed,w,stride=1,data_format="NCHW",padding='VALID',name="post_01")
            conv = tf.nn.bias_add(conv,self.var['postprocessing']['bias'])
            transformed=tf.nn.leaky_relu(conv)
            transformed=tf.layers.batch_normalization(transformed,training=self.is_train)
            conv = tf.nn.conv1d(transformed,w2,stride=1,data_format="NCHW",padding='VALID',name="post_02")
            conv = tf.nn.bias_add(conv,self.var['postprocessing']['bias2'])
        sm=tf.transpose(conv, perm=[0,2,1])
        sm = tf.nn.softmax(sm,axis=2)
        sm=tf.transpose(sm, perm=[0,2,1])
        return sm , conv
    def causal_layer(self,current_otp,reuse):
        with tf.variable_scope("causual",reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            w =self.var['causal_layer']['filter']
            res= tf.nn.conv1d(current_otp,w,stride=2,data_format="NCHW",padding='VALID',name="cursual_layer_conv1d")
            return tf.nn.leaky_relu(res)
    def dilation_layer(self,reuse,in_put,global_cond,depth):
        with tf.variable_scope("dil",reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            #前処理
            etan=tf.layers.batch_normalization(in_put,training=self.is_train)
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-1']
            etan = tf.nn.conv1d(etan, w, stride=2,data_format="NCHW", padding='VALID',name="conv1d_"+str(depth)+"-"+str(1))
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-1i']
            etan=etan + tf.nn.conv1d(global_cond, w, stride=2**(depth+2),data_format="NCHW", padding='VALID',name="conv1d_"+str(depth)+"-"+str(1.5))
            etan=tf.nn.tanh(etan)
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-2']
            esig=tf.layers.batch_normalization(in_put,training=self.is_train)
            esig = tf.nn.conv1d(esig, w, stride=2,data_format="NCHW", padding='VALID',name="conv1d_"+str(depth)+"-"+str(2))
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-2i']
            esig=esig + tf.nn.conv1d(global_cond, w, stride=2**(depth+2),data_format="NCHW", padding='VALID',name="conv1d_"+str(depth)+"-"+str(2.5))
            esig=tf.nn.sigmoid(esig)
            d8=tf.multiply(etan,esig)
            d8=tf.layers.batch_normalization(d8,training=self.is_train)
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-3']
            d9=tf.layers.dropout(d8, rate=self.rate,training=self.is_train)
            otp=tf.nn.conv1d(d9, w, stride=1, padding="SAME",data_format="NCHW", name="dense")
            obs=tf.shape(in_put)[2]-tf.shape(otp)[2]
            w=self.var['dilated_stack'][depth]['w'+str(depth)+'-4']
            skp=tf.nn.conv1d(d8, w, stride=1, padding="VALID",data_format="NCHW", name="skip")
            return skp,otp+tf.slice(in_put, [0,0,obs],[-1,-1,-1])

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
def upload(voice,drive):
    p=pyaudio.PyAudio()
    FORMAT = p.paInt16
    ww = wave.open("tmp.wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(16000)
    ww.writeframes(voice.tobytes())
    ww.close()
    p.terminate()
    f = drive.CreateFile({'id':'1jOONrLOutTRekKM_f23QEcd-FdfC2Pax'})
    f.SetContentFile("tmp.wav")
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