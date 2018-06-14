import glob
import numpy as np
import os

f_path="../train/Model/datasets/train/Source_data/"
t_path="../train/Model/datasets/train/04/Source_data/"
NFFT=128
SHIFT=NFFT//2
target=563.666
term = 4096
length=term//SHIFT+1
cutoff=5
ff=glob.glob(f_path+"*-stri.npy")
print(length-cutoff)
for f in ff:
    fs=np.load(f)
    s=length-cutoff-fs.shape[0]
    if s>0:
        fs=np.pad(fs,((0,s),(0,0)),"reflect")
    else:
        fs=fs[:length-cutoff,:]
    n=os.path.basename(f)
    np.save(t_path+n,fs)
print(fs.shape)
print("finished!!")