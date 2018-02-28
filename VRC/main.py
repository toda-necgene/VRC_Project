'''
Created on 2018/02/16

@author: tadop
'''
from collections import namedtuple
import sys
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
debug=False
# gauth=GoogleAuth()
# gauth.LocalWebserverAuth()
# drive=GoogleDrive(gauth)
# f = drive.CreateFile({'id':'1jOONrLOutTRekKM_f23QEcd-FdfC2Pax'})
# f.SetContentFile("tmp.wav")
# f.Upload()
# print(f['id'])
if __name__ == '__main__':
    pass
from Model import model2 as model
if len(sys.argv)!=0 and sys.argv.__contains__("--debug"):
    debug=True
net = model.Model(debug)
print(" [*]Building Model...")
net.build_model()
print(" [*]Built Model!!")
args=namedtuple('checkpoint_dir', 'train_size')
args.checkpoint_dir="./datasets"
args.train_size=110
net.train(args)
