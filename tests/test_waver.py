from VRC.waver import Waver
import os

wv = Waver()
file = os.path.join('..', 'dataset', 'wave', 'A', '200004.wav')
f0, sp, ap, psp = wv.encode(file)
print(psp.shape)
wv.decode(f0, ap, sp=sp, file='resynthe.wav')