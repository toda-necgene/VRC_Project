
"""
製作者:TODA
モデルの定義
"""
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.link_hooks.spectral_normalization import SpectralNormalization as spn



class Discriminator(chainer.Chain):
    """
        識別側ネットワーク
        4層CNN(Conv1d-leaky_relu)
        最終層のみチャンネルを32づつ4グループにして各グループで最大値をとる
    """
    def __init__(self, chs=None):
        """
        モデル定義
        Parameter
        ---------
        chs : list of int
        各レイヤーごとの出力チャンネル数
        """
        super(Discriminator, self).__init__()
        if chs is None:
            chs = [512, 256, 128]
        with self.init_scope():
            he_init = chainer.initializers.HeNormal()
            # (N, 1025, 200)
            self.c_0 = L.Convolution1D(1025, chs[0], 6, stride=2, pad=2, initialW=he_init).add_hook(spn())
            # (N, 512, 100)
            self.c_1 = L.Convolution1D(chs[0], chs[1], 6, stride=2, pad=2, initialW=he_init).add_hook(spn())
            # (N, 256, 50)
            self.c_2 = L.Convolution1D(chs[1], chs[2], 10, stride=5, pad=0, initialW=he_init).add_hook(spn())
            # (N, 128, 9)
            self.c_3 = L.Convolution1D(chs[2], 128, 9, pad=4, initialW=he_init)
            # (N, 4, 7)
    def __call__(self, *_x, **kwargs):
        """
        モデルのグラフ実装
        Parameter
            ---------
            x: ndarray(tuple)
                特徴量
                shape: [N, 1024, 200, 1]
            Returns
            -------
            _y: ndarray
                評価
                shape: [N, 4, 3]
        """
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)
        _y = F.max(F.reshape(_y, (-1 ,32, 4, 9)), axis=1)
        return _y
class Generator(chainer.Chain):
    """
        学習用生成側ネットワーク
    """
    def __init__(self):
        """
        レイヤー定義
        Parameter
        ---------
        chs : int
        residualblockのチャンネル数
        layres : int
        residualblockの数
        """
        super(Generator, self).__init__()
        he_init = chainer.initializers.HeNormal()
        with self.init_scope():
            self.e = L.Convolution2D(1025, chs, (4, 1), stride=(4, 1), initialW=he_init).add_hook(spn())
            self.c11 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c12 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c21 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c22 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c31 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c32 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c41 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.c42 = L.Convolution2D(chs, chs, (5, 1), initialW=he_init).add_hook(spn())
            self.d = L.Deconvolution2D(chs, 1025, (4, 1), stride=(4, 1), initialW=he_init).add_hook(spn())

    def __call__(self, *_x, **kwargs):
        """
            モデルのグラフ実装
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,1025,200,1]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,1025,200,1]
        """
        links = self.children()
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        _y = self.e(_y)
        _y = F.leaky_relu(_y)
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 5, 10))
        # 1
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c11(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c12(_h)
        _y = F.leaky_relu(_h) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 10, 5))
        # 2
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c21(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c22(_h)
        _y = F.leaky_relu(_h) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 25, 2))
        # 3
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c31(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c32(_h)
        _y = F.leaky_relu(_h) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 50, 1))
        # 4 
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c41(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = self.c42(_h)
        _y = F.leaky_relu(_h) + _y

        _y = self.d(_y)
        _y = F.transpose(_y, (0, 2, 1, 3))
        return _y

class GeneratorSimple(chainer.Chain):
    """
        実行用生成側ネットワーク
        要 model_convert.py
    """
    def __init__(self, chs=256, layers=6):
        """
        レイヤー定義
        """
        super(GeneratorSimple, self).__init__()
        with self.init_scope():
            self.e = L.Convolution2D(1025, chs, (4, 1), stride=(4, 1))
            self.c11 = L.Convolution2D(chs, chs, (5, 1))
            self.c12 = L.Convolution2D(chs, chs, (5, 1))
            self.c21 = L.Convolution2D(chs, chs, (5, 1))
            self.c22 = L.Convolution2D(chs, chs, (5, 1))
            self.c31 = L.Convolution2D(chs, chs, (5, 1))
            self.c32 = L.Convolution2D(chs, chs, (5, 1))
            self.c41 = L.Convolution2D(chs, chs, (5, 1))
            self.c42 = L.Convolution2D(chs, chs, (5, 1))
            self.d = L.Deconvolution2D(chs, 1025, (4, 1), stride=(4, 1))

    def __call__(self, *_x, **kwargs):
        """
            実行
        """
        links = self.children()
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        _y = F.leaky_relu(self.e(_y))
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 5, 10))
        # 1
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c11(_h))
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c12(_h)) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 10, 5))
        # 2
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c21(_h))
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c22(_h)) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 25, 2))
        # 3
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c31(_h))
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c32(_h)) + _y
        _y = F.reshape(_y, (_y.shape[0], _y.shape[1], 50, 1))
        # 4 
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c41(_h))
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (4, 0), (0, 0)), mode="constant") 
        _h = F.leaky_relu(self.c42(_h)) + _y
        _y = self.d(_y)
        _y = F.transpose(_y, (0, 2, 1, 3))
        return _y
