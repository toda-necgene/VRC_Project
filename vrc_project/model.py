
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
        CNN構造
        4層構造
        活性化関数はleaky_relu（最後のみ無し）
    """
    def __init__(self):
        """
        モデル定義
        重みはHeの初期化
        最終層は0.0002の分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1, 1025, 200)
            self.c_0 = L.Convolution2D(1, 128, (5, 2), stride=(4, 2), initialW=w_init).add_hook(spn())
            # (N, 128, 256, 100)
            self.c_1 = L.Convolution2D(128, 256, (4, 2), stride=(4, 2), initialW=w_init).add_hook(spn())
            # (N, 256, 64, 50)
            self.c_2 = L.Convolution2D(256, 512, (8, 10), stride=(8, 5), initialW=w_init).add_hook(spn())
            # (N, 512, 8, 9)
            self.c_3 = L.Convolution2D(512, 4, (8, 9), pad=(0, 3), initialW=chainer.initializers.Normal(1e-3)).add_hook(spn())
            # (N, 4, 6)
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
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)
        return _y[:, :, 0, :]
class Generator(chainer.Chain):
    """
        生成側ネットワーク
        DilatedConvを用いた6ブロックResnet（内4ブロックがDilated）
        各ブロック
        Conv(use_bias)-Leaky_relu-add
    """
    def __init__(self):
        """
        モデル定義
        重みの初期化はHeの初期化,最終層のみ0.0001分散で固定
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1025, 200)
            self.e_0 = L.Convolution2D(1025, 256, (4, 1), stride=(4, 1), initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            self.r_0 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.r_1 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.r_2 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.r_3 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.r_4 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            self.r_5 = L.Convolution2D(256, 256, (11, 1), pad=(5, 0), initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            self.d_0 = L.Deconvolution2D(256, 1025, (4, 1), stride=(4, 1), initialW=chainer.initializers.HeNormal(0.5)).add_hook(spn())
            # (N, 1025, 200)
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
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        _y = self.e_0(_y)
        _y = F.leaky_relu(_y)
        _y = F.leaky_relu(self.r_0(_y)) + _y
        _y = F.leaky_relu(self.r_1(_y)) + _y
        _y = F.leaky_relu(self.r_2(_y)) + _y
        _y = F.leaky_relu(self.r_3(_y)) + _y
        _y = F.leaky_relu(self.r_4(_y)) + _y
        _y = F.leaky_relu(self.r_5(_y)) + _y
        _y = self.d_0(_y)
        _y = F.transpose(_y, (0, 2, 1, 3))
        return _y
