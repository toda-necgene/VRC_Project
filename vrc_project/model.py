
"""
製作者:TODA
モデルの定義
Spectral normを追加した事により、学習の安定性と品質が向上
"""
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.link_hooks.spectral_normalization import SpectralNormalization as spn
class Discriminator(chainer.Chain):
    """
        識別側ネットワーク
        周波数軸はチャンネル軸として扱います。（全結合される）
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みはHeの初期化
        最終層は0.0002の分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1, 200, 64)
            self.c_0 = L.Convolution2D(1, 128, (2, 2), stride=(2, 2), initialW=w_init).add_hook(spn())
            # (N, 128, 100, 32)
            self.c_1 = L.Convolution2D(128, 256, (2, 2), stride=(2, 2), initialW=w_init).add_hook(spn())
            # (N, 256, 50, 16)
            self.c_2 = L.Convolution2D(256, 512, (10, 8), stride=(5, 2), initialW=w_init).add_hook(spn())
            # (N, 512, 9, 5)
            self.c_3 = L.Convolution2D(512, 4, (3, 5), initialW=chainer.initializers.Normal(2e-4)).add_hook(spn())
            # (N, 4, 7)
    def __call__(self, *_x, **kwargs):
        """
        呼び出し関数
        実際の計算を担う
        Parameter
            ---------
            x: ndarray(tuple)
                特徴量
                shape: [N, 64, 200, 1] => [N, 1, 200, 64]
            Returns
            -------
            _y: ndarray
                評価
                shape: [N, 3, 7]
        """
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)[:, :, :, 0]
        return _y
class Generator(chainer.Chain):
    """
        生成側ネットワーク
    """
    def __init__(self):
        """
        重み等の変数の初期化を担う
        重みの初期化はHeNormal
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 64, 200)
            self.e_0 = L.Convolution1D(64, 128, 2, stride=2, initialW=w_init).add_hook(spn())
            # (N, 128, 100)
            self.e_1 = L.Convolution1D(128, 128, 11, pad=5, initialW=w_init).add_hook(spn())
            self.e_2 = L.Convolution1D(128, 128, 11, pad=5, initialW=w_init).add_hook(spn())
            # (N, 128, 100)
            self.e_3 = L.Convolution1D(128, 256, 2, stride=2, initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            # resnet
            self.r_1 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_2 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_3 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_4 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_5 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_6 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_7 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_8 = L.Convolution1D(256, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.f_0 = L.Linear(100, 100, initialW=w_init)
            # (N, 256, 50)
            self.d_3 = L.Deconvolution1D(256, 128, 2, stride=2, initialW=w_init).add_hook(spn())
            self.d_2 = L.Convolution1D(128, 128, 11, pad=5, initialW=w_init).add_hook(spn())
            self.d_1 = L.Convolution1D(128, 128, 11, pad=5, initialW=w_init).add_hook(spn())
            self.d_0 = L.Deconvolution1D(128, 64, 2, stride=2, initialW=w_init, nobias=True).add_hook(spn())
            self.b = chainer.Parameter(chainer.initializers.Zero(), (1, 64, 1))
            # (N, 64, 200)
    def __call__(self, *_x, **kwargs):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,64,200,1]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,64,200,1]
        """
        _y = _x[0][:, :, :, 0]
        # encoding
        _y = self.e_0(_y)
        _y = F.relu(_y)
        _h = self.e_1(_y)
        _y = F.relu(_h) +_y
        _h = self.e_2(_y)
        _y = F.relu(_h) +_y
        _y = self.e_3(_y)
        _y = F.relu(_y)
        _ya = _y
        # conversion (single-conv-4blocks-resnet)
        _h = self.r_1(_y)
        _y = F.relu(_h) + _y
        _h = self.r_2(_y)
        _y = F.relu(_h) + _y
        _h = self.r_3(_y)
        _h = F.dropout(_h, 0.1)
        _y = F.relu(_h) + _y
        _h = self.r_4(_y)
        _h = F.dropout(_h, 0.15)
        _y = F.relu(_h) + _y
        _h = self.r_5(_y)
        _h = F.dropout(_h, 0.2)
        _y = F.relu(_h) + _y
        _h = self.r_6(_y)
        _h = F.dropout(_h, 0.25)
        _y = F.relu(_h) + _y
        _h = self.r_7(_y)
        _h = F.dropout(_h, 0.3)
        _y = F.relu(_h) + _y
        _h = self.r_8(_y)
        _h = F.dropout(_h, 0.4)
        _y = F.relu(_h) + _y
        _h = self.f_0(F.reshape(_y, (-1, 100)))
        _h = F.dropout(_h, 0.5)
        _y = F.relu(F.reshape(_h, _y.shape)) + _y
        # decoding
        _y = self.d_3(_y)
        _y = F.relu(_y)
        _h = self.d_2(_y)
        _y = F.relu(_h) + _y
        _h = self.d_1(_y)
        _y = F.relu(_h) + _y
        _y = self.d_0(_y) + self.b
        _y = F.expand_dims(_y, 3)
        _ya = self.d_3(_ya)
        _ya = F.relu(_ya)
        _ha = self.d_2(_ya)
        _ya = F.relu(_ha) + _ya
        _ha = self.d_1(_ya)
        _ya = F.relu(_ha) + _ya
        _ya = self.d_0(_ya) + self.b
        _ya = F.expand_dims(_ya, 3)
        return _y, _ya
