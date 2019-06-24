
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
        周波数軸はチャンネル軸として扱います。（全結合される）
    """
    def __init__(self):
        """
        初期化関数
        重み等の変数の初期化を担う
        重みはHeの初期化
        最終層は0.01の分散
        """
        super(Discriminator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 1, 200, 64)
            self.c_0 = L.Convolution2D(1, 128, (6, 4), stride=(2, 2), initialW=w_init).add_hook(spn())
            # (N, 128, 98, 31)
            self.c_1 = L.Convolution2D(128, 256, (10, 7), stride=(2, 4), initialW=w_init).add_hook(spn())
            # (N, 256, 45, 7)
            self.c_2 = L.Convolution2D(256, 512, (5, 5), stride=(4, 2), initialW=w_init).add_hook(spn())
            # (N, 512, 11, 2)
            self.c_3 = L.Convolution2D(512, 5, (3, 2), initialW=chainer.initializers.Normal(2e-4)).add_hook(spn())
            # (N, 3, 7)
    def __call__(self, *_x, **kwargs):
        """
        呼び出し関数
        実際の計算を担う
        Parameter
            ---------
            x: ndarray(tuple)
                特徴量
                range: (-1.0, 1.0)
                shape: [N, 513, 100, 2]
            Returns
            -------
            _y: ndarray
                評価
                range: (-inf, inf)
                shape: [N, 1, 1]
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
        最終層のみ分散0.02
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init = chainer.initializers.HeNormal()
            # (N, 64, 200)
            self.e_0 = L.Convolution1D(64, 128, 2, stride=2, initialW=w_init).add_hook(spn())
            # (N, 128, 100)
            self.e_1 = L.Convolution1D(128, 128, 2, stride=2, initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            # resnet
            self.r_1 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_2 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_3 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_4 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_5 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_6 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_7 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            self.r_8 = L.Convolution1D(128, 256, 11, pad=5, initialW=w_init).add_hook(spn())
            # (N, 256, 50)
            self.d_1 = L.Deconvolution1D(128, 64, 4, stride=4, initialW=w_init).add_hook(spn())
            # (N, 64, 200)
    def __call__(self, *_x, **kwargs):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,36,100,2]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,36,100,2]
        """
        _y = _x[0][:, :, :, 0]
        # encoding
        _y = self.e_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.e_1(_y)
        _y = F.leaky_relu(_y)
        # conversion (single-conv-4blocks-resnet)
        _h = self.r_1(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_2(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_3(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_4(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_5(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_6(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_7(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        _h = self.r_8(_y)
        _y = F.sigmoid(_h[:,:128]) * _h[:,128:] + _y
        # decoding
        _y = self.d_1(_y)
        _y = F.expand_dims(_y, 3)
        return _y
