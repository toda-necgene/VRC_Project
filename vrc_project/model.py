
"""
製作者:TODA
モデルの定義
"""
import chainer
import chainer.links as L
import chainer.functions as F

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
            # self.b_0 = L.BatchNormalization(axis=(2, 3))
            # (N, 1, 100, 513)
            self.c_0 = L.Convolution2D(1, 128, (6, 9), stride=(2, 6), initialW=w_init)
            # (N, 128, 48, 85)
            self.c_1 = L.Convolution2D(128, 256, (10, 9), stride=(2, 4), initialW=w_init)
            # (N, 256, 20, 20)
            self.c_2 = L.Convolution2D(256, 512, (4, 4), stride=(2, 4), initialW=w_init)
            # (N, 512, 9, 5)
            self.c_3 = L.Convolution2D(512, 3, (9, 5), initialW=chainer.initializers.Normal(0.0002))
            # (N, 3, 1)
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
        # _y = self.b_0(_y)
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
        構造は
        (N, 256, 10, 19)形式にエンコード
        single-reluからBNを削除したresnet6ブロック
        入力と同じ(N, 2, 100, 513)にデコード
    """
    def __init__(self):
        """
        重み等の変数の初期化を担う
        重みの初期化はHeNormal
        最終層のみ分散0.02
        """
        super(Generator, self).__init__()
        with self.init_scope():
            w_init_H = chainer.initializers.HeNormal()
            w_init_N = chainer.initializers.Normal(0.002)
            # (N, 513, 200)
            self.e_0 = L.Convolution1D(513, 64, 2, stride=2, initialW=w_init_H)
            # (N, 64, 100)
            self.e_1 = L.Convolution1D(64, 64, 5, stride=5, initialW=w_init_H)
            # (N, 64, 20)
            # resnet
            self.r_1 = L.Convolution1D(64, 64, 7, pad=3, initialW=w_init_N, nobias=True)
            self.b_r1 = L.BatchNormalization(64)
            self.r_2 = L.Convolution1D(64, 64, 7, pad=3, initialW=w_init_N, nobias=True)
            self.b_r2 = L.BatchNormalization(64)
            self.r_3 = L.Convolution1D(64, 64, 7, pad=3, initialW=w_init_N, nobias=True)
            self.b_r3 = L.BatchNormalization(64)
            self.r_4 = L.Convolution1D(64, 64, 7, pad=3, initialW=w_init_N, nobias=True)
            self.b_r4 = L.BatchNormalization(64)
            # (N, 64, 10)
            self.d_1 = L.Deconvolution1D(64, 64, 10, stride=10, initialW=w_init_N)
            # (N, 64, 100)
            self.d_2p = L.Convolution1D(64, 128, 1, initialW=w_init_N, nobias=True)
            self.b_d2 = L.BatchNormalization(128)
            self.d_2 = L.Deconvolution1D(128, 64, 1, initialW=w_init_N)
            # (N, 128, 100) --concatnate with y--> (N, 64 + 64, 100)
            self.d_3p = L.Convolution1D(128, 128, 1, initialW=w_init_N, nobias=True)
            self.b_d3 = L.BatchNormalization(128)
            self.d_3 = L.Deconvolution1D(128, 128, 1, initialW=w_init_N)
            # (N, 128, 100) --concatnate with y--> (N, 128 + 128 = 513, 100)
            self.d_4p = L.Convolution1D(256, 128, 1, initialW=w_init_N, nobias=True)
            self.b_d4 = L.BatchNormalization(128)
            self.d_4 = L.Deconvolution1D(128, 257, 1, initialW=w_init_N)
            # (N, 129, 100) --concatnate with y--> (N, 256 + 257 = 513, 100)
    def __call__(self, *_x, **kwargs):
        """
            呼び出し関数
            実際の計算を担う
            Parameter
            ---------
            x: ndarray(tuple)
                変換前特徴量
                shape: [N,513,100,2]
            Returns
            -------
            _y: ndarray
                変換後特徴量
                shape: [N,513,100,2]
        """
        _y = _x[0][:, :, :, 0]
        # encoding
        _y = self.e_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.e_1(_y)
        _y = F.leaky_relu(_y)
        # conversion (single-conv-4blocks-resnet)
        _h = self.r_1(_y)
        _h = self.b_r1(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_2(_y)
        _h = self.b_r2(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_3(_y)
        _h = self.b_r3(_h)
        _y = F.leaky_relu(_h) + _y
        _h = self.r_4(_y)
        _h = self.b_r4(_h)
        _y = F.leaky_relu(_h) + _y
        # decoding
        _h = self.d_1(_y)
        _y = F.tanh(_h)
        _h = self.d_2p(_y)
        _h = self.b_d2(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_2(_h)
        _y = F.concat([_y, F.tanh(_h)], axis=1)
        _h = self.d_3p(_y)
        _h = self.b_d3(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_3(_h)
        _y = F.concat([_y, F.tanh(_h)], axis=1)
        _h = self.d_4p(_y)
        _h = self.b_d4(_h)
        _h = F.leaky_relu(_h)
        _h = self.d_4(_h)
        _y = F.concat([_y, F.tanh(_h)], axis=1)
        _y = F.expand_dims(_y, 3)
        return F.tanh(_y)
