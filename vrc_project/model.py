
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
            # (N, 1, 200, 513)
            self.c_0 = L.Convolution2D(1, 64, (6, 9), stride=(2, 9), initialW=w_init)
            # (N, 64, 98, 57)
            self.c_1 = L.Convolution2D(64, 128, (6, 6), stride=(2, 3), initialW=w_init)
            # (N, 128, 47, 18)
            self.c_2 = L.Convolution2D(128, 256, (9, 4), stride=(2, 2), initialW=w_init)
            # (N, 256, 20, 8)
            self.c_3 = L.Convolution2D(256, 512, (3, 8), stride=(1, 1), initialW=w_init)
            # (N, 512, 18, 1)
            self.d_l = L.Convolution2D(512, 1, (4, 1), stride=(2, 1), initialW=chainer.initializers.Normal(0.0002))
            # (N, 1, 8, 1)
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
        _y = self.c_3(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_l(_y)[:, 0, :, :]
        return _y
class GeneratorBlock(chainer.Chain):
    """
    generatorのブロック部
    現在はAtention構造に似たもの
    """
    def __init__(self):
        """
        通常の3x3-128ch畳み込みと周波数方向をチャンネルとして扱い1x1畳み込み
        """
        super(GeneratorBlock, self).__init__()
        with self.init_scope():
            self.conv_A = L.Convolution2D(18, 18, (1, 1), initialW=chainer.initializers.HeNormal(), nobias=True)
            self.bn = L.BatchNormalization(18)
            self.conv_B = L.Convolution2D(128, 128, (9, 1), pad=(4, 0), initialW=chainer.initializers.Normal(0.02))

    def __call__(self, *_x, **kwargs):
        """
        Residualブロック
        """
        _h = F.transpose(_x[0], (0, 3, 2, 1))
        _h = self.conv_A(_h)
        _h = self.bn(_h)
        _h = F.transpose(_h, (0, 3, 2, 1))
        _h = F.relu(_h)
        _a = self.conv_B(_h)
        return F.tanh(_a) * _h + _x[0]
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
            w_init_N = chainer.initializers.Normal(0.02)
            # (N, 2, 200, 513)
            self.e_0 = L.Convolution2D(1, 64, (2, 3), stride=(2, 3), initialW=w_init_H)
            # (N, 64, 100, 171)
            self.e_1 = L.Convolution2D(64, 128, (2, 3), stride=(2, 3), initialW=w_init_H)
            # (N, 128, 50, 57)
            self.e_2 = L.Convolution2D(128, 128, (5, 6), stride=(5, 3), initialW=w_init_H)
            # (N, 128, 10, 18)
            self.layers = list()
            for _ in range(6):
                self.layers.append(GeneratorBlock())
            # (N, 128, 10, 18)
            self.d_2 = L.Deconvolution2D(128, 128, (5, 6), stride=(5, 3), initialW=w_init_N, nobias=True)
            self.b_2 = L.BatchNormalization(128)
            # (N, 128, 50, 57)
            self.d_1 = L.Deconvolution2D(128, 64, (2, 3), stride=(2, 3), initialW=w_init_N, nobias=True)
            self.b_1 = L.BatchNormalization(64)
            # (N, 64, 100, 171)
            self.d_0 = L.Deconvolution2D(64, 1, (2, 3), stride=(2, 3), initialW=w_init_N)
            # (N, 2, 200, 513)
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
        _y = F.transpose(_x[0], (0, 3, 2, 1))
        _y = self.e_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.e_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.e_2(_y)
        _y = F.leaky_relu(_y)
        for l in self.layers:
            _y = l(_y)
        _y = self.d_2(_y)
        _y = self.b_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_1(_y)
        _y = self.b_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.d_0(_y)
        _y = F.transpose(_y, (0, 3, 2, 1))
        return F.tanh(_y)
    def to_gpu(self, device=None):
        super(Generator, self).to_gpu(device)
        for l in self.layers:
            l.to_gpu()
