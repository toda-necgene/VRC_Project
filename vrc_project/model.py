
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
        出力の多様化を促し、Gとの収束スピードのバランスを保つ
    """
    def __init__(self):
        """
        モデル定義
        """
        super(Discriminator, self).__init__()
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
            self.c_3 = L.Convolution1D(chs[2], 512, 9, pad=4, initialW=he_init)
            # (N, 4, 7)
    def __call__(self, *_x, **kwargs):
        """
        Parameter
        ---------
        _x = { "type": "xp.ndarray"
                "dtype": "xp.float32"
                "shape": "[N,1025,200,1]"
                "range": "[-1.0, 1.0]"
                "description": "spectral_envelope"}
        Return
        ---------
        _y = { "type": "xp.ndarray"
                "dtype": "xp.float32"
                "shape": "[N,4,9]"
                "range": "[-inf, inf]"
                "description": "descriminated result"}
        """
        _y = F.transpose(_x[0][:, :, :, 0], (0, 2, 1))
        _y = self.c_0(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_1(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_2(_y)
        _y = F.leaky_relu(_y)
        _y = self.c_3(_y)
        _y = F.max(F.reshape(_y, (-1 ,128, 4, 9)), axis=1)
        return _y
class Generator(chainer.Chain):
    """
        学習用生成側ネットワーク
    """
    def __init__(self):
        """
        レイヤー定義
        e -> leaky_relu ->[ skipconnection-> cn1 -> leaky_relu -> cn2 -> leaky_relu -> add] x4 -> d
        """
        super(Generator, self).__init__()
        he_init = chainer.initializers.HeNormal()
        self.add_link("a_encode", L.Convolution2D(1025, 256, (4, 1), stride=(4, 1), initialW=he_init).add_hook(spn()))
        ch = [256, 256, 256, 256]
        self.shapes=[[256, 5, 10], [256, 10, 5], [256, 25, 2], [256, 50, 1]]
        for i in range(4):
            self.add_link("b_conv1d_{}_0".format(i),L.Convolution2D(ch[i], ch[i], (21, 1), initialW=he_init).add_hook(spn()))
            self.add_link("b_conv1d_{}_1".format(i),L.Convolution2D(ch[i], ch[i], (21, 1), initialW=he_init).add_hook(spn()))
        self.add_link("c_decode",L.Deconvolution2D(256, 1025, (4, 1), stride=(4, 1), initialW=he_init).add_hook(spn()))

    def __call__(self, *_x, **kwargs):
        """
            Parameter
            ---------
            x = { "type": "xp.ndarray"
                  "dtype": "xp.float32"
                  "shape": "[N,1025,200,1]"
                  "range": "[-inf, inf]"
                  "description": "spectral_envelope"}
            Return
            ---------
            _y = { "type": "xp.ndarray"
                  "dtype": "xp.float32"
                  "shape": "[N,1025,200,1]"
                  "range": "[-inf, inf]"
                  "description": "spectral_envelope"}
        """
        l = self.children()
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        _y = next(l)(_y)
        _y = F.leaky_relu(_y)
        _y = F.reshape(_y, (_y.shape[0], self.shapes[0][0], self.shapes[0][1], self.shapes[0][2]))
        # 1-3 layers
        for i in range(3):
            _h = F.pad(_y, pad_width=((0, 0), (0, 0), (20, 0), (0, 0)), mode="constant")
            _h = next(l)(_h)
            _h = F.leaky_relu(_h)
            _h = F.pad(_h, pad_width=((0, 0), (0, 0), (20, 0), (0, 0)), mode="constant") 
            _h = next(l)(_h)
            _y = F.leaky_relu(_h) + _y
            _y = F.reshape(_y, (_y.shape[0], self.shapes[i+1][0], self.shapes[i+1][1], self.shapes[i+1][2]))
        # 4 
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (20, 0), (0, 0)), mode="constant")
        _h = next(l)(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (20, 0), (0, 0)), mode="constant") 
        _h = next(l)(_h)
        _y = F.leaky_relu(_h) + _y
        _y = next(l)(_y)
        _y = F.transpose(_y, (0, 2, 1, 3))
        return _y

class GeneratorSimple(chainer.Chain):
    """
        network to execute
        use calcurated spectral norm for prediction.
        do not load data without model_simplify(model_simplifier.py). 

        実行用生成側ネットワーク
        spectral_norm を事前計算することによって高速化を図っている
    """
    def __init__(self):
        """
        レイヤー定義
        model_simplifier.pyのmodel_simplefyを **必ず実行** したデータを使ってくっださい
        ※ 実行しないでデータを読み込んでも動きますが推定される値はとても悪いです
        """
        super(GeneratorSimple, self).__init__()
        he_init = chainer.initializers.HeNormal()
        self.add_link("a_encode", L.Convolution2D(36, 64, (1, 1)))
        ch = [64, 128, 256, 256]
        self.shapes=[[64, 20, 10], [128, 20, 5], [256, 25, 2], [256, 50, 1]]
        for i in range(4):
            self.add_link("b_conv1d_{}_0".format(i),L.Convolution2D(ch[i], ch[i], (11, 1)))
            self.add_link("b_conv1d_{}_1".format(i),L.Convolution2D(ch[i], ch[i], (11, 1)))
        self.add_link("c_decode",L.Deconvolution2D(256, 36, (4, 1), stride=(4, 1)))

    def __call__(self, *_x, **kwargs):
        """
            Parameter
            ---------
            x = { "type": "xp.ndarray"
                  "dtype": "xp.float32"
                  "shape": "[N,1025,200,1]"
                  "range": "[-inf, inf]"
                  "description": "spectral_envelope"}
            Return
            ---------
            _y = { "type": "xp.ndarray"
                  "dtype": "xp.float32"
                  "shape": "[N,1025,200,1]"
                  "range": "[-inf, inf]"
                  "description": "spectral_envelope"}
        """
        l = self.children()
        _y = F.transpose(_x[0], (0, 2, 1, 3))
        _y = next(l)(_y)
        _y = F.leaky_relu(_y)
        _y = F.reshape(_y, (_y.shape[0], self.shapes[0][0], self.shapes[0][1], self.shapes[0][2]))
        # 1-3 layers
        for i in range(3):
            _h = F.pad(_y, pad_width=((0, 0), (0, 0), (10, 0), (0, 0)), mode="constant")
            _h = next(l)(_h)
            _h = F.leaky_relu(_h)
            _h = F.pad(_h, pad_width=((0, 0), (0, 0), (10, 0), (0, 0)), mode="constant") 
            _h = next(l)(_h)
            _y = F.leaky_relu(_h) + _y
            _y = F.reshape(_y, (_y.shape[0], self.shapes[i+1][0], self.shapes[i+1][1], self.shapes[i+1][2]))
        # 4 
        _h = F.pad(_y, pad_width=((0, 0), (0, 0), (10, 0), (0, 0)), mode="constant")
        _h = next(l)(_h)
        _h = F.leaky_relu(_h)
        _h = F.pad(_h, pad_width=((0, 0), (0, 0), (10, 0), (0, 0)), mode="constant") 
        _h = next(l)(_h)
        _y = F.leaky_relu(_h) + _y
        _y = next(l)(_y)
        _y = F.transpose(_y, (0, 2, 1, 3))
        return _y
