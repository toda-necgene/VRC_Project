"""
    製作者:TODA
    学習の際に変数を更新するために必要な関数群
"""
import chainer
import chainer.functions as F
class CycleGANUpdater(chainer.training.updaters.StandardUpdater):
    """
        CycleGANのためのUpdater
        以下のものを含む
        - 目的函数
        - 更新のコア関数
    """
    def __init__(self, model, max_itr, *args, **kwargs):
        """
        初期化関数
        Parameters
        ----------
        model: dict
        生成モデルA,生成モデルB,識別モデルA,識別モデルB
        maxitr: int
        最大イテレーション回数
        """
        self.gen_ab = model["main"]
        self.gen_ba = model["inverse"]
        self.disa = model["disa"]
        # self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_ab_optimizer = self.get_optimizer("gen_ab1")
        gen_ba_optimizer = self.get_optimizer("gen_ba1")
        disa_optimizer = self.get_optimizer("disa")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        _xp = chainer.backend.get_array_module(batch_a.data)
        noise_rate = 0.001
        batch_a_n = noise_put(_xp, batch_a, noise_rate)
        batch_b_n = noise_put(_xp, batch_b, noise_rate)
        # D update
        self.disa.cleargrads()
        fake_ab = self.gen_ab(batch_a_n)
        fake_ba = self.gen_ba(batch_b_n)
        y_af = self.disa(fake_ba)
        y_bf = self.disa(fake_ab)
        y_at = self.disa(batch_a_n)
        y_bt = self.disa(batch_b_n)
        y_label_A = _xp.zeros(y_af.shape, dtype="float32")
        y_label_A[:, 0, :] = 1.0
        y_label_B = _xp.zeros(y_af.shape, dtype="float32")
        y_label_B[:, 1, :] = 1.0
        y_label_O = _xp.zeros(y_af.shape, dtype="float32")
        y_label_O[:, 2, :] = 1.0
        loss_d_af = F.mean_squared_error(y_af, y_label_O) * 0.5
        loss_d_bf = F.mean_squared_error(y_bf, y_label_O) * 0.5
        loss_d_ar = F.mean_squared_error(y_at, y_label_A) * 0.5
        loss_d_br = F.mean_squared_error(y_bt, y_label_B) * 0.5
        chainer.report({"D_A_REAL": loss_d_ar,
                        "D_A_FAKE": loss_d_af,
                        "D_B_REAL": loss_d_br,
                        "D_B_FAKE": loss_d_bf})
        (loss_d_af + loss_d_ar + loss_d_bf + loss_d_br).backward()
        disa_optimizer.update()
        # G update
        _lambda = 5.0
        self.gen_ab.cleargrads()
        self.gen_ba.cleargrads()
        fake_ba = self.gen_ba(batch_b_n)
        fake_ab = self.gen_ab(batch_a_n)
        fake_bab = self.gen_ab(fake_ba)
        fake_aba = self.gen_ba(fake_ab)
        y_fake_ba = self.disa(fake_ba)
        y_fake_ab = self.disa(fake_ab)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_B) * 0.5
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_A) * 0.5
        loss_cycb = F.mean_absolute_error(fake_bab, batch_b)
        loss_cyca = F.mean_absolute_error(fake_aba, batch_a)
        gloss = (loss_cyca + loss_cycb) * _lambda + loss_ganba + loss_ganab
        gloss.backward()
        chainer.report({"G_AB__GAN": loss_ganab,
                        "G_BA__GAN": loss_ganba,
                        "G_ABA_L1N": loss_cyca,
                        "G_BAB_L1N": loss_cycb})
        gen_ba_optimizer.update()
        gen_ab_optimizer.update()

def noise_put(_xp, x, stddev):
    """
    正規分布に沿ったノイズを載せる
    Parameter
    ---------
    _xp: 計算ライブラリ
    x: 入力テンソル（４階）
    stddev: 標準偏差
    """
    if stddev != 0:
        noise_shape = [x.shape[0], 1, x.shape[2], 1]
        x_s = x +_xp.random.randn(noise_shape[0], noise_shape[1], noise_shape[2], noise_shape[3]) * stddev
        return x_s
    else:
        return x
