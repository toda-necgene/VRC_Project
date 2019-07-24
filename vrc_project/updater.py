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
        # disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        _xp = chainer.backend.get_array_module(batch_a.data)
        noise_rate = 0.001
        batch_a_n = noise_put(_xp, batch_a, noise_rate * _xp.exp(-self.iteration / self.max_iteration*30).astype(_xp.float32))
        batch_b_n = noise_put(_xp, batch_b, noise_rate * _xp.exp(-self.iteration / self.max_iteration*30).astype(_xp.float32))
        # D update
        self.disa.cleargrads()
        # self.disb.cleargrads()
        fake_ab = self.gen_ab(batch_a_n)[0]
        fake_ba = self.gen_ba(batch_b_n)[0]
        y_af = self.disa(fake_ba)
        y_bf = self.disa(fake_ab)
        y_at = self.disa(batch_a_n)
        y_bt = self.disa(batch_b_n)
        y_label_TA = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TA[:, 0] = 1.0
        y_label_TB = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TB[:, 1] = 1.0
        y_label_FA = _xp.zeros(y_af.shape, dtype="float32")
        y_label_FA[:, 2] = 1.0
        y_label_FB = _xp.zeros(y_af.shape, dtype="float32")
        y_label_FB[:, 3] = 1.0
        loss_d_af = F.mean_squared_error(y_af, y_label_FA)
        loss_d_bf = F.mean_squared_error(y_bf, y_label_FB)
        loss_d_ar = F.mean_squared_error(y_at, y_label_TA)
        loss_d_br = F.mean_squared_error(y_bt, y_label_TB)
        chainer.report({"D_A_REAL": loss_d_ar,
                        "D_A_FAKE": loss_d_af,
                        "D_B_REAL": loss_d_br,
                        "D_B_FAKE": loss_d_bf})
        (loss_d_af + loss_d_ar + loss_d_bf + loss_d_br).backward()
        disa_optimizer.update()
        # disb_optimizer.update()
        # G update
        self.gen_ab.cleargrads()
        self.gen_ba.cleargrads()
        fake_ba, through_out_ba = self.gen_ba(batch_b_n)
        fake_ab, through_out_ab = self.gen_ab(batch_a_n)
        fake_aba, _ = self.gen_ba(fake_ab)
        fake_bab, _ = self.gen_ab(fake_ba)
        y_fake_ba = self.disa(fake_ba)
        y_fake_ab = self.disa(fake_ab)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_TB)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_TA)
        bf_low = F.average_pooling_2d(F.transpose(fake_ba, (0, 3, 2, 1)), (50, 64), stride=(10, 64))
        br_low = F.average_pooling_2d(F.transpose(batch_b_n, (0, 3, 2, 1)), (50, 64), stride=(10, 64))
        af_low = F.average_pooling_2d(F.transpose(fake_ab, (0, 3, 2, 1)), (50, 64), stride=(10, 64))
        ar_low = F.average_pooling_2d(F.transpose(batch_a_n, (0, 3, 2, 1)), (50, 64), stride=(10, 64))
        loss_po_a = F.mean_absolute_error(bf_low, br_low)
        loss_po_b = F.mean_absolute_error(af_low, ar_low)
        loss_th_a = F.mean_absolute_error(through_out_ab, batch_a_n)
        loss_th_b = F.mean_absolute_error(through_out_ba, batch_b_n)
        loss_cycb = F.mean_absolute_error(fake_bab, batch_b_n)
        loss_cyca = F.mean_absolute_error(fake_aba, batch_a_n)
        gloss = (loss_ganba + loss_ganab) * 1 +\
                (loss_po_a + loss_po_b) * 1 +\
                (loss_cyca + loss_cycb) * 2.5 +\
                (loss_th_a + loss_th_b) * 2.5
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
        noise_shape = [x.shape[0], x.shape[1], x.shape[2], 1]
        x_s = x + F.relu(_xp.random.randn(noise_shape[0], noise_shape[1], noise_shape[2], noise_shape[3]).astype(_xp.float32) * stddev)
        return x_s
    else:
        return x
