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
    def __init__(self, model, max_itr, f0_param, *args, cyc_lambda=5.0, **kwargs):
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
        _hat_f0_param = {
            "log_rate":float(1/f0_param["log_rate"]),
        }
        _f0_param = {
            "log_rate":float(f0_param["log_rate"]),
        }
        self.v_arg = [_f0_param, _hat_f0_param]
        self.cyc_lambda = cyc_lambda
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_ab_optimizer = self.get_optimizer("gen_ab")
        gen_ba_optimizer = self.get_optimizer("gen_ba")
        disa_optimizer = self.get_optimizer("disa")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        _xp = chainer.backend.get_array_module(batch_a.data)
        # D update
        self.disa.cleargrads()
        rate = 1.0 - (self.iteration / self.max_iteration)
        batch_an = batch_a[:, :, :-1, :]
        batch_bn = batch_b[:, :, :-1, :]
        batch_af = F.transpose(batch_a[:, :, -1, :], (0, 2, 1))
        batch_bf = F.transpose(batch_b[:, :, -1, :], (0, 2, 1))
        batch_ab_f = batch_af * self.v_arg[0]["log_rate"]
        batch_ba_f = batch_bf * self.v_arg[1]["log_rate"]
        batch_an = batch_an * (_xp.random.randn(batch_an.shape[0], 1, batch_an.shape[2], 1).astype(_xp.float32) * 0.002 * rate + _xp.ones([batch_an.shape[0], 1, 1, 1]))
        batch_bn = batch_bn * (_xp.random.randn(batch_bn.shape[0], 1, batch_bn.shape[2], 1).astype(_xp.float32) * 0.002 * rate + _xp.ones([batch_bn.shape[0], 1, 1, 1]))
        fake_ab = self.gen_ab(batch_an)
        fake_ba = self.gen_ba(batch_bn)
        y_af, y_af_f0 = self.disa(fake_ba)
        y_bf, y_bf_f0 = self.disa(fake_ab)
        y_at, y_at_f0 = self.disa(batch_an)
        y_bt, y_bt_f0 = self.disa(batch_bn)
        y_label_TA = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TA[:, 0] = 1.0
        y_label_TB = _xp.zeros(y_af.shape, dtype="float32")
        y_label_TB[:, 1] = 1.0
        y_label_FA = _xp.zeros(y_bf.shape, dtype="float32")
        y_label_FB = _xp.zeros(y_bf.shape, dtype="float32")
        loss_d_af = F.mean_squared_error(y_af, y_label_FA)* 0.9 + F.mean_squared_error(y_af_f0, batch_ba_f)* 0.1
        loss_d_bf = F.mean_squared_error(y_bf, y_label_FB)* 0.9 + F.mean_squared_error(y_bf_f0, batch_ab_f)* 0.1
        loss_d_ar = F.mean_squared_error(y_at, y_label_TA)* 0.9 + F.mean_squared_error(y_at_f0, batch_af)* 0.1
        loss_d_br = F.mean_squared_error(y_bt, y_label_TB)* 0.9 + F.mean_squared_error(y_bt_f0, batch_bf)* 0.1
        chainer.report({"D_A_REAL": loss_d_ar,
                        "D_A_FAKE": loss_d_af,
                        "D_B_REAL": loss_d_br,
                        "D_B_FAKE": loss_d_bf})
        (loss_d_af + loss_d_ar).backward()
        (loss_d_bf + loss_d_br).backward()
        disa_optimizer.update()
        # G update
        self.gen_ab.cleargrads()
        self.gen_ba.cleargrads()
        fake_ba = self.gen_ba(batch_bn)
        fake_ab = self.gen_ab(batch_an)
        y_fake_ba, y_fake_ba_f0 = self.disa(fake_ba)
        y_fake_ab, y_fake_ab_f0 = self.disa(fake_ab)
        fake_aba = self.gen_ba(fake_ab)
        fake_bab = self.gen_ab(fake_ba)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_TB) * 0.9 + F.mean_squared_error(y_fake_ab_f0, batch_ab_f) * 0.1
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_TA) * 0.9 + F.mean_squared_error(y_fake_ba_f0, batch_ba_f) * 0.1
        loss_cycb = F.sqrt(F.mean_squared_error(fake_bab, batch_bn))
        loss_cyca = F.sqrt(F.mean_squared_error(fake_aba, batch_an))
        gloss = loss_ganba + loss_ganab + (loss_cyca + loss_cycb) * self.cyc_lambda
        gloss.backward()
        chainer.report({"G_AB__GAN": loss_ganab,
                        "G_BA__GAN": loss_ganba,
                        "G_ABA_CYC": loss_cyca,
                        "G_BAB_CYC": loss_cycb})
        gen_ba_optimizer.update()
        gen_ab_optimizer.update()
