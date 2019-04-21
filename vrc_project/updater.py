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
        self.gen_ab1 = model["main"]
        self.gen_ba1 = model["inverse"]
        self.disa = model["disa"]
        self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_ab_optimizer1 = self.get_optimizer("gen_ab1")
        gen_ba_optimizer1 = self.get_optimizer("gen_ba1")
        disa_optimizer = self.get_optimizer("disa")
        disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_1 = self.gen_ab1(batch_a)
        fake_ba_1 = self.gen_ba1(batch_b)
        # D update
        for _ in range(2):
            self.disa.cleargrads()
            self.disb.cleargrads()
            y_at = self.disa(batch_a)
            y_bt = self.disb(batch_b)
            y_af = self.disa(fake_ba_1)
            y_bf = self.disb(fake_ab_1)
            wave_length = y_af.shape[2]
            y_label_o = _xp.ones([batch_size, 1, wave_length], dtype="float32")
            y_label_z = _xp.zeros([batch_size, 1, wave_length], dtype="float32")
            loss_d_a = F.mean_squared_error(y_af, y_label_z)
            loss_d_b = F.mean_squared_error(y_bf, y_label_z)
            loss_d_a.backward()
            loss_d_b.backward()
            disa_optimizer.update()
            disb_optimizer.update()
            self.disa.cleargrads()
            self.disb.cleargrads()
            loss_d_a = F.mean_squared_error(y_at, y_label_o)
            loss_d_b = F.mean_squared_error(y_bt, y_label_o)
            loss_d_a.backward()
            loss_d_b.backward()
            chainer.report({"loss": loss_d_a}, self.disa)
            chainer.report({"loss": loss_d_b}, self.disb)
            disa_optimizer.update()
            disb_optimizer.update()
        # G update
        _lamda = 50.0
        self.gen_ba1.cleargrads()
        self.gen_ab1.cleargrads()
        fake_ba_1 = self.gen_ba1(batch_b)
        y_fake_ba = self.disa(fake_ba_1)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o)
        fake_bab1 = self.gen_ab1(fake_ba_1)
        loss_cycb = F.mean_absolute_error(fake_bab1, batch_b)
        chainer.report({"GAN": loss_ganba, "CYC": loss_cycb}, self.gen_ba1)
        gloss = loss_cycb * _lamda + loss_ganba
        gloss.backward()
        gen_ab_optimizer1.update()
        gen_ba_optimizer1.update()
        self.gen_ba1.cleargrads()
        self.gen_ab1.cleargrads()
        fake_ab_1 = self.gen_ab1(batch_a)
        y_fake_ab = self.disb(fake_ab_1)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o)
        fake_aba1 = self.gen_ba1(fake_ab_1)
        loss_cyca = F.mean_absolute_error(fake_aba1, batch_a)
        gloss = loss_cyca * _lamda + loss_ganab
        chainer.report({"GAN": loss_ganab, "CYC": loss_cyca}, self.gen_ab1)
        gloss.backward()
        gen_ab_optimizer1.update()
        gen_ba_optimizer1.update()
        