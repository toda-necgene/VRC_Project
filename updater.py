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
        self.gen_en = model["encoder"]
        self.gen_de = model["decoder"]
        self.gen_ab1 = model["main"]
        self.gen_ba1 = model["inverse"]
        self.gen_ab2 = model["main2"]
        self.gen_ba2 = model["inverse2"]
        self.disa = model["disa"]
        self.disb = model["disb"]
        self.max_iteration = max_itr
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def update_core(self):
        gen_en_optimizer = self.get_optimizer("gen_en")
        gen_de_optimizer = self.get_optimizer("gen_de")
        gen_ab_optimizer1 = self.get_optimizer("gen_ab1")
        gen_ba_optimizer1 = self.get_optimizer("gen_ba1")
        gen_ab_optimizer2 = self.get_optimizer("gen_ab2")
        gen_ba_optimizer2 = self.get_optimizer("gen_ba2")
        disa_optimizer = self.get_optimizer("disa")
        disb_optimizer = self.get_optimizer("disb")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_1_p = self.gen_ab1(self.gen_en(batch_a))
        fake_ba_1_p = self.gen_ba1(self.gen_en(batch_b))
        fake_ab_1 = self.gen_de(fake_ab_1_p)
        fake_ba_1 = self.gen_de(fake_ba_1_p)
        fake_ab_2 = self.gen_de(self.gen_ab2(fake_ab_1_p))
        fake_ba_2 = self.gen_de(self.gen_ba2(fake_ba_1_p))
        # D_true update
        self.disa.cleargrads()
        self.disb.cleargrads()
        y_ta = self.disa(batch_a)
        y_tb = self.disb(batch_b)
        wave_length = y_ta.shape[2]
        y_label_o = _xp.ones([batch_size, 1, wave_length], dtype="float32")
        loss_d_t_a = F.mean_squared_error(y_ta, y_label_o)*0.5
        loss_d_t_b = F.mean_squared_error(y_tb, y_label_o)*0.5
        y_label_z = _xp.zeros([batch_size, 1, wave_length], dtype="float32")
        y_fa1 = self.disa(fake_ba_1)
        y_fb1 = self.disb(fake_ab_1)
        y_fa2 = self.disa(fake_ba_2)
        y_fb2 = self.disb(fake_ab_2)
        loss_d_f_a = F.mean_squared_error(y_fa1, y_label_z)*0.25+F.mean_squared_error(y_fa2, y_label_z)*0.25
        loss_d_f_b = F.mean_squared_error(y_fb1, y_label_z)*0.25+F.mean_squared_error(y_fb2, y_label_z)*0.25
        (loss_d_t_a + loss_d_t_b + loss_d_f_a + loss_d_f_b).backward()
        chainer.report({"loss": loss_d_f_a+loss_d_t_a}, self.disa)
        chainer.report({"loss": loss_d_f_b+loss_d_t_b}, self.disb)
        disa_optimizer.update()
        disb_optimizer.update()
        # G(en-de) update
        self.gen_en.cleargrads()
        self.gen_de.cleargrads()
        cyc_a = self.gen_de(self.gen_en(batch_a))
        cyc_b = self.gen_de(self.gen_en(batch_b))
        loss_cyc_aa = F.mean_squared_error(cyc_a, batch_a)*0.5
        loss_cyc_bb = F.mean_squared_error(cyc_b, batch_b)*0.5
        (loss_cyc_aa+loss_cyc_bb).backward()
        gen_en_optimizer.update()
        gen_de_optimizer.update()
        # G(layer1) update
        _lamda = 10.0 - self.iteration / self.max_iteration * 5.0
        self.gen_ba1.cleargrads()
        self.gen_ab1.cleargrads()
        self.gen_en.cleargrads()
        self.gen_de.cleargrads()
        fake_ab_1_p = self.gen_ab1(self.gen_en(batch_a))
        fake_ba_1_p = self.gen_ba1(self.gen_en(batch_b))
        fake_ab_1 = self.gen_de(fake_ab_1_p)
        fake_ba_1 = self.gen_de(fake_ba_1_p)
        fake_aba1 = self.gen_de(self.gen_ba1(fake_ab_1_p))
        fake_bab1 = self.gen_de(self.gen_ab1(fake_ba_1_p))
        y_fake_ba = self.disa(fake_ba_1)
        y_fake_ab = self.disb(fake_ab_1)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o)
        loss_cyca = F.mean_absolute_error(fake_aba1, batch_a)
        loss_cycb = F.mean_absolute_error(fake_bab1, batch_b)
        chainer.report({"GAN": loss_ganba, "CYC": loss_cyca+loss_cycb}, self.gen_ba1)
        chainer.report({"GAN": loss_ganab}, self.gen_ab1)
        gloss = (loss_cyca + loss_cycb) * _lamda + (loss_ganab + loss_ganba) * 0.5
        gloss.backward()
        gen_ba_optimizer1.update()
        gen_ab_optimizer1.update()
        gen_en_optimizer.update()
        gen_de_optimizer.update()
        # G(layer2) update
        self.gen_ba2.cleargrads()
        self.gen_ab2.cleargrads()
        self.gen_en.cleargrads()
        self.gen_de.cleargrads()
        fake_ab_1_p = self.gen_ab1(self.gen_en(batch_a))
        fake_ba_1_p = self.gen_ba1(self.gen_en(batch_b))
        fake_ab_2_p = self.gen_ab2(fake_ab_1_p)
        fake_ab_2 = self.gen_de(fake_ab_2_p)
        fake_aba2 = self.gen_de(self.gen_ba2(self.gen_ba1(fake_ab_2_p)))
        fake_ba_2_p = self.gen_ba2(fake_ba_1_p)
        fake_ba_2 = self.gen_de(fake_ba_2_p)
        fake_bab2 = self.gen_de(self.gen_ab2(self.gen_ab1(fake_ba_2_p)))
        y_fake_ba = self.disa(fake_ba_2)
        y_fake_ab = self.disb(fake_ab_2)
        loss_ganab = F.mean_squared_error(y_fake_ab, y_label_o)
        loss_ganba = F.mean_squared_error(y_fake_ba, y_label_o)
        loss_cyca = F.mean_absolute_error(fake_aba2, batch_a)
        loss_cycb = F.mean_absolute_error(fake_bab2, batch_b)
        chainer.report({"GAN": loss_ganba, "CYC": loss_cyca+loss_cycb}, self.gen_ba2)
        chainer.report({"GAN": loss_ganab}, self.gen_ab2)
        gloss = (loss_cyca + loss_cycb) * _lamda + (loss_ganab + loss_ganba) * 0.5
        gloss.backward()
        gen_ba_optimizer2.update()
        gen_ab_optimizer2.update()
        gen_en_optimizer.update()
        gen_de_optimizer.update()

