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
    def __init__(self, _models, *args, **kwargs):
        """
        初期化関数
        Parameters
        ----------
        _models: list
        生成モデルA,生成モデルB,識別モデル
            len: 3
        """
        self.gen_ab, self.gen_ba, self.dis = _models
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
    def g_loss(self, gen, y_fake, y_label, sepc_fake_fake, sepc_source, _xp):
        """
        生成側の目的関数
        Parametars
        ----------
        gen:生成モデル
        y_fake:変換後の識別結果
        spec_fake_fake:変換結果を変換したもの
        spec_source:変換元
        xp:cupy or numpy
        Returns
        -------
        loss:損失
        """
        loss_gan = F.mean_squared_error(y_fake, y_label)
        loss_l1 = F.mean_absolute_error(sepc_fake_fake, sepc_source)
        loss = loss_gan+loss_l1*(4.5*_xp.cos(self.iteration/500*_xp.pi)+5.5).astype(_xp.float32)
        chainer.report({"loss": loss}, gen)
        return loss
    def d_loss(self, dis, y_batch, y_label):
        """
        識別側の目的関数
        Parameters
        ----------
        dis:識別モデル
        y_batch:識別結果
        y_label:目標値
        Returns
        -------
        loss:損失
        """
        loss = F.mean_squared_error(y_batch, y_label)
        chainer.report({"loss": loss}, dis)
        return loss
    def update_core(self):
        gen_ab_optimizer = self.get_optimizer("gen_ab")
        gen_ba_optimizer = self.get_optimizer("gen_ba")
        dis_optimizer = self.get_optimizer("dis")
        batch_a = chainer.Variable(self.converter(self.get_iterator("main").next()))
        batch_b = chainer.Variable(self.converter(self.get_iterator("data_b").next()))
        batch_size = len(batch_a)
        _xp = chainer.backend.get_array_module(batch_a.data)
        fake_ab_ = self.gen_ab(batch_a)
        fake_aba = self.gen_ba(fake_ab_)
        fake_ba_ = self.gen_ba(batch_b)
        fake_bab = self.gen_ab(fake_ba_)
        x_batch = F.concat((fake_ab_, fake_ba_, batch_a, batch_b), axis=0)
        label_a = _xp.zeros([batch_size, 3, batch_a.shape[2]])
        label_a[:, 0, :] = 1.0
        label_a = label_a.astype(_xp.float32)
        label_b = _xp.zeros([batch_size, 3, batch_a.shape[2]]).astype(_xp.float32)
        label_b[:, 1, :] = 1.0
        label_b = label_a.astype(_xp.float32)
        label_f = _xp.zeros([batch_size, 3, batch_a.shape[2]]).astype(_xp.float32)
        label_f[:, 2, :] = 1.0
        label_f = label_a.astype(_xp.float32)
        y_label = F.concat((label_f, label_f, label_a, label_b), axis=0)
        y_batch = self.dis(x_batch)
        y_fake_ab = y_batch[:batch_size]
        y_fake_ba = y_batch[batch_size:batch_size*2]
        gen_ab_optimizer.update(self.g_loss, self.gen_ab, y_fake_ab, label_a, fake_aba, batch_a, _xp)
        gen_ba_optimizer.update(self.g_loss, self.gen_ba, y_fake_ba, label_b, fake_bab, batch_b, _xp)
        dis_optimizer.update(self.d_loss, self.dis, y_batch, y_label)
