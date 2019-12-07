"""
製作者:TODA
"""
import os
import shutil
import wave
import itertools
import torch
import torch.nn
import numpy as np
from tqdm import trange
from vrc_project.model import Discriminator, Generator
from vrc_project.seq_dataset import SeqData
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.eval import TestModel
from vrc_project.notify  import send_msg_img


def load_wave_file(_path_to_file):
    """
    Parameters
    ----------
    _path_to_file: str
    Returns
    -------
    _data: int16
    """
    wave_data = wave.open(_path_to_file, "rb")
    _data = np.zeros([1], dtype=np.int16)
    dds = wave_data.readframes(1024)
    while dds != b'':
        _data = np.append(_data, np.frombuffer(dds, "int16"))
        dds = wave_data.readframes(1024)
    wave_data.close()
    _data = _data[1:]
    return _data

def w_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def load_model(_checkpoint_dir: str,
                g_ab: Generator,
                g_ba: Generator,
                fis: Discriminator) -> bool:
    """
    モデルのロード
    Parameters
    ----------
    _checkpoint_dir: str
    保存済みモデルの格納ディレクトリ
    _trainer: chainer.training.trainer
    読み込み先のトレーナー
    Returns
    -------
    flag: bool
    読み込み完了フラグ
    """
    print(" [*] Reading checkpoint...")
    if os.path.exists(_checkpoint_dir) and os.path.exists(_checkpoint_dir+"/gen_ab.pth"):
        print(" [I] checkpoint is found. loading file name : %s " % (_checkpoint_dir))
        g_ab.load_state_dict(torch.load(_checkpoint_dir+"/gen_ab.pth"))
        g_ba.load_state_dict(torch.load(_checkpoint_dir+"/gen_ba.pth"))
        dis.load_state_dict(torch.load(_checkpoint_dir+"/dis.pth"))
        print(" [I] loaded checkpoint successfully.")
        return True
    else:
        g_ab.apply(w_init)
        g_ba.apply(w_init)
        dis.apply(w_init)
        if not os.path.exists(_checkpoint_dir):
            os.makedirs(_checkpoint_dir)
    print(" [I] checkpoint is not found.")
    return False

def dataset_pre_process_controler(args):
    """
    Parameters
    ----------
    _args: dict
    Returns
    -------
    _train_iter_a: chainer.iterators.Iterator
    _train_iter_b: chainer.iterators.Iterator
    _voice_profile: dict (float64)
        f0 parameters.
        keys: (pre_sub, pitch_rate, postad)
    _length_sp: int
    """
    _sounds_a = None
    _sounds_b = None
    if not (args["use_old_dataset"] and os.path.exists("./dataset/patch/A.npy") and os.path.exists("./dataset/patch/B.npy")):
        _sounds_a, _sounds_b = create_dataset(args["input_size"])
    else:
        # preparing training-data
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
    _length_sp = 200
    # f0 parameters(基本周波数F0の変換に使用する定数。詳しくは./vrc_project/voice_to_dataset_cycle.py L65周辺)
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _sounds_a, _sounds_b, _voice_profile, _length_sp
if __name__ == '__main__':
    # NOTE : Setting
    TERMS = 10
    TERM_ITERATION = 1000
    batch_size = [  32,   16,     8,     8,     1,    1,    1,    1,    1,    1]
    noise_rate = [0.02, 0.01, 0.005, 0.002, 0.002,  0.0,  0.0,  0.0,  0.0,  0.0]
    alpha_rate = [2e-4, 2e-4,  2e-4,  2e-4,  2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1e-5]
    lambda_cyc = 5.0
    # Setting end
    _args = load_setting_from_json("setting.json")
    test = load_wave_file("./dataset/test/test.wav") / 32767.0
    _label_sample = load_wave_file("./dataset/test/label.wav") / 32767.0    
    if  _args["wave_otp_dir"] is not "False":
        _args["wave_otp_dir"] = _args["wave_otp_dir"] + _args["model_name"] +  _args["version"]+"/"
        if not os.path.exists(_args["wave_otp_dir"]):
            os.makedirs(_args["wave_otp_dir"])
    sounds_a, sounds_b, voice_profile, length_sp = dataset_pre_process_controler(_args)
    g_a_to_b = Generator()
    g_b_to_a = Generator()
    dis = Discriminator()
    g_loss_gan = torch.nn.MSELoss()
    g_loss_cyc = torch.nn.L1Loss()
    d_loss_gan = torch.nn.MSELoss()
    # to_gpu
    if _args["gpu"] >= 0:
        d_loss_gan.cuda()
        g_loss_gan.cuda()
        g_loss_cyc.cuda()
        g_a_to_b.cuda()
        g_b_to_a.cuda()
        dis.cuda()
        Tensor = torch.cuda.FloatTensor if _args["gpu"] >= 0 else torch.Tensor
    load_model(_args["name_save"], g_a_to_b, g_b_to_a, dis)
    if _args["line_notify"]:
        with open("line_api_token.txt", "rb") as s:
                key = s.readline().decode("utf8")
    for i in trange(TERMS, leave=False):
        # (re)define optimizer and initialize
        d_optimizer = torch.optim.Adam(dis.parameters(), lr=alpha_rate[i], betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(itertools.chain(g_a_to_b.parameters(),g_b_to_a.parameters()), lr=alpha_rate[i], betas=(0.5, 0.999))
        # iterator reset
        _train_iter_a = torch.utils.data.DataLoader(SeqData(sounds_a, 200), batch_size=batch_size[i], shuffle=True)
        _train_iter_b = torch.utils.data.DataLoader(SeqData(sounds_b, 200), batch_size=batch_size[i], shuffle=True)
        # main training
        test_obj = None
        if _args["test"]:
            _args["length_sp"] = length_sp
            test_obj = TestModel(g_a_to_b, _args, [test, _label_sample, voice_profile], name_ad=str(i)+"_")
        print("\riterations\tloss_d\tloss_g\ttest_loss")
        for j in trange(TERM_ITERATION):
            batch_a = torch.autograd.Variable(next(iter(_train_iter_a)).type(Tensor))
            batch_b = torch.autograd.Variable(next(iter(_train_iter_b)).type(Tensor))
            batch_a = batch_a.permute([0, 2, 1, 3])
            batch_b = batch_b.permute([0, 2, 1, 3])
            batch_an = batch_a * (torch.randn(batch[i], batch_a.shape[1], 1, 1) * noise_rate[i] + 1)
            batch_bn = batch_b * (torch.randn(batch[i], batch_b.shape[1], 1, 1) * noise_rate[i] + 1)
            # D update
            dis.train()
            d_optimizer.zero_grad()
            fake_ab = g_a_to_b(batch_an)
            fake_ba = g_b_to_a(batch_bn)
            y_af = dis(fake_ba)
            y_bf = dis(fake_ab)
            y_at = dis(batch_an)
            y_bt = dis(batch_bn)
            # labels
            eye = np.eye(4)
            y_label_TA = np.repeat(np.repeat(eye[0].reshape(1, 4, 1), batch_size[i], axis=0), 9, axis=2)
            y_label_TA = Tensor(y_label_TA)
            y_label_TB = np.repeat(np.repeat(eye[1].reshape(1, 4, 1), batch_size[i], axis=0), 9, axis=2)
            y_label_TB = Tensor(y_label_TB)
            y_label_FA = np.repeat(np.repeat(eye[2].reshape(1, 4, 1), batch_size[i], axis=0), 9, axis=2)
            y_label_FA = Tensor(y_label_FA)
            y_label_FB = np.repeat(np.repeat(eye[3].reshape(1, 4, 1), batch_size[i], axis=0), 9, axis=2)
            y_label_FB = Tensor(y_label_FB)
            loss_d_af = d_loss_gan(y_af, y_label_FA) * 0.5
            loss_d_bf = d_loss_gan(y_bf, y_label_FB) * 0.5
            loss_d_ar = d_loss_gan(y_at, y_label_TA) * 0.5
            loss_d_br = d_loss_gan(y_bt, y_label_TB) * 0.5
            d_loss = (loss_d_af + loss_d_ar + loss_d_bf + loss_d_br)
            d_loss.backward()
            d_optimizer.step()
            # G update
            g_a_to_b.train()
            g_b_to_a.train()
            g_optimizer.zero_grad()
            fake_ba = g_b_to_a(batch_bn)
            fake_ab = g_a_to_b(batch_an)
            y_fake_ba = dis(fake_ba)
            y_fake_ab = dis(fake_ab)
            fake_aba = g_b_to_a(fake_ab)
            fake_bab = g_a_to_b(fake_ba)
            loss_ganab = g_loss_gan(y_fake_ab, y_label_TB) * 0.5
            loss_ganba = g_loss_gan(y_fake_ba, y_label_TA) * 0.5
            loss_cycb = g_loss_cyc(fake_bab, batch_b)
            loss_cyca = g_loss_cyc(fake_aba, batch_a)
            gloss = loss_ganba + loss_ganab + (loss_cyca + loss_cycb) * lambda_cyc
            gloss.backward()
            g_optimizer.step()
            if j % 100 == 99:
                if test_obj is not None:
                    test_loss = test_obj(j)
                    print("\r{}\t{}\t{}\t{}".format(j, d_loss, gloss, test_loss))
                else:
                    print("\r{}\t{}\t{}\t{}".format(j, d_loss_gan, g_loss_gan, g_loss_cyc))
        torch.save(g_a_to_b.state_dict(), _args["name_save"]+"g_ab.pth")
        torch.save(g_b_to_a.state_dict(), _args["name_save"]+"g_ba.pth")
        torch.save(dis.state_dict(), _args["name_save"]+"dis.pth")
        if test_obj is not None:
            send_msg_img(key, "{} Finished. score:{}".format(i+1, test_loss), "latest.png")
            print(" [*] {} terms finished. score:{}".format(i+1, test_loss))
        else:
            print(" [*] {} terms finished.".format(i+1))
    print("[*] all_finish")
    