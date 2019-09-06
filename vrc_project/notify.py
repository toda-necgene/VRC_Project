"""
製作者:Toda
Line通知関連
"""
import subprocess
import datetime
import chainer

class LineNotify(chainer.training.Extension):
    """
    テストを行うExtention
    """
    def __init__(self, _trainer, key):
        self.key = key
        super(LineNotify, self).initialize(_trainer)
    def __call__(self, _trainer):
        per = _trainer.updater.iteration / _trainer.updater.max_iteration
        t = 1 - per
        est = _trainer.elapsed_time / per
        remain = est * t
        d = datetime.timedelta(seconds=remain)
        dd = "%d:%d:%d" % (remain//3600, remain//60%60, remain%60)
        estimated = datetime.datetime.now() + d
        ed = estimated.strftime("%dth %H:%M:%S")
        send_msg_img(self.key, "iteration:%d ETA: %s(%s)" % (_trainer.updater.iteration, dd, ed), "./latest.png")

def send_msg_img(key, msg, img_path):
    """
    画像とメッセージを送信
    """
    subprocess.getoutput('curl https://notify-api.line.me/api/notify -X POST -H "Authorization: Bearer %s" -F "message=%s" -F "imageFile=@%s"' % (key, msg, img_path))
