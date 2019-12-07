"""
製作者:Toda
Line通知関連
"""
import subprocess
import datetime

def send_msg_img(key, msg, img_path):
    """
    画像とメッセージを送信
    """
    subprocess.getoutput('curl https://notify-api.line.me/api/notify -X POST -H "Authorization: Bearer %s" -F "message=%s" -F "imageFile=@%s"' % (key, msg, img_path))
def send_msg(key, msg):
    """
    画像とメッセージを送信
    """
    subprocess.getoutput('curl https://notify-api.line.me/api/notify -X POST -H "Authorization: Bearer %s" -F "message=%s"' % (key, msg))
