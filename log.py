import sys
from datetime import datetime

def _write_message(level, str):
    tag = "VEWDI"
    sys.stderr.write(" ".join([
        tag[level],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "[CycleGAN FACTORY]",
        str]) + '\n')

def v(str): _write_message(0, str)
def e(str): _write_message(1, str)
def w(str): _write_message(2, str)
def d(str): _write_message(3, str)
def i(str): _write_message(4, str)
