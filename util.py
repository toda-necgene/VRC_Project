import json


def config_reader(path, default={}):
    # reading json file
    try:
        with open(path, "r") as f:
            dd = json.load(f)
            keys = dd.keys()
            for j in keys:
                data = dd[j]
                keys2 = data.keys()
                for k in keys2:
                    if k in default:
                        if type(default[k]) == type(data[k]):
                            default[k] = data[k]
                        else:
                            print(
                                " [W] Argumet \"" + k +
                                "\" is incorrect data type. Please change to \""
                                + str(type(default[k])) + "\"")
                    elif k[0] == "#":
                        pass
                    else:
                        print(" [W] Argument \"" + k + "\" is not exsits.")

    except json.JSONDecodeError as e:
        print(" [W] JSONDecodeError: ", e)
        print(" [W] Use default setting")
        raise e
    except FileNotFoundError as e:
        print(" [W] Setting file is not found :", path)
        print(" [W] Use default setting")
        raise e
    except Exception as e:
        raise e

    return default


import numpy as np


def generate_f0_transfer(source_mean, target_mean=None, stdev_rate=None):
    """
	f0 変換関数を提供します

	Parameters
	----------
	source_mean : float or str
		typeがstrであれば、指定された文字列が[source_mean, target_mean, stdev_rate]が
		格納されたnpyファイルであるとみなします。
		また、以降の引数は無視されます

		typeがfloatであれば、A -> B変換のAの平均f0です
	target_mean : float
		Bの平均f0です
	stdev_rate : float
		A及び、Bのf0標準偏差の比です。(B_std / A_std)

	Returns
	-------
	transfer : function
		f0変換を提供する関数です
	

	"""
    if type(source_mean) is str:
        a = np.load(source_mean)
        source_mean = a[0]
        target_mean = a[1]
        stdev_rate = a[2]

    def transfer(f0):
        f0 = (f0 - source_mean) * stdev_rate + target_mean
        return f0

    return transfer

import wave
def isread(path):
    """
    WAVEファイルから各フレームをINT16配列にして返す

    Returns
    -------
    長さ160,000(10秒分)のdtype=np.int16のndarray

    Notes
    -----
    WAVEファイルのサンプリングレートは16[kHz]でなければならない。
    WAVEファイルが10秒未満の場合、配列の長さが10秒になるようにパディングする
    WAVEファイルが10秒を超える場合、超えた分を切り詰める
    """

    wf = wave.open(path, "rb")
    ans = np.zeros([1], dtype=np.int16)
    dds = wf.readframes(1024)
    while dds != b'':
        ans = np.append(ans, np.frombuffer(dds, "int16"))
        dds = wf.readframes(1024)
    wf.close()
    ans = ans[1:]
    i = 160000 - ans.shape[0]
    if i > 0:
        ans = np.pad(ans, (0, i), "constant")
    else:
        ans = ans[:160000]
    return ans


from pyworld import pyworld as pw
def encode(data, fs, frame_period=5.0, fft_size=None):
    """
    振幅データから音響特性を抽出する
    Parameters
    ----------
        data : np.ndarray(`T`, dtype=np.int16)
    Returns
    -------
        f0 : ndarray(`T // (fs // 1000 * frame_period_ms) + 1`, dtype=np.float64)
        psp : ndarray([`T // (fs // 1000 * frame_period_ms) + 1`, fftl(1024) / 2 + 1], dtype=np.float64)
        ap : ndarray([`T // (fs // 1000 * frame_period_ms)`, fftl(1024) / 2 + 1], dtype=np.float64)
            # 非周期性指標の抽出
    
    Notes
    -----
        frame_period_msはデフォルト5.0[msec]であるため、
        fsはデフォルトで16000[Hz]であるため、
        f0のサイズは[ T // 80 + 1 ] となる
    """
    _f0, t = pw.dio(data, fs, frame_period=frame_period)
    f0 = pw.stonemask(data, _f0, t, fs)
    sp = pw.cheaptrick(data, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(data, f0, t, fs, fft_size=fft_size)
    psp = np.clip((np.log(sp) + 15) / 20, -1.0, 1.0)
    return f0.astype(np.float64), psp.astype(np.float64), ap.astype(np.float64)

def decode(fs, f0, psp, ap):
    fs = 16000
    ap = ap.reshape(-1, 513).astype(np.float)
    f0 = f0.reshape(-1).astype(np.float)
    sp = np.exp(psp.reshape(-1, 1, 513).astype(np.float) * 20 - 15)
    sp = sp.reshape(-1, 513).astype(np.float)
    return pw.synthesize(f0, sp, ap, fs)


def fft(data):
    time_ruler = data.shape[0] // 512
    if data.shape[0] % 512 == 0:
        time_ruler -= 1
    window = np.hamming(1024)
    pos = 0
    wined = np.zeros([time_ruler, 1024])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + 1024]
        r = 1024 - frame.shape[0]
        if r > 0:
            frame = np.pad(frame, (0, r), "constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    re = fft_r.real.reshape(time_ruler, -1)
    im = fft_r.imag.reshape(time_ruler, -1)
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(
        time_ruler, -1)[:, 512:]
    return np.clip(c, -15.0, 5.0)
