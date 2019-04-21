"""
製作者:TODA

学習データにランダムなノイズを付与する。
"""
import random
import chainer
class Noisy_dataset(chainer.dataset.DatasetMixin):
    """
    学習データにランダムなノイズを付与する。
    Attributes
    ----------
    _data: 学習データ
    _rate: ノイズの分散
    _xp: バックエンドのモジュール
    """
    def __init__(self, data, rate=0.01):
        self._data = data
        self._rate = rate
        self._xp = chainer.backend.get_array_module(data[0])
    def __len__(self):
        return len(self._data)
    def __getitem__(self, index):
        _tmp_data = self._data[index]
        noise_shape = [_tmp_data.shape[0], 1, 1]
        if random.random() < 0.5:
            _tmp_data += self._xp.random.randn(noise_shape[0], noise_shape[1], noise_shape[2]) * self._rate
            _tmp_data = self._xp.clip(_tmp_data, -1.0, 1.0)
        return _tmp_data.astype(self._xp.float32)
    def get_example(self, i):
        return self._data[i]
