import chainer
class SeqData(chainer.dataset.DatasetMixin):
    def __init__(self, data, size):
        self.__data = data
        self.__size = size
        self._xp = chainer.backend.get_array_module(data)
    def __len__(self):
        return self.__data.shape[0] // (self.__size//2)
    def get_example(self, i):
        p = self.__size // 2
        st = int(i * p + self._xp.random.randint(-p//2, p//2, 1))
        if st < 0:
            _r = self.__data[0:st+self.__size]
            pa = int(self.__size-_r.shape[0])
            _r = self._xp.pad(_r, ((pa, 0), (0, 0), (0, 0)), "constant", constant_values=-1)
        else:
            _r = self.__data[st:st+self.__size]
            if _r.shape[0] < self.__size:
                pa = int(self.__size-_r.shape[0])
                _r = self._xp.pad(_r, ((0, pa), (0, 0), (0, 0)), "constant", constant_values=-1)
        return _r
