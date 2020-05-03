import os

from abc import abstractmethod, ABC
from typing import Sequence, overload, Iterable, Tuple


class _LabIterable(tuple):

    def __new__(cls, iterable):
        try:
            for t in iterable:
                if not isinstance(t, tuple) or \
                        not isinstance(t[0], float) or \
                        not isinstance(t[1], float) or \
                        not isinstance(t[1], str):
                    raise ValueError("Not valid values")
            _last = 0
            for t in iterable:
                if t[0] > t[1] or t[0] < _last:
                    raise ValueError("Time error")
                _last = t[0]
        except IndexError:
            raise ValueError("Not valid values")

        return super().__new__(tuple, iterable)

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> Sequence[Tuple[float, float, str]]:
        return tuple(self[i] for i in range(s.start, s.stop, s.step))

    def __getitem__(self, i: int) -> Tuple[float, float, str]:
        return super().__getitem__(i)


class Format(ABC):

    @abstractmethod
    def encode(self, iterable: Iterable) -> str:
        pass

    @abstractmethod
    def decode(self, string: str) -> Iterable:
        pass


class _LabFormat(Format):

    def encode(self, iterable: Iterable[Tuple[float, float, str]]) -> str:
        _result = ""
        for sequence in iter(iterable):
            for start, stop, chord in sequence:
                _result += '%s %s %s\n' % (start, stop, chord)

        return _result[:-1]  # remove last newline character

    def decode(self, string: str) -> Iterable[Tuple[float, float, str]]:
        _lines = string.split('\n')
        _iterable = _LabIterable(())
        for line in _lines:
            _iterable += _LabIterable(line.split(""))
        return _iterable


# TODO register_formatter()

def get_formatter(path: str) -> Format:
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.lab':
        return _LabFormat()
    raise NotImplementedError("Not supported file format")
