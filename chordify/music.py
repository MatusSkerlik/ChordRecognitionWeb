from abc import abstractmethod, ABCMeta, ABC
from collections import deque
from enum import Enum
from functools import cached_property
from itertools import cycle, product, accumulate, chain
from math import log
from operator import mul
from typing import Tuple, List, Iterable, Collection, Sized, Sequence, Union

from .exceptions import IllegalStateError, IllegalArgumentError


def _rotate_right(vector: 'Vector', r: int) -> 'Vector':
    _vector = deque(vector)
    _vector.rotate(r)
    return Vector(_vector)


def _frequency(pitch: int) -> float:
    if pitch < 0:
        raise IllegalArgumentError
    if pitch > 127:
        raise IllegalArgumentError
    return pow(2, (pitch - 69) / 12) * 440


def _harmonics(start: 'ChordKey', length=8) -> Tuple['ChordKey']:
    fq = list(reversed(tuple(i * start.frequency() for i in range(1, length + 1))))

    chk_seq: List['ChordKey'] = list()

    while len(fq) > 0:
        subject = fq.pop()

        before: ChordKey = start
        for i, k in zip(range(12 * (int(log(length, 2)) + 1)),
                        cycle(k for k in iter(ChordKey))):
            if _frequency(i) < subject:
                before = k
            else:
                d_bs = subject - _frequency(i - 1 if i - 1 >= 0 else 0)
                d_cs = _frequency(i) - subject

                if d_bs < d_cs:
                    chk_seq.append(before)
                else:
                    chk_seq.append(k)
                break

    return tuple(chk_seq)


def _harm_to_vector(harms: Collection) -> 'Vector':
    _vector = ZeroVector()

    for key in harms:
        _vector[key.pos()] += 1

    return _vector + ZeroVector()


class ChordType(Enum):
    MAJOR = ""
    MINOR = ":min"
    AUGMENTED = ":aug"
    DIMINISHED = ":dim"

    def __str__(self):
        return '%s' % self.value


class ChordKey(Enum):
    C = "C"
    Cs = "C#"
    D = "D"
    Ds = "D#"
    E = "E"
    F = "F"
    Fs = "F#"
    G = "G"
    Gs = "G#"
    A = "A"
    As = "A#"
    B = "B"

    def __str__(self):
        return '%s' % self.value

    def frequency(self):
        for i, k in enumerate(self.__class__.__iter__()):
            if k == self:
                return _frequency(i)
        raise IllegalStateError

    def pos(self):
        for i, k in enumerate(self.__class__.__iter__()):
            if k == self:
                return i
        raise IllegalStateError


class Vector(Sized, Iterable):
    _vector: Tuple[float, ...]

    def __init__(self, vector: Sequence[float]) -> None:
        super().__init__()

        self._vector = tuple(vector)

    def __iter__(self):
        return iter(self._vector)

    def __len__(self):
        return len(self._vector)

    def __getitem__(self, item):
        return self._vector[item]

    def __sub__(self, other):
        if isinstance(other, Vector):
            _r = tuple(a - b for a, b in zip(self._vector, other._vector))
            m = max(_r)
            return Vector(tuple(a / m for a in _r))
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Vector):
            _r = tuple(a + b for a, b in zip(self._vector, other._vector))
            m = max(_r)
            return Vector(tuple(a / m for a in _r))
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Vector):
            return sum(a * b for a, b in zip(self._vector, other._vector))
        raise NotImplementedError

    def __repr__(self):
        return '(' + ', '.join(map(str, self._vector)) + ')'


class MutableVector(Vector):

    def __setitem__(self, key, value):
        _vector = list(self._vector)
        _vector[key] = float(value)
        self._vector = tuple(_vector)


class ZeroVector(MutableVector):

    def __init__(self) -> None:
        super().__init__(tuple(0.0 for i in range(12)))


class IChord(object):
    _chord_key: ChordKey
    _chord_type: Union[ChordType, None]

    def __init__(self, chord_key: Union[ChordKey, None] = None, chord_type: Union[ChordType, None] = None):
        super().__init__()

        if chord_key is None and chord_type is None:
            self._chord_key = ChordKey("N")
            self._chord_type = None
        else:
            self._chord_type = chord_type
            self._chord_key = chord_key

    def __eq__(self, other):
        if isinstance(other, IChord):
            return self._chord_key == other._chord_key and self._chord_type == other._chord_type
        raise NotImplementedError

    def __ne__(self, other):
        if isinstance(other, IChord):
            return self._chord_key != other._chord_key or self._chord_type != other._chord_type
        raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, IChord):
            return self.__repr__() < other.__repr__()
        raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, IChord):
            return self.__repr__() > other.__repr__()
        raise NotImplementedError

    def __hash__(self):
        return hash((self._chord_key, self._chord_type))

    def __repr__(self) -> str:
        try:
            return '%s%s' % (self._chord_key.value, self._chord_type.value)
        except AttributeError:
            return '%s' % self._chord_key.value


class Chord(IChord, metaclass=ABCMeta):

    def __len__(self):
        return 12

    def __iter__(self):
        return iter(self.vector)

    def __getitem__(self, item):
        return self.vector[item]

    @property
    def key(self):
        return self._chord_key

    @property
    def type(self):
        return self._chord_type

    @cached_property
    @abstractmethod
    def vector(self) -> Vector:
        pass


class TemplateChord(Chord):
    _MAJOR: Vector = (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)
    _MINOR: Vector = (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    _DIMINISHED: Vector = (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0)
    _AUGMENTED: Vector = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

    def shift(self):
        for i, key in enumerate(iter(ChordKey)):
            if key == self._chord_key:
                return i
        raise IllegalStateError

    @cached_property
    def vector(self) -> Vector:

        _shift = self.shift()

        if self.type == ChordType.MAJOR:
            return _rotate_right(self._MAJOR, _shift)
        if self.type == ChordType.MINOR:
            return _rotate_right(self._MINOR, _shift)
        if self.type == ChordType.DIMINISHED:
            return _rotate_right(self._DIMINISHED, _shift)
        if self.type == ChordType.AUGMENTED:
            return _rotate_right(self._AUGMENTED, _shift)
        raise IllegalStateError


class HarmonicChord(TemplateChord):

    @cached_property
    def vector(self) -> Vector:

        _result = ZeroVector()
        for i, key in enumerate(super().vector):
            if key:
                _harm = _harmonics(tuple(iter(ChordKey))[i])
                for _i, _key in enumerate(k for k in iter(ChordKey)):
                    _count = _harm.count(_key)
                    _result[_i] += sum(accumulate([.0125 for u in range(_count)], mul))

        return _result + super().vector


class Resolution(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    def __reversed__(self):
        return reversed(tuple(self.__iter__()))


class BasicResolution(Resolution):

    def __init__(self) -> None:
        super().__init__()

    def __iter__(self):
        return iter(IChord(k, t) for t, k in product(iter(ChordType), iter(ChordKey)))


class StrictResolution(Resolution):

    def __init__(self, chords: Sequence[IChord]) -> None:
        super().__init__()
        self._chords = set(sorted(chords))

    def __iter__(self):
        return iter(self._chords)


class TemplateChords(object):
    MAJOR = tuple(TemplateChord(key, ChordType.MAJOR) for key in ChordKey)
    MINOR = tuple(TemplateChord(key, ChordType.MINOR) for key in ChordKey)
    AUGMENTED = tuple(TemplateChord(key, ChordType.AUGMENTED) for key in ChordKey)
    DIMINISHED = tuple(TemplateChord(key, ChordType.DIMINISHED) for key in ChordKey)
    ALL = tuple(chain(MAJOR, MINOR, AUGMENTED, DIMINISHED))


class HarmonicChords(object):
    MAJOR = tuple(HarmonicChord(key, ChordType.MAJOR) for key in ChordKey)
    MINOR = tuple(HarmonicChord(key, ChordType.MINOR) for key in ChordKey)
    AUGMENTED = tuple(HarmonicChord(key, ChordType.AUGMENTED) for key in ChordKey)
    DIMINISHED = tuple(HarmonicChord(key, ChordType.DIMINISHED) for key in ChordKey)
    ALL = tuple(chain(MAJOR, MINOR, AUGMENTED, DIMINISHED))
