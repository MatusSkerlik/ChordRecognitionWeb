#  Copyright 2020 Matúš Škerlík
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
#  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
#

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this
#  software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify, merge,
#  publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
#  to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Tuple, List, Sequence
from pandas import read_csv
from .exceptions import IllegalArgumentError
from .music import IChord


class ChordTimeline(Sequence):
    _stop: List[float]
    _chords: List[IChord]
    _counter: int = 0

    def __init__(self):
        super().__init__()
        self._stop = list()
        self._chords = list()

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self) -> Tuple[float, float, IChord]:
        if self._counter < len(self):
            self._counter += 1
            return self.start()[self._counter - 1], self.stop()[self._counter - 1], self.chords()[self._counter - 1]
        else:
            raise StopIteration

    def __getitem__(self, item) -> Tuple[float, float, IChord]:
        return self.start()[item], self.stop()[item], self.chords()[item]

    def __len__(self) -> int:
        return len(self._chords)

    def append(self, stop: float, chord: IChord):
        if stop is None or stop <= 0 or chord is None:
            raise IllegalArgumentError

        if len(self._stop) > 0 and self._stop[-1] > stop:
            raise IllegalArgumentError
        self._stop.append(stop)
        self._chords.append(chord)

    def start(self) -> Tuple[float, ...]:
        return tuple(chain((0.0,), self._stop))

    def stop(self) -> Tuple[float, ...]:
        return tuple(self._stop)

    def chords(self) -> Tuple[IChord, ...]:
        return tuple(self._chords)

    def duration(self) -> float:
        return self._stop[-1]


class Strategy(ABC):

    @classmethod
    @abstractmethod
    def factory(cls, config):
        pass


class AnnotationParser(Strategy):

    @staticmethod
    @abstractmethod
    def accept(ext: str) -> bool:
        pass

    @abstractmethod
    def parse(self, absolute_path) -> ChordTimeline:
        pass


class LabParser(AnnotationParser):

    @staticmethod
    def accept(ext: Path) -> bool:
        return ext.match("*.lab")

    @classmethod
    def factory(cls, config):
        return LabParser(config["CHORD_RESOLUTION"])

    def __init__(self, chord_resolution) -> None:
        super().__init__()

        self.chord_resolution = chord_resolution

    def parse(self, absolute_path) -> ChordTimeline:
        assert self.__class__.accept(absolute_path)

        csv = read_csv(filepath_or_buffer=absolute_path, header=None, skip_blank_lines=True, delimiter=" ")

        _timeline = ChordTimeline()
        for i, row in csv.iterrows():
            start, stop, chord = row
            for _chord in self.chord_resolution:
                if chord == str(_chord):
                    _timeline.append(stop, _chord)
                    break
            _timeline.append(stop, IChord())

        return _timeline


def get_parser(config, annotation_path: Path) -> AnnotationParser:
    for annotation_processor in __processors__:
        if annotation_processor.accept(annotation_path):
            return annotation_processor.factory(config)
    raise NotImplementedError("Not supported file format.")


def timeline_from(time_segments: Tuple[float, ...], labels: [IChord, ...]) -> ChordTimeline:
    _timeline = ChordTimeline()
    for stop, chord in zip(time_segments[1:], labels):
        _timeline.append(stop, chord)
    return _timeline


__processors__ = [
    LabParser
]
