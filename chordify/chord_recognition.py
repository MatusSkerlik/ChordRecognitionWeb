from abc import *
from typing import Sequence

import numpy as np

from chordify.logger import log
from .music import TemplateChords, HarmonicChords, Resolution, BasicResolution, IChord


class Strategy(ABC):

    @classmethod
    @abstractmethod
    def factory(cls, config):
        pass


class PredictStrategy(Strategy):

    @abstractmethod
    def predict(self, chroma) -> Sequence[IChord]:
        pass

    @property
    @abstractmethod
    def resolution(self) -> Resolution:
        pass


class TemplatePredictStrategy(PredictStrategy):

    @classmethod
    def factory(cls, config):
        log(cls, "Init")
        return TemplatePredictStrategy()

    @property
    def resolution(self) -> Resolution:
        return BasicResolution()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        log(self.__class__, "Predicting...")
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, TemplateChords.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(TemplateChords.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)


class HarmonicPredictStrategy(PredictStrategy):

    @classmethod
    def factory(cls, config):
        log(cls, "Init")
        return HarmonicPredictStrategy()

    @property
    def resolution(self) -> Resolution:
        return BasicResolution()

    def predict(self, chroma: np.ndarray, filter_func=lambda d: d) -> tuple:
        log(self.__class__, "Predicting...")
        _chord_prg = list()
        _ch_v = np.array(list(map(lambda c: c.vector, HarmonicChords.ALL)))

        for ch in chroma.T:
            _dots = filter_func(_ch_v.dot(ch))
            _chord_prg.append(HarmonicChords.ALL[np.argmax(_dots)])

        return tuple(_chord_prg)
