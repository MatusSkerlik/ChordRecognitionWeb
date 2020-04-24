import os
from abc import abstractmethod, ABC
from collections import Sized, Iterator
from pickle import dump
from typing import List, Tuple, Dict, BinaryIO

import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from .chord_recognition import PredictStrategy
from .exceptions import IllegalArgumentError
from .logger import log
from .music import Vector, IChord, Resolution, StrictResolution


class RGridSearchCV(GridSearchCV):
    _encoder: LabelEncoder

    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False):
        super().__init__(estimator, param_grid, scoring, n_jobs, iid, refit, cv, verbose, pre_dispatch, error_score,
                         return_train_score)
        self._encoder = LabelEncoder()

    def fit(self, x: Tuple[Vector], y: Tuple[IChord] = None, groups=None, **fit_params):
        _y = self._encoder.fit_transform(tuple(map(str, y)))
        return super().fit(np.array(x), _y, groups, **fit_params)

    def r_predict(self, vectors: np.ndarray, chord_resolution: Resolution) -> Tuple[IChord]:
        _l_ch_map: Dict[str, IChord] = {str(r): r for r in chord_resolution}
        _y = self.predict(vectors)
        return tuple(map(lambda l: _l_ch_map[l], self._encoder.inverse_transform(_y)))


class SupervisedVectors(Sized, Iterator):
    _vectors: List[Vector]
    _labels: List[IChord]

    _n: int
    _len: int

    def __init__(self) -> None:
        super().__init__()
        self._vectors = list()
        self._labels = list()
        self._n = 0
        self._len = 0

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < self._len:
            self._n += 1
            return self._vectors[self._n - 1], self._labels[self._n - 1]
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self._vectors[item], self._labels[item]

    def __len__(self) -> int:
        return self._len

    def append(self, vector: Vector, label: IChord):
        if vector is None or label is None:
            raise IllegalArgumentError

        self._vectors.append(vector)
        self._labels.append(label)
        self._len += 1

    def vectors(self) -> Tuple[Vector]:
        return tuple(self._vectors)

    def labels(self) -> Tuple[IChord]:
        return tuple(self._labels)


class Strategy(ABC):

    @classmethod
    @abstractmethod
    def factory(cls, config):
        pass


class LearnStrategy(PredictStrategy):

    @classmethod
    def factory(cls, config):
        pass

    @abstractmethod
    def learn(self, supervised_vectors: SupervisedVectors):
        pass


class ScikitLearnStrategy(LearnStrategy):
    ch_resolution: StrictResolution
    classifier: RGridSearchCV

    def __init__(self, estimator: BaseEstimator, file: BinaryIO, **kwargs) -> None:
        self.classifier = RGridSearchCV(estimator, kwargs, cv=5, n_jobs=-1)
        self.output_file = file

    @property
    def resolution(self) -> Resolution:
        return self.ch_resolution

    def learn(self, supervised_vectors: SupervisedVectors):
        log(self.__class__, "Learning...")
        self.ch_resolution = StrictResolution(supervised_vectors.labels())
        self.classifier.fit(supervised_vectors.vectors(), supervised_vectors.labels())
        log(self.__class__, "Learning done...")

        output_file = self.output_file
        del self.output_file

        with output_file as f:
            log(self.__class__, "Dumping model = " + str(f))
            dump(self, f)

    def predict(self, vectors: np.ndarray) -> Tuple[IChord]:
        log(self.__class__, "Predicting...")
        return self.classifier.r_predict(vectors.T, self.ch_resolution)


class SVCScikitLearnStrategy(ScikitLearnStrategy):

    @classmethod
    def factory(cls, config) -> ScikitLearnStrategy:
        log(cls, "Init")
        filepath = os.path.join(config["MODEL_OUTPUT_DIR"], cls.__name__ + ".pickle")
        return SVCScikitLearnStrategy(svm.SVC(), open(filepath, "wb"), C=[1, 50])
