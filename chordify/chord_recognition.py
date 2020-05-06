import logging
from abc import ABC, abstractmethod
from typing import Protocol, Sequence, List, Tuple, overload, Callable, runtime_checkable

_logger = logging.getLogger(__name__)


@runtime_checkable
class _Vector(Protocol):
    def __matmul__(self, other) -> float: ...

    def __rmatmul__(self, other) -> float: ...


class _FrameSequence(tuple):

    def __new__(cls, iterable: Sequence[Tuple[float, _Vector]]):
        try:
            for t in iterable:
                if not isinstance(t, tuple) or \
                        not isinstance(t[0], float) or \
                        not isinstance(t[1], object) or \
                        not hasattr(t[1], '__matmul__') or \
                        not hasattr(t[1], '__rmatmul__'):
                    raise ValueError("Not valid values")
            _last = 0
            for t in iterable:
                if t[0] < _last:
                    raise ValueError("Frame time error")
                _last = t[0]
        except IndexError:
            raise ValueError("Not valid values")

        return super().__new__(tuple, iterable)

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> Sequence[Tuple[float, _Vector]]:
        return tuple(self[i] for i in range(s.start, s.stop, s.step))

    def __getitem__(self, i: int) -> Tuple[float, _Vector]:
        return super().__getitem__(i)


@runtime_checkable
class PredictStrategy(Protocol):

    def predict(self, frame: _Vector, threshold: Callable[[float], bool] = lambda t: t) -> _Vector:
        ...


@runtime_checkable
class PredictStrategyFactory(Protocol):

    def __call__(self, config: dict) -> PredictStrategy: ...


class _PredictStrategy(PredictStrategy, ABC):

    @property
    @abstractmethod
    def templates(self) -> Sequence[_Vector]:
        ...

    def predict(self, frame: _Vector, threshold: Callable[[float], bool] = lambda t: t) -> _Vector:
        _logger.info(self.__class__, "Predicting...")

        all_products: List[float] = list()

        for vector in self.templates:
            all_products.append(vector @ frame)

        return self.templates[all_products.index(max(all_products))]


class _TemplatePredictStrategy(_PredictStrategy):

    def __init__(self, templates: Sequence[_Vector]) -> None:
        super().__init__()
        self._templates = templates

    @property
    def templates(self) -> Sequence[_Vector]:
        return self._templates


def TemplatePredictStrategyFactory(templates: Sequence[_Vector]) -> PredictStrategyFactory:
    for template in templates:
        if not isinstance(template, _Vector):
            raise NameError('Templates must obey Chord protocol.')

    def wrapper(config: dict) -> PredictStrategy:
        return _TemplatePredictStrategy(templates)

    return wrapper


class ChordRecognizer(Protocol):

    def apply(self, sequence: Sequence[Tuple[float, _Vector]],
              threshold: Callable[[float], bool] = lambda t: t) -> Sequence[Tuple[float, _Vector]]: ...


class _ChordRecognizer:

    def __init__(self, strategy: PredictStrategy) -> None:
        super().__init__()
        self.strategy = strategy

    def apply(self, sequence: Sequence[Tuple[float, _Vector]],
              threshold: Callable[[float], bool] = lambda t: t) -> Sequence[Tuple[float, _Vector]]:
        frame_sequence = _FrameSequence(sequence)

        chord_sequence: List[Tuple[float, _Vector]] = list()
        for stop, frame in frame_sequence:
            chord_sequence.append((stop, self.strategy.predict(frame, threshold)))

        return chord_sequence


def _ChordRecognizerFactory(config: dict) -> ChordRecognizer:
    predict_strategy_factory = config['PREDICT_STRATEGY_FACTORY']

    if not isinstance(predict_strategy_factory, PredictStrategyFactory):
        raise ValueError('PredictStrategyFactory must obey protocol PredictStrategyFactory.')

    predict_strategy = predict_strategy_factory(config)

    if not isinstance(predict_strategy, PredictStrategy):
        raise ValueError('PredictStrategy must obey protocol PredictStrategy.')

    return _ChordRecognizer(predict_strategy)
