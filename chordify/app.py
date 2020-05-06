import logging
import os
import pickle
import tempfile
from abc import abstractmethod
from collections import ChainMap
from itertools import product, tee
from pickle import Pickler, Unpickler
from typing import Iterable, Tuple, Mapping, runtime_checkable, Protocol, Sequence

from joblib import delayed, Parallel
from librosa import note_to_hz

from .audio_processing import _AudioProcessingFactory, PathLoadStrategyFactory, CQTExtractionStrategyFactory, \
    DefaultChromaStrategyFactory, DefaultSegmentationStrategyFactory, LoadStrategyFactory, ExtractionStrategyFactory, \
    ChromaStrategyFactory, SegmentationStrategyFactory, VectorSegmentationStrategyFactory
from .chord_recognition import _ChordRecognizerFactory, TemplatePredictStrategyFactory, PredictStrategyFactory, \
    PredictStrategy
from .learn import SVMClassifier
from .notation import Chord

_logger = logging.getLogger(__name__)


def _default_templates() -> Sequence[Chord]:
    _key = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    _type = (':maj', ':min')
    return tuple(Chord(''.join(s)) for s in product(_key, _type))


default_config = {

    # 'DEBUG': True,

    'LOAD_STRATEGY_FACTORY': PathLoadStrategyFactory,
    'EXTRACTION_STRATEGY_FACTORY': CQTExtractionStrategyFactory,
    'CHROMA_STRATEGY_FACTORY': DefaultChromaStrategyFactory,
    'SEGMENTATION_STRATEGY_FACTORY': DefaultSegmentationStrategyFactory,
    'PREDICT_STRATEGY_FACTORY': TemplatePredictStrategyFactory(_default_templates()),

    'SAMPLING_FREQUENCY': 44100,
    'N_BINS': 12 * 2 * 4,
    'BINS_PER_OCTAVE': 12 * 2,
    'MIN_FREQ': note_to_hz('C2'),
    'HOP_LENGTH': 4096,

    'MODEL_OUTPUT_DIR': tempfile.gettempdir()
}


@runtime_checkable
class Transcript(Protocol):
    """ Transcript audio file """

    @abstractmethod
    def from_audio(self, audio_filepath: str) -> Sequence[Tuple[float, object]]:
        """ Transcript audio file. Returns sequence of timestamps and chords."""


@runtime_checkable
class LearnedStrategy(Protocol):
    """ Proxy for predict strategy with save method """

    def save(self, filename: str):
        """ Save last learned strategy to MODEL_OUTPUT_DIR + filename, can be called after :method from_samples()  """
        ...


@runtime_checkable
class Learner(Protocol):
    """ Learns chords from audio samples """

    @abstractmethod
    def from_samples(self, samples: Iterable[Tuple[str, str]]) -> LearnedStrategy:
        """ First argument of tuple is chord string, second is filepath of sample. Return path to saved model. """
        ...

    def load(self, filename: str) -> LearnedStrategy:
        """ Load predict strategy from MODEL_OUTPUT_DIR + filename  """
        ...


class _ConfigBuilder:

    def __init__(self, config: Mapping = None) -> None:
        super().__init__()
        if config is None:
            config = {}
        self._config = dict(ChainMap(config, default_config))

    def setLoadStrategyFactory(self, factory: LoadStrategyFactory):
        self._config['LOAD_STRATEGY_FACTORY'] = factory
        return self

    def setExtractionStrategyFactory(self, factory: ExtractionStrategyFactory):
        self._config['EXTRACTION_STRATEGY_FACTORY'] = factory
        return self

    def setChromaStrategyFactory(self, factory: ChromaStrategyFactory):
        self._config['CHROMA _STRATEGY_FACTORY'] = factory
        return self

    def setSegmentationStrategyFactory(self, factory: SegmentationStrategyFactory):
        self._config['SEGMENTATION_STRATEGY_FACTORY'] = factory
        return self

    def setPredictStrategyFactory(self, factory: PredictStrategyFactory):
        self._config['PREDICT_STRATEGY_FACTORY'] = factory
        return self

    def build(self) -> dict:
        return self._config


class _Transcript(Transcript):

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.audio = _AudioProcessingFactory(config)
        self.recognize = _ChordRecognizerFactory(config)

    def from_audio(self, audio_filepath: str):
        frame, time = self.audio.process(audio_filepath)
        chord_sequence = self.recognize.apply(tuple(zip(time, frame.T)))
        return chord_sequence


class TranscriptBuilder(_ConfigBuilder):
    """ Use for instantiate Transcript """

    # TODO not properly set, bug in chord_recognition module (PredictStrategyFactory protocol check will pass)
    def setLearnedStrategy(self, strategy: LearnedStrategy):
        self._config['PREDICT_STRATEGY_FACTORY'] = lambda config: strategy
        return self

    @staticmethod
    def default() -> Transcript:
        return _Transcript(default_config)

    def build(self) -> Transcript:
        return _Transcript(dict(ChainMap(
            super().build(),
            default_config
        )))


class _LearnedStrategy(LearnedStrategy):

    def __init__(self, predict_strategy: PredictStrategy, model_output_dir: str) -> None:
        super().__init__()

        if not isinstance(predict_strategy, PredictStrategy):
            raise NameError('Wrong type of predict_strategy.')

        self._predict_strategy = predict_strategy
        self._model_output_dir = model_output_dir

    def __getattr__(self, attr):
        """ invoked if the attribute wasn't found the usual ways """

        if '_predict_strategy' not in self.__dict__:
            raise AttributeError
        if '_model_output_dir' not in self.__dict__:
            raise AttributeError

        if hasattr(self._predict_strategy, attr):
            return getattr(self._predict_strategy, attr)

        raise AttributeError

    def save(self, filename: str):
        with open(os.path.join(self._model_output_dir, filename), 'wb') as file:
            Pickler(file, pickle.DEFAULT_PROTOCOL).dump(self)


class _Learner(Learner):

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.audio = _AudioProcessingFactory(config)
        self.model_output = config['MODEL_OUTPUT_DIR']

    def from_samples(self, samples: Iterable[Tuple[str, str]]) -> LearnedStrategy:
        label_set, vector_time_set = tee(samples)

        label_set = tuple(label_filepath[0] for label_filepath in label_set)
        vector_time_set = Parallel(n_jobs=2)(
            delayed(self.audio.process)(label_filepath[1]) for label_filepath in vector_time_set
        )
        vector_set = tuple(vector_time[0] for vector_time in vector_time_set)

        # TODO change classifier for argument or builder setter
        return _LearnedStrategy(SVMClassifier(vector_set, label_set), self.model_output)

    def load(self, filename: str) -> LearnedStrategy:
        with open(os.path.join(self.model_output, filename), 'rb') as file:
            return Unpickler(file).load()


class LearnerBuilder(_ConfigBuilder):
    """ Use for instantiate Learner """

    # TODO check if dir exists
    def setModelOutputDir(self, path: str):
        self._config['MODEL_OUTPUT_DIR'] = path
        return self

    @staticmethod
    def default() -> Learner:
        return _Learner(dict(ChainMap(
            {'SEGMENTATION_STRATEGY_FACTORY': VectorSegmentationStrategyFactory},
            default_config
        )))

    def build(self) -> Learner:
        return _Learner(dict(ChainMap(
            {'SEGMENTATION_STRATEGY_FACTORY': VectorSegmentationStrategyFactory},
            super().build(),
            default_config
        )))
