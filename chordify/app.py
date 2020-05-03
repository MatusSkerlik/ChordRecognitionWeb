import logging
import tempfile
from abc import abstractmethod
from collections import ChainMap
from itertools import product
from typing import Iterable, Tuple, Mapping, runtime_checkable, Protocol, Sequence

from librosa import note_to_hz

from chordify.audio_processing import _AudioProcessingFactory, PathLoadStrategyFactory, CQTExtractionStrategyFactory, \
    DefaultChromaStrategyFactory, DefaultSegmentationStrategyFactory, LoadStrategyFactory, ExtractionStrategyFactory, \
    ChromaStrategyFactory, SegmentationStrategyFactory
from chordify.chord_recognition import _ChordRecognizerFactory, TemplatePredictStrategyFactory, PredictStrategyFactory
from chordify.notation import Chord

_logger = logging.getLogger(__name__)


def _default_templates() -> Sequence[Chord]:
    _key = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    _type = (':maj', ':min')
    return tuple(Chord(''.join(s)) for s in product(_key, _type))


default_config = {

    # "DEBUG": True,
    # "PLOT_CLASS": Plotter,def _templates():
    #     _key = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    #     _type = (':maj', ':min')
    #     return product(_key, _type)
    # "CHARTS_HEIGHT": None,
    # "CHARTS_WIDTH": 6,
    # "CHARTS_ROWS": None,
    # "CHARTS_COLS": None,

    "LOAD_STRATEGY_FACTORY": PathLoadStrategyFactory,
    "EXTRACTION_STRATEGY_FACTORY": CQTExtractionStrategyFactory,
    "CHROMA_STRATEGY_FACTORY": DefaultChromaStrategyFactory,
    "SEGMENTATION_STRATEGY_FACTORY": DefaultSegmentationStrategyFactory,
    "PREDICT_STRATEGY_FACTORY": TemplatePredictStrategyFactory(_default_templates()),

    "SAMPLING_FREQUENCY": 44100,
    "N_BINS": 12 * 2 * 4,
    "BINS_PER_OCTAVE": 12 * 2,
    "MIN_FREQ": note_to_hz("C2"),
    "HOP_LENGTH": 4096,

    "MODEL_OUTPUT_DIR": tempfile.gettempdir()
}


@runtime_checkable
class Transcript(Protocol):
    """ Transcript audio file """

    @abstractmethod
    def from_audio(self, audio_filepath: str) -> Sequence[Tuple[float, object]]:
        """ Transcript audio file. Returns sequence of timestamps and chords."""


@runtime_checkable
class Learner(Protocol):
    """ Learns chords from audio samples """

    @abstractmethod
    def from_samples(self, samples: Iterable[Tuple[str, str]]) -> str:
        """ First argument of tuple is chord string, second is filepath of sample. Return path to saved model. """
        ...


class _Transcript(Transcript):

    def __init__(self, config: Mapping = None) -> None:
        super().__init__()
        if config is None:
            config = {}
        self.config = dict(ChainMap(config, default_config))
        self.audio = _AudioProcessingFactory(self.config)
        self.recognize = _ChordRecognizerFactory(self.config)

    def from_audio(self, audio_filepath: str):
        frame, time = self.audio.process(audio_filepath)
        chord_sequence = self.recognize.apply(tuple(zip(time, frame.T)))
        return chord_sequence


class Builder:

    def __init__(self, config) -> None:
        super().__init__()
        if config is None:
            config = {}
        self.config = config

    def setLoadStrategyFactory(self, factory: LoadStrategyFactory):
        self.config['LOAD_STRATEGY_FACTORY'] = factory

    def setExtractionStrategyFactory(self, factory: ExtractionStrategyFactory):
        self.config['EXTRACTION_STRATEGY_FACTORY'] = factory

    def setChromaStrategyFactory(self, factory: ChromaStrategyFactory):
        self.config['CHROMA _STRATEGY_FACTORY'] = factory

    def setSegmentationStrategyFactory(self, factory: SegmentationStrategyFactory):
        self.config['SEGMENTATION_STRATEGY_FACTORY'] = factory

    def setPredictStrategyFactory(self, factory: PredictStrategyFactory):
        self.config['PREDICT_STRATEGY_FACTORY'] = factory

    @staticmethod
    def default() -> Transcript:
        return _Transcript()

    def build(self) -> Transcript:
        return _Transcript(self.config)

# class Learn(object):
#     default_config = {
#         "CHORD_LEARNING_CLASS": SVCScikitLearnStrategy,
#     }
#
#     @log_exception()
#     def __init__(self, config: Union[dict, None] = None) -> None:
#         super().__init__()
#
#         self.config = ChainMap(config or {}, Learn.default_config, default_config)
#         self.debug = self.config["DEBUG"]
#
#         self.chord_learner = self.config["CHORD_LEARNING_CLASS"].factory(self.config)
#         self.config["CHORD_RESOLUTION"] = self.chord_learner.resolution
#
#         self.audio_processing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config)
#         self.plotter = self.config["PLOT_CLASS"].factory(self.config)
#
#     @log_exception()
#     def from_samples(self, paths: Iterator[Path] = None, labels: Iterator[Vector] = None,
#                      iterable: Iterator = None):
#         log(self.__class__, "Start learning")
#         try:
#             if iterable is None and (paths is None or labels is None):
#                 raise IllegalArgumentError
#
#             _iter = tuple(zip(paths, labels) if iterable is None else iterable)
#             _supervised_vectors = SupervisedVectors()
#
#             with Parallel(n_jobs=-3) as parallel:
#                 out = parallel(delayed(self.audio_processing.process)(path) for path, label in _iter)
#                 for vector_beat, path_label in zip(out, _iter):
#                     _supervised_vectors.append(_Vector(vector_beat[0]), path_label[1])
#
#             self.chord_learner.learn(_supervised_vectors)
#         finally:
#             log(self.__class__, "Stop learning")
#
#
# class Transcript(object):
#     default_config = {
#         "CHORD_RECOGNITION_CLASS": _TemplatePredictStrategy,
#     }
#
#     @log_exception()
#     def __init__(self, config: Union[dict, None] = None) -> None:
#         super().__init__()
#
#         self.config = ChainMap(config or {}, Transcript.default_config, default_config)
#         self.debug: bool = self.config["DEBUG"]
#
#         self.chord_recognition: _PredictStrategy = self.config["CHORD_RECOGNITION_CLASS"].factory(self.config)
#         self.config["CHORD_RESOLUTION"] = self.chord_recognition.resolution
#
#         self.audio_processing: _AudioProcessing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config)
#         self.plotter: Plotter = self.config["PLOT_CLASS"].factory(self.config)
#
#     @log_exception()
#     def from_audio(self, absolute_path: str, annotation_path: Union[str, None] = None):
#         log(self.__class__, "Start predicting")
#         _annotation_parser = None
#         _prediction_parser = None
#         _annotation_path = None
#         try:
#
#             _absolute_path = Path(absolute_path)
#             log(self.__class__, "File = " + str(_absolute_path.resolve()))
#             if annotation_path is not None:
#                 _annotation_path = Path(annotation_path)
#                 log(self.__class__, "Annotation = " + str(_annotation_path.resolve()))
#                 _annotation_parser = get_parser(self.config, _annotation_path)
#
#             chroma_sync, time_segments = self.audio_processing.process(_absolute_path)
#             labels = self.chord_recognition.predict(chroma_sync, )
#
#             if self.debug:
#                 self.plotter.chromagram(chroma_sync, time_segments)
#                 prediction_timeline = timeline_from(time_segments, labels)
#                 if annotation_path is not None:
#                     annotation_timeline = _annotation_parser.parse(_annotation_path)
#                     self.plotter.prediction(prediction_timeline, annotation_timeline)
#                 else:
#                     self.plotter.prediction(prediction_timeline, None)
#
#             self.plotter.show()
#
#             log(self.__class__, "Result: " + str(labels))
#             return labels
#         finally:
#             log(self.__class__, "Stop predicting")
