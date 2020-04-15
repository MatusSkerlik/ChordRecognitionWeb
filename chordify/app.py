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

import tempfile
from collections import ChainMap
from typing import Iterator, Union

from joblib import Parallel, delayed
from librosa import note_to_hz

from chordify.annotation import get_parser, timeline_from
from chordify.exceptions import IllegalConfigError
from chordify.learn import SupervisedVectors, SVCScikitLearnStrategy
from .audio_processing import *
from .chord_recognition import *
from .display import Plotter
from .music import Vector


def check_config(config):
    if "AUDIO_PROCESSING_CLASS" in config and not issubclass(config["AUDIO_PROCESSING_CLASS"], AudioProcessing):
        raise IllegalConfigError
    if "AP_LOAD_STRATEGY_CLASS" in config and not issubclass(config["AP_LOAD_STRATEGY_CLASS"], LoadStrategy):
        raise IllegalConfigError
    if "AP_EXTRACTION_STRATEGY_CLASS" in config and not issubclass(config["AP_STFT_STRATEGY_CLASS"], ExtractionStrategy):
        raise IllegalConfigError
    if "AP_CHROMA_STRATEGY_CLASS" in config and not issubclass(config["AP_CHROMA_STRATEGY_CLASS"],
                                                               ChromaStrategy):
        raise IllegalConfigError
    if "AP_SEGMENTATION_STRATEGY_CLASS" in config and not issubclass(config["AP_BEAT_STRATEGY_CLASS"], SegmentationStrategy):
        raise IllegalConfigError
    if "CHORD_RECOGNITION_CLASS" in config and not issubclass(config["CHORD_RECOGNITION_CLASS"],
                                                              Strategy):
        raise IllegalConfigError
    if "CHORD_LEARNING_CLASS" in config and not issubclass(config["CHORD_LEARNING_CLASS"], Strategy):
        raise IllegalConfigError


default_config = {

    "DEBUG": True,
    "PLOT_CLASS": Plotter,
    "CHARTS_HEIGHT": None,
    "CHARTS_WIDTH": 6,
    "CHARTS_ROWS": None,
    "CHARTS_COLS": None,

    "AUDIO_PROCESSING_CLASS": AudioProcessing,
    "AP_LOAD_STRATEGY_CLASS": PathLoadStrategy,
    "AP_EXTRACTION_STRATEGY_CLASS": CQTExtractionStrategy,
    "AP_CHROMA_STRATEGY_CLASS": DefaultChromaStrategy,
    "AP_SEGMENTATION_STRATEGY_CLASS": DefaultSegmentationStrategy,

    "SAMPLING_FREQUENCY": 44100,
    "N_BINS": 12 * 2 * 4,
    "BINS_PER_OCTAVE": 12 * 2,
    "MIN_FREQ": note_to_hz("C2"),
    "HOP_LENGTH": 4096,

    "MODEL_OUTPUT_DIR": tempfile.gettempdir()
}


class Learn(object):
    default_config = {
        "CHORD_LEARNING_CLASS": SVCScikitLearnStrategy,
    }

    def __init__(self,  config: Union[dict, None] = None) -> None:
        super().__init__()

        self.config = ChainMap(config or {}, Learn.default_config, default_config)
        self.debug = self.config["DEBUG"]

        self.chord_learner = self.config["CHORD_LEARNING_CLASS"].factory(self.config)
        self.config["CHORD_RESOLUTION"] = self.chord_learner.resolution

        self.audio_processing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config)
        self.plotter = self.config["PLOT_CLASS"].factory(self.config)

    def from_samples(self, paths: Iterator[Path] = None, labels: Iterator[IChord] = None,
                     iterable: Iterator = None):
        log(self.__class__, "Start learning")
        try:
            if iterable is None and (paths is None or labels is None):
                raise IllegalArgumentError

            _iter = tuple(zip(paths, labels) if iterable is None else iterable)
            _supervised_vectors = SupervisedVectors()

            with Parallel(n_jobs=-3) as parallel:
                out = parallel(delayed(self.audio_processing.process)(path) for path, label in _iter)
                for vector_beat, path_label in zip(out, _iter):
                    _supervised_vectors.append(Vector(vector_beat[0]), path_label[1])

            self.chord_learner.learn(_supervised_vectors)
        finally:
            log(self.__class__, "Stop learning")


class Transcript(object):
    default_config = {
        "CHORD_RECOGNITION_CLASS": TemplatePredictStrategy,
    }

    def __init__(self, config: Union[dict, None] = None) -> None:
        super().__init__()

        self.config = ChainMap(config or {}, Transcript.default_config, default_config)
        self.debug: bool = self.config["DEBUG"]

        self.chord_recognition: PredictStrategy = self.config["CHORD_RECOGNITION_CLASS"].factory(self.config)
        self.config["CHORD_RESOLUTION"] = self.chord_recognition.resolution

        self.audio_processing: AudioProcessing = self.config["AUDIO_PROCESSING_CLASS"].factory(self.config)
        self.plotter: Plotter = self.config["PLOT_CLASS"].factory(self.config)

    def from_audio(self, absolute_path: str, annotation_path: Union[str, None] = None):
        log(self.__class__, "Start predicting")
        _annotation_parser = None
        _prediction_parser = None
        _annotation_path = None
        try:

            _absolute_path = Path(absolute_path)
            log(self.__class__, "File = " + str(_absolute_path.resolve()))
            if annotation_path is not None:
                _annotation_path = Path(annotation_path)
                log(self.__class__, "Annotation = " + str(_annotation_path.resolve()))
                _annotation_parser = get_parser(self.config, _annotation_path)

            chroma_sync, time_segments = self.audio_processing.process(_absolute_path)
            labels = self.chord_recognition.predict(chroma_sync)

            if self.debug:
                self.plotter.chromagram(chroma_sync, time_segments)
                prediction_timeline = timeline_from(time_segments, labels)
                if annotation_path is not None:
                    annotation_timeline = _annotation_parser.parse(_annotation_path)
                    self.plotter.prediction(prediction_timeline, annotation_timeline)
                else:
                    self.plotter.prediction(prediction_timeline, None)

            self.plotter.show()

            log(self.__class__, "Result: " + str(labels))
            return labels
        finally:
            log(self.__class__, "Stop predicting")
