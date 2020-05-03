import logging
from abc import abstractmethod
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable, Union, Sequence

import librosa
import numpy

from .hcdf import get_segments

_logger = logging.getLogger(__name__)


class LoadStrategy:
    """ Loads music file and returns y """

    @abstractmethod
    def run(self, absolute_path: str) -> numpy.ndarray:
        pass


class ExtractionStrategy:
    """ Extracts furrier coefficients and returns bins """

    @abstractmethod
    def run(self, y: numpy.ndarray) -> numpy.ndarray:
        pass


class ChromaStrategy:
    """ Unify furrier coefficients (bins) into frames (12-d vector)"""

    @abstractmethod
    def run(self, bins: numpy.ndarray) -> numpy.ndarray:
        pass


class SegmentationStrategy:
    """ Join multiple frames into one by onset detection or HCDF """

    @abstractmethod
    def run(self, y: numpy.ndarray, chroma: numpy.ndarray) -> (numpy.ndarray, Any):
        pass


class _PathLoadStrategy(LoadStrategy):

    def __init__(self, sampling_frequency: int):
        super().__init__()

        self._sr = sampling_frequency

    @lru_cache(maxsize=None)
    def run(self, absolute_path: str) -> numpy.ndarray:
        y, sr = librosa.load(absolute_path, self._sr)
        return y


class _CQTExtractionStrategy(ExtractionStrategy):

    def __init__(self, sampling_frequency: int, hop_length: int, min_freq: int, n_bins: int,
                 bins_per_octave: int) -> None:
        super().__init__()

        self.bins_per_octave = bins_per_octave
        self._n_bins = n_bins
        self._min_freq = min_freq
        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: numpy.ndarray) -> numpy.ndarray:
        return numpy.abs(librosa.cqt(y,
                                     sr=self._sr,
                                     hop_length=self._hop_length,
                                     fmin=self._min_freq,
                                     bins_per_octave=self.bins_per_octave,
                                     n_bins=self._n_bins)
                         )


class _DefaultChromaStrategy(ChromaStrategy):

    def __init__(self, hop_length: int, min_freq: int, bins_per_octave: int) -> None:
        super().__init__()
        self._hop_length = hop_length
        self._min_freq = min_freq
        self._bins_per_octave = bins_per_octave

    def run(self, bins: numpy.ndarray) -> numpy.ndarray:
        return librosa.feature.chroma_cqt(
            C=bins,
            hop_length=self._hop_length,
            fmin=self._min_freq,
            bins_per_octave=self._bins_per_octave,
        )


class _SmoothingChromaStrategy(ChromaStrategy):

    def __init__(self, hop_length: int, min_freq: int, bins_per_octave: int) -> None:
        super().__init__()
        self._hop_length = hop_length
        self._min_freq = min_freq
        self._bins_per_octave = bins_per_octave

    def run(self, bins: numpy.ndarray) -> numpy.ndarray:
        chroma = librosa.feature.chroma_cqt(
            C=bins,
            hop_length=self._hop_length,
            fmin=self._min_freq,
            bins_per_octave=self._bins_per_octave,
        )

        return numpy.minimum(chroma,
                             librosa.decompose.nn_filter(chroma,
                                                         aggregate=numpy.median,
                                                         metric='cosine'))


class _HPSSChromaStrategy(ChromaStrategy):

    def __init__(self, hop_length: int, min_freq: int, bins_per_octave: int, n_octaves: int) -> None:
        super().__init__()
        self._hop_length = hop_length
        self._min_freq = min_freq
        self._bins_per_octave = bins_per_octave
        self._n_octaves = n_octaves

    def run(self, bins: numpy.ndarray) -> numpy.ndarray:
        h, p = librosa.decompose.hpss(bins)

        chroma = librosa.feature.chroma_cqt(
            C=h,
            hop_length=self._hop_length,
            fmin=self._min_freq,
            bins_per_octave=self._bins_per_octave,
            n_octaves=self._n_octaves
        )

        chroma = numpy.minimum(chroma,
                               librosa.decompose.nn_filter(chroma,
                                                           aggregate=numpy.median,
                                                           metric='cosine'))
        return chroma


class _DefaultSegmentationStrategy(SegmentationStrategy):

    def __init__(self, sampling_frequency: int, hop_length: int) -> None:
        super().__init__()

        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: numpy.ndarray, chroma: numpy.ndarray) -> (numpy.ndarray, Union[Sequence[float], None]):
        return chroma, librosa.frames_to_time(list(range(chroma.shape[1])),
                                              sr=self._sr,
                                              hop_length=self._hop_length)


class _VectorSegmentationStrategy(SegmentationStrategy):

    def __init__(self) -> None:
        super().__init__()

    def run(self, y: numpy.ndarray, chroma: numpy.ndarray) -> (numpy.ndarray, Union[Sequence[float], None]):
        frame = librosa.util.sync(chroma, [0], aggregate=numpy.median).flatten()
        return frame, [0, 0]


class _BeatSegmentationStrategy(SegmentationStrategy):

    def __init__(self, sampling_frequency: int, hop_length: int) -> None:
        super().__init__()

        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: numpy.ndarray, chroma: numpy.ndarray) -> (numpy.ndarray, Union[Sequence[float], None]):
        tempo, beat_f = librosa.beat.beat_track(y=y, sr=self._sr, hop_length=self._hop_length, trim=False)
        beat_f = librosa.util.fix_frames(beat_f, x_max=chroma.shape[1])
        sync_chroma = librosa.util.sync(chroma, beat_f, aggregate=numpy.median)
        beat_t = librosa.frames_to_time(beat_f, sr=self._sr, hop_length=self._hop_length)
        return sync_chroma, beat_t


class _HCDFSegmentationStrategy(SegmentationStrategy):

    def __init__(self, sampling_frequency: int, hop_length: int) -> None:
        super().__init__()

        self._hop_length = hop_length
        self._sr = sampling_frequency

    def run(self, y: numpy.ndarray, chroma: numpy.ndarray) -> (numpy.ndarray, Union[Sequence[float], None]):
        _segments, _peaks = get_segments(chroma)
        _med_segments = list()
        for vectors in _segments:
            vector = librosa.util.sync(vectors, [0], aggregate=numpy.median)
            _med_segments.append(vector.flatten())
        return numpy.array(_med_segments).T, librosa.frames_to_time(_peaks, sr=self._sr, hop_length=self._hop_length)


class _AudioProcessing:

    def __init__(self, load_strategy: LoadStrategy, extraction_strategy: ExtractionStrategy,
                 chroma_strategy: ChromaStrategy,
                 segmentation_strategy: SegmentationStrategy) -> None:
        super().__init__()

        if load_strategy is None:
            raise NameError("Load strategy cannot be None !")
        if extraction_strategy is None:
            raise NameError("Extraction strategy cannot be None !")
        if chroma_strategy is None:
            raise NameError("Chroma strategy cannot be None !")
        if segmentation_strategy is None:
            raise NameError("Segmentation strategy cannot be None !")

        self.load_strategy = load_strategy
        self.extraction_strategy = extraction_strategy
        self.chroma_strategy = chroma_strategy
        self.segmentation_strategy = segmentation_strategy

    def process(self, absolute_path: str) -> (numpy.ndarray, Union[Sequence[float], None]):
        _logger.info(self.__class__, "Processing = " + absolute_path)
        y = self.load_strategy.run(absolute_path)
        bins = self.extraction_strategy.run(y)
        chroma = self.chroma_strategy.run(bins)
        return self.segmentation_strategy.run(y, chroma)


@runtime_checkable
class LoadStrategyFactory(Protocol):
    def __call__(self, config: dict) -> LoadStrategy: ...


@runtime_checkable
class ChromaStrategyFactory(Protocol):
    def __call__(self, config: dict) -> ChromaStrategy: ...


@runtime_checkable
class ExtractionStrategyFactory(Protocol):
    def __call__(self, config: dict) -> ExtractionStrategy: ...


@runtime_checkable
class SegmentationStrategyFactory(Protocol):
    def __call__(self, config: dict) -> SegmentationStrategy: ...


def _AudioProcessingFactory(config: dict):
    load_strategy_factory = config['LOAD_STRATEGY_FACTORY']
    extraction_strategy_factory = config['EXTRACTION_STRATEGY_FACTORY']
    chroma_strategy_factory = config['CHROMA_STRATEGY_FACTORY']
    segmentation_strategy_factory = config['SEGMENTATION_STRATEGY_FACTORY']

    if not isinstance(load_strategy_factory, LoadStrategyFactory):
        raise ValueError("Load strategy must obey LoadStrategyFactory Protocol.")
    if not isinstance(extraction_strategy_factory, ExtractionStrategyFactory):
        raise ValueError("Load strategy must obey ExtractionStrategyFactory Protocol.")
    if not isinstance(chroma_strategy_factory, ChromaStrategyFactory):
        raise ValueError("Load strategy must obey ChromaStrategyFactory Protocol.")
    if not isinstance(segmentation_strategy_factory, SegmentationStrategyFactory):
        raise ValueError("Load strategy must obey SegmentationStrategyFactory Protocol.")

    return _AudioProcessing(
        load_strategy_factory(config),
        extraction_strategy_factory(config),
        chroma_strategy_factory(config),
        segmentation_strategy_factory(config)
    )


def PathLoadStrategyFactory(config: dict) -> LoadStrategy:
    return _PathLoadStrategy(config["SAMPLING_FREQUENCY"])


def CQTExtractionStrategyFactory(config: dict) -> ExtractionStrategy:
    return _CQTExtractionStrategy(config["SAMPLING_FREQUENCY"],
                                  config["HOP_LENGTH"],
                                  config["MIN_FREQ"],
                                  config["N_BINS"],
                                  config["BINS_PER_OCTAVE"])


def DefaultChromaStrategyFactory(config: dict) -> ChromaStrategy:
    return _DefaultChromaStrategy(
        config["HOP_LENGTH"],
        config["MIN_FREQ"],
        config["BINS_PER_OCTAVE"]
    )


def SmoothingChromaStrategyFactory(config: dict) -> ChromaStrategy:
    return _SmoothingChromaStrategy(
        config["HOP_LENGTH"],
        config["MIN_FREQ"],
        config["BINS_PER_OCTAVE"]
    )


def HPSSChromaStrategyFactory(config: dict) -> ChromaStrategy:
    return _HPSSChromaStrategy(
        config["HOP_LENGTH"],
        config["MIN_FREQ"],
        config["BINS_PER_OCTAVE"],
        config["N_OCTAVES"]
    )


def DefaultSegmentationStrategyFactory(config: dict) -> SegmentationStrategy:
    return _DefaultSegmentationStrategy(
        config["SAMPLING_FREQUENCY"],
        config["HOP_LENGTH"]
    )


def BeatSegmentationStrategyFactory(config: dict) -> SegmentationStrategy:
    return _BeatSegmentationStrategy(
        config["SAMPLING_FREQUENCY"],
        config["HOP_LENGTH"]
    )


def VectorSegmentationStrategyFactory(config: dict) -> SegmentationStrategy:
    return _VectorSegmentationStrategy()


def HCDFSegmentationStrategy(config: dict) -> SegmentationStrategy:
    return _HCDFSegmentationStrategy(
        config["SAMPLING_FREQUENCY"],
        config["HOP_LENGTH"]
    )
