import logging
import os
import unittest
from itertools import chain, cycle

import librosa

from chordify.app import TranscriptBuilder, Transcript, LearnerBuilder
from chordify.audio_processing import VectorSegmentationStrategyFactory, CQTExtractionStrategyFactory, \
    PathLoadStrategyFactory, DefaultChromaStrategyFactory

_logger = logging.getLogger(__name__)


class TestTranscript(unittest.TestCase):
    def test_error_if_bad_config(self):
        with self.assertRaises(ValueError):
            TranscriptBuilder().setChromaStrategyFactory(VectorSegmentationStrategyFactory).build()
            TranscriptBuilder().setLoadStrategyFactory(VectorSegmentationStrategyFactory).build()
            TranscriptBuilder().setExtractionStrategyFactory(VectorSegmentationStrategyFactory).build()

            TranscriptBuilder().setChromaStrategyFactory(CQTExtractionStrategyFactory).build()
            TranscriptBuilder().setLoadStrategyFactory(CQTExtractionStrategyFactory).build()
            TranscriptBuilder().setSegmentationStrategyFactory(CQTExtractionStrategyFactory).build()

            TranscriptBuilder().setChromaStrategyFactory(PathLoadStrategyFactory).build()
            TranscriptBuilder().setExtractionStrategyFactory(PathLoadStrategyFactory).build()
            TranscriptBuilder().setSegmentationStrategyFactory(PathLoadStrategyFactory).build()

            TranscriptBuilder().setLoadStrategyFactory(DefaultChromaStrategyFactory).build()
            TranscriptBuilder().setExtractionStrategyFactory(DefaultChromaStrategyFactory).build()
            TranscriptBuilder().setSegmentationStrategyFactory(DefaultChromaStrategyFactory).build()

    def test_app(self):
        app = TranscriptBuilder.default()

        self.assertTrue(isinstance(app, Transcript))

        result = app.from_audio(librosa.util.example_audio_file())

        self.assertTrue(result)

        for result_pair in result:
            self.assertTrue(isinstance(result_pair[0], float))
            self.assertTrue(isinstance(result_pair[1], object))

    def test_error_if_file_does_not_exist(self):
        app = TranscriptBuilder.default()
        with self.assertRaises(IOError):
            app.from_audio('unknown_path')


class TestLearner(unittest.TestCase):

    # TODO configure logger
    def setUp(self) -> None:
        pass

    def test_error_if_bad_config(self):
        with self.assertRaises(ValueError):
            LearnerBuilder().setChromaStrategyFactory(VectorSegmentationStrategyFactory).build()
            LearnerBuilder().setLoadStrategyFactory(VectorSegmentationStrategyFactory).build()
            LearnerBuilder().setExtractionStrategyFactory(VectorSegmentationStrategyFactory).build()

            LearnerBuilder().setChromaStrategyFactory(CQTExtractionStrategyFactory).build()
            LearnerBuilder().setLoadStrategyFactory(CQTExtractionStrategyFactory).build()
            LearnerBuilder().setSegmentationStrategyFactory(CQTExtractionStrategyFactory).build()

            LearnerBuilder().setChromaStrategyFactory(PathLoadStrategyFactory).build()
            LearnerBuilder().setExtractionStrategyFactory(PathLoadStrategyFactory).build()
            LearnerBuilder().setSegmentationStrategyFactory(PathLoadStrategyFactory).build()

            LearnerBuilder().setLoadStrategyFactory(DefaultChromaStrategyFactory).build()
            LearnerBuilder().setExtractionStrategyFactory(DefaultChromaStrategyFactory).build()
            LearnerBuilder().setSegmentationStrategyFactory(DefaultChromaStrategyFactory).build()

    def test_app(self):
        app = LearnerBuilder.default()
        result = app.from_samples(chain(
            zip(cycle('A'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/a/'))),
            zip(cycle('C'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/c/'))),
            zip(cycle('D'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/d/'))),
            zip(cycle('E'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/e/'))),
            zip(cycle('F'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/f/'))),
            zip(cycle('G'), tuple(directory.path for directory in
                                  os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/g/'))),
            zip(cycle(('A:min',)), tuple(directory.path for directory in
                                         os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/am/'))),
            zip(cycle(('B:min',)), tuple(directory.path for directory in
                                         os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/bm/'))),
            zip(cycle(('D:min',)), tuple(directory.path for directory in
                                         os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/dm/'))),
            zip(cycle(('E:min',)), tuple(directory.path for directory in
                                         os.scandir('/Users/matusskerlik/Downloads/jim2012Chords/Guitar_Only/em/'))),
        ))
        self.assertTrue(result)

    # TODO
    def test_save(self):
        pass

    # TODO
    def test_load(self):
        pass


if __name__ == '__main__':
    unittest.main()
