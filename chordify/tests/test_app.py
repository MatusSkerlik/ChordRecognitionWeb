import os
import tempfile
import unittest
from itertools import chain, cycle
from typing import Callable, Iterable

import librosa

from chordify.app import TranscriptBuilder, Transcript, LearnerBuilder, LearnedStrategy, _LearnedStrategy
from chordify.audio_processing import VectorSegmentationStrategyFactory, CQTExtractionStrategyFactory, \
    PathLoadStrategyFactory, DefaultChromaStrategyFactory
from chordify.chord_recognition import PredictStrategy, TemplatePredictStrategyFactory
from chordify.notation import Chord


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


class TestLearnStrategy(unittest.TestCase):
    def test_proxy_behaviour(self):
        strategy = _LearnedStrategy(TemplatePredictStrategyFactory((Chord('A'), Chord('C')))(None), os.getcwd())

        self.assertIsInstance(strategy.predict, Callable)
        self.assertIsInstance(strategy._predict_strategy, PredictStrategy)
        self.assertIsInstance(strategy._model_output_dir, str)

        with self.assertRaises(AttributeError):
            getattr(strategy, 'foo')
            getattr(strategy, 'bar')
            getattr(strategy, 'baz')


class TestLearner(unittest.TestCase):
    MODEL_FILENAME = 'model.pickle'
    MODEL_SAVE_DIR = tempfile.gettempdir()

    # TODO configure logger
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        if os.path.exists(os.path.join(self.MODEL_SAVE_DIR, self.MODEL_FILENAME)):
            os.remove(os.path.join(self.MODEL_SAVE_DIR, self.MODEL_FILENAME))

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
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\a\\'))),
            zip(cycle('C'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\c\\'))),
            zip(cycle('D'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\d\\'))),
            zip(cycle('E'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\e\\'))),
            zip(cycle('F'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\f\\'))),
            zip(cycle('G'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\g\\'))),
            zip(cycle(('A:min',)), tuple(directory.path for directory in
                                         os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\am\\'))),
            zip(cycle(('B:min',)), tuple(directory.path for directory in
                                         os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\bm\\'))),
            zip(cycle(('D:min',)), tuple(directory.path for directory in
                                         os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\dm\\'))),
            zip(cycle(('E:min',)), tuple(directory.path for directory in
                                         os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\em\\'))),
        ))
        self.assertTrue(isinstance(result, PredictStrategy))
        self.assertTrue(isinstance(result, LearnedStrategy))

    def test_save_load(self):
        app = LearnerBuilder().setModelOutputDir(self.MODEL_SAVE_DIR).build()

        result = app.from_samples(chain(
            zip(cycle('A'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\a\\'))),
            zip(cycle('C'), tuple(directory.path for directory in
                                  os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\c\\')))
        ))

        result.save('model.pickle')

        self.assertTrue(os.path.exists(os.path.join(self.MODEL_SAVE_DIR, self.MODEL_FILENAME)))

        result = app.load(self.MODEL_FILENAME)

        self.assertIsInstance(result, LearnedStrategy)

    def test_load_error_if_no_file_exists(self):
        app = LearnerBuilder().build()

        with self.assertRaises(IOError):
            app.load(self.MODEL_FILENAME)


class TestLearnerTranscriptInteraction(unittest.TestCase):

    def test_save_load_and_use_in_transcript(self):
        app = LearnerBuilder().build()

        result = app.from_samples(chain(
            zip(cycle((Chord('A'),)), tuple(directory.path for directory in
                                            os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\a\\'))),
            zip(cycle((Chord('C'),)), tuple(directory.path for directory in
                                            os.scandir('D:\\download\\jim2012Chords\\Guitar_Only\\c\\')))
        ))

        app = TranscriptBuilder().setLearnedStrategy(result).build()
        result = app.from_audio(librosa.util.example_audio_file())

        self.assertIsInstance(result, Iterable)

        for row in result:
            self.assertIsInstance(row[0], float)
            self.assertIsInstance(row[1], Chord)


if __name__ == '__main__':
    unittest.main()
