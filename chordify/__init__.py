from .annotation import LabParser
from .audio_processing import AudioProcessing, PathLoadStrategy, CQTExtractionStrategy, HCDFSegmentationStrategy, \
    BeatSegmentationStrategy, DefaultSegmentationStrategy, VectorSegmentationStrategy, HPSSChromaStrategy, \
    SmoothingChromaStrategy
from .chord_recognition import TemplatePredictStrategy, HarmonicPredictStrategy
from .learn import SVCScikitLearnStrategy
from .app import Transcript
