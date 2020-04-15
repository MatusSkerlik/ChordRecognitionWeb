import os
import tempfile

from chordify import Transcript
from chordify.app import Learn

Transcript().from_audio(os.path.join(tempfile.gettempdir(), "axel_f.mp3"))
# Transcript({"DEBUG": True}).from_audio(os.path.join(tempfile.gettempdir(), "axel_f.mp3"))
