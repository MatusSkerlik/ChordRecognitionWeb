import unittest

import librosa

from chordify.app import Builder


class MyTestCase(unittest.TestCase):
    def test_app(self):
        app = Builder.default()
        result = app.from_audio(librosa.util.example_audio_file())
        self.assertTrue(result)
        print(result)


if __name__ == '__main__':
    unittest.main()
