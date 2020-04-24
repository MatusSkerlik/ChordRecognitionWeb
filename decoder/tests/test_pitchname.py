import unittest

from decoder import pitchname, Stream


class TestPitchName(unittest.TestCase):

    def testParse(self):
        self.assertEqual(pitchname(Stream("A")), "A")
        self.assertEqual(pitchname(Stream("Bbb")), "Bbb")
        self.assertEqual(pitchname(Stream("Cb#")), "Cb#")
        self.assertEqual(pitchname(Stream("D/")), "D")
        self.assertEqual(pitchname(Stream("E(1,2,3)")), "E")
        self.assertEqual(pitchname(Stream("F")), "F")
        self.assertEqual(pitchname(Stream("G")), "G")

    def testPointer(self):
        stream = Stream("E(1,2,3)")
        self.assertEqual(pitchname(stream), "E")
        self.assertEqual(stream.get(), "(")

        stream = Stream("E####(1,2,3)")
        self.assertEqual(pitchname(stream), "E####")
        self.assertEqual(stream.get(), "(")

        stream = Stream("E####/5")
        self.assertEqual(pitchname(stream), "E####")
        self.assertEqual(stream.get(), "/")

        stream = Stream("E")
        self.assertEqual(pitchname(stream), "E")

        with self.assertRaises(IndexError):
            stream.get()

    def testRaise(self):
        with self.assertRaises(NameError):
            pitchname(Stream("#G"))
            pitchname(Stream("bG"))
            pitchname(Stream("p"))
            pitchname(Stream("g"))
            pitchname(Stream("2"))

        with self.assertRaises(IndexError):
            pitchname(Stream(""))


if __name__ == '__main__':
    unittest.main()
