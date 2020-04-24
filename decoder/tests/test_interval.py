import unittest

from decoder import interval, Stream


class TestInterval(unittest.TestCase):

    def testParse(self):
        self.assertEqual(interval(Stream("/bb123/AV")), "bb123")
        self.assertEqual(interval(Stream("/b1230123")), "b1230")

    def testPointer(self):
        stream = Stream("/bb123(")
        interval(stream)
        self.assertEqual(stream.get(), "(")

        stream = Stream("/bb12301(")
        interval(stream)
        self.assertEqual(stream.get(), "1")

    def testRaise(self):
        with self.assertRaises(NameError):
            interval(Stream("bb123"))
            interval(Stream("a123"))
            interval(Stream("(123"))
            interval(Stream("123"))
            interval(Stream("/A"))
            interval(Stream("/(1,2)"))

        with self.assertRaises(IndexError):
            interval(Stream("/"))
            interval(Stream(""))


if __name__ == '__main__':
    unittest.main()
