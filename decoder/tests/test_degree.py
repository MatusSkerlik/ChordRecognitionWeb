import unittest

from decoder import degree, Stream


class TestDegree(unittest.TestCase):

    def testParse(self):
        self.assertEqual(degree(Stream("123")), "123")
        self.assertEqual(degree(Stream("1230123")), "1230")

    def testPointer(self):
        stream = Stream("123(")
        degree(stream)
        self.assertEqual(stream.get(), "(")

        stream = Stream("123ABC")
        degree(stream)
        self.assertEqual(stream.get(), "A")
        
    def testRaise(self):
        with self.assertRaises(NameError):
            degree(Stream("bb123"))
            degree(Stream("a123"))
            degree(Stream("(123"))
            degree(Stream("/123"))

        with self.assertRaises(IndexError):
            degree(Stream(""))


if __name__ == '__main__':
    unittest.main()
