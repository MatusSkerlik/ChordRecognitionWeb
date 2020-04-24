import unittest

from decoder import modifier, Stream


class TestModifier(unittest.TestCase):

    def testParse(self):
        self.assertEqual(modifier(Stream("b")), "b")
        self.assertEqual(modifier(Stream("#")), "#")

    def testRaise(self):
        with self.assertRaises(NameError):
            modifier(Stream("U"))
            modifier(Stream("R"))
            modifier(Stream("r"))
            modifier(Stream("2"))
            modifier(Stream("3"))
            modifier(Stream(":"))
            modifier(Stream("("))

        with self.assertRaises(IndexError):
            modifier(Stream(""))


if __name__ == '__main__':
    unittest.main()
