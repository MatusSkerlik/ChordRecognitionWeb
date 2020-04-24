import unittest

from decoder import natural, Stream


class TestNatural(unittest.TestCase):

    def testParse(self):
        self.assertEqual(natural(Stream("A")), "A")
        self.assertEqual(natural(Stream("B")), "B")
        self.assertEqual(natural(Stream("C")), "C")
        self.assertEqual(natural(Stream("D")), "D")
        self.assertEqual(natural(Stream("E")), "E")
        self.assertEqual(natural(Stream("F")), "F")
        self.assertEqual(natural(Stream("G")), "G")

    def testRaise(self):
        with self.assertRaises(NameError):
            natural(Stream("U"))
            natural(Stream("R"))
            natural(Stream("r"))
            natural(Stream("2"))
            natural(Stream("3"))
            natural(Stream(":"))
            natural(Stream("("))

        with self.assertRaises(IndexError):
            natural(Stream(""))


if __name__ == '__main__':
    unittest.main()
