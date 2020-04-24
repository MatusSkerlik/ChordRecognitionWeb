import unittest

from decoder import ilist, Stream


class TestIList(unittest.TestCase):

    def testParse(self):
        self.assertEqual(ilist(Stream("(1)")), ["1"])
        self.assertEqual(ilist(Stream("(b1)")), ["b1"])
        self.assertEqual(ilist(Stream("(10)")), ["10"])
        self.assertEqual(ilist(Stream("(1,b11)")), ["1", "b11"])
        self.assertEqual(ilist(Stream("(bb1,1,5)")), ["bb1", "1", "5"])

    def testPointer(self):
        stream = Stream("(1,2,3)/")
        self.assertEqual(ilist(stream), ["1", "2", "3"])
        self.assertEqual(stream.get(), "/")

        stream = Stream("(1,bbbbbbbb###b2,####3)/")
        self.assertEqual(ilist(stream), ["1", "bbbbbbbb###b2", "####3"])
        self.assertEqual(stream.get(), "/")

    def testFalse(self):
        self.assertFalse(ilist(Stream("A(1,2)")))
        self.assertFalse(ilist(Stream("1,2)")))

    def testRaise(self):
        with self.assertRaises(NameError):
            ilist(Stream("(1,)"))
            ilist(Stream("(1,2,3"))
            ilist(Stream("(1,2,)"))
            ilist(Stream("(1,)"))

        # with self.assertRaises(IndexError):
        #     ilist(Stream(""))


if __name__ == '__main__':
    unittest.main()
