import unittest

from decoder import TerminatedStream, decrypt


class TestDecrypt(unittest.TestCase):

    def testParse(self):
        self.assertTrue(decrypt(TerminatedStream("A")))
        self.assertTrue(decrypt(TerminatedStream("A:min")))
        self.assertTrue(decrypt(TerminatedStream("A:maj/5")))
        self.assertTrue(decrypt(TerminatedStream("A:maj(*1,6,7)/5")))

        self.assertIsNone(decrypt(TerminatedStream("")))

    def testRaise(self):
        with self.assertRaises(NameError):
            decrypt(TerminatedStream("H"))
            decrypt(TerminatedStream("A/A"))
            decrypt(TerminatedStream("Ad1"))
            decrypt(TerminatedStream("A/(1,2)"))
            decrypt(TerminatedStream("Cb/G"))
            decrypt(TerminatedStream("Cb(1,/G"))
            decrypt(TerminatedStream("Cb(1,)/G"))
            decrypt(TerminatedStream("Cb(1,2)/5a"))
            decrypt(TerminatedStream("zCb(1,2)/5a"))
            decrypt(TerminatedStream("C/ "))
            decrypt(TerminatedStream("Ca"))
            decrypt(TerminatedStream("C:(1,2,3) "))
            decrypt(TerminatedStream("C:(1,2,3)d"))
            decrypt(TerminatedStream("C:(1b,2,3)"))
            decrypt(TerminatedStream("C(1,##2,bbb3)/"))

        with self.assertRaises(IndexError):
            decrypt(TerminatedStream("C:(1,##2,bbb3)/"))
            decrypt(TerminatedStream("C:("))
            decrypt(TerminatedStream("C:()"))
            decrypt(TerminatedStream("C:()/"))
            decrypt(TerminatedStream("C:"))
            decrypt(TerminatedStream("C/"))
            decrypt(TerminatedStream(""))


if __name__ == '__main__':
    unittest.main()
