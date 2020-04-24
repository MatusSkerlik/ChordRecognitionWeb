import unittest

from decoder import shorthand, Stream


class TestShorthand(unittest.TestCase):

    def testParse(self):
        self.assertTrue(shorthand(Stream("min")))
        self.assertTrue(shorthand(Stream("maj")))
        self.assertTrue(shorthand(Stream("maj7")))
        self.assertTrue(shorthand(Stream("7")))
        self.assertTrue(shorthand(Stream("min6")))

    def testPointer(self):
        stream = Stream("maj(1,2,3)")
        shorthand(stream)
        self.assertEqual(stream.get(), "(")

        stream = Stream("maj/5")
        shorthand(stream)
        self.assertEqual(stream.get(), "/")

        stream = Stream("maj")
        shorthand(stream)
        with self.assertRaises(IndexError):
            stream.get()

    def testRaise(self):
        self.assertFalse(shorthand(Stream("/5")))
        self.assertFalse(shorthand(Stream("(1,2)")))
        self.assertFalse(shorthand(Stream("ma")))

        # with self.assertRaises(IndexError):
        #     shorthand(Stream(""))


if __name__ == '__main__':
    unittest.main()
