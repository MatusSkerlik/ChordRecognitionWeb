import unittest

from decoder import Stream, TerminatedStream


class TestStream(unittest.TestCase):

    def testGet(self):
        stream = Stream("12345")
        self.assertEqual(stream.get(), "1")
        self.assertEqual(stream.get(), "2")
        self.assertEqual(stream.get(), "3")
        self.assertEqual(stream.get(), "4")
        self.assertEqual(stream.get(), "5")

        with self.assertRaises(IndexError):
            stream.get()

    def testMatch(self):
        stream = Stream("(12345)")
        with self.assertRaises(IndexError):
            stream.match("/")

        stream = Stream("(12345)")
        stream.match(")")
        self.assertEqual(stream.get(), ")")

    def testSlice(self):
        stream = Stream("(12345)")
        stream.get()
        stream.match(")")
        self.assertEqual(stream.slice(), "12345")

    def testUndo(self):
        stream = Stream("12345")
        self.assertEqual(stream.get(), "1")
        self.assertEqual(stream.get(), "2")
        self.assertEqual(stream.get(), "3")
        self.assertEqual(stream.get(), "4")
        self.assertEqual(stream.get(), "5")
        stream.undo()
        self.assertEqual(stream.get(), "5")
        stream.undo()
        stream.undo()
        self.assertEqual(stream.get(), "4")
        stream.undo()
        stream.undo()
        self.assertEqual(stream.get(), "3")
        stream.undo()
        stream.undo()
        self.assertEqual(stream.get(), "2")
        stream.undo()
        stream.undo()
        self.assertEqual(stream.get(), "1")
        stream.undo()

        with self.assertRaises(IndexError):
            stream.undo()


class TestTerminatedStream(unittest.TestCase):

    def testEof(self):
        stream = TerminatedStream("12345")
        self.assertEqual(stream.get(), "1")
        self.assertEqual(stream.get(), "2")
        self.assertEqual(stream.get(), "3")
        self.assertEqual(stream.get(), "4")
        self.assertEqual(stream.get(), "5")

        self.assertTrue(stream.eof())
        stream.undo()
        self.assertFalse(stream.eof())
        stream.get()
        self.assertTrue(stream.eof())


if __name__ == '__main__':
    unittest.main()
