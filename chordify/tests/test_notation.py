import unittest

import numpy
from lark.exceptions import UnexpectedInput

from chordify.notation import parse, _Vector, Chord


class TestNotation(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(UnexpectedInput):
            parse('')
            parse(' ')
            parse('Q')
            parse('L')
            parse('A:')

            parse('A:(0, 1)')

            parse('A:(')
            parse('A:)')
            parse('A:()')
            parse('A:()')

            parse('A/A')
            parse('A/B')
            parse('A/D')
            parse('A/*5')
            parse('A/06')

            parse('A:mi')
            parse('A:min(')
            parse('A:min)')
            parse('A:min()')

            parse('A:min/')
            parse('A:min/0')
            parse('A:min/01')
            parse('A:min/005')

            parse('A:min(/')
            parse('A:min)/')
            parse('A:min()/')

            parse('A:min/A')

            parse('A:hol')
            parse('A:dim8')
            parse('A:dim2')
            parse('A:di m')
            parse('A:d im')

    def test_pass(self):
        self.assertEqual(parse('A:(*1,*3,*5)'), ('A', None, None))

        self.assertEqual(parse('A'), ('A', ('1', '3', '5'), None))
        self.assertEqual(parse('B'), ('B', ('1', '3', '5'), None))
        self.assertEqual(parse('C'), ('C', ('1', '3', '5'), None))
        self.assertEqual(parse('D'), ('D', ('1', '3', '5'), None))
        self.assertEqual(parse('E'), ('E', ('1', '3', '5'), None))
        self.assertEqual(parse('F'), ('F', ('1', '3', '5'), None))
        self.assertEqual(parse('G'), ('G', ('1', '3', '5'), None))
        self.assertEqual(parse('Abb'), ('Abb', ('1', '3', '5'), None))
        self.assertEqual(parse('A#'), ('A#', ('1', '3', '5'), None))

        self.assertEqual(parse('A#/5'), ('A#', ('1', '3', '5'), '5'))
        self.assertEqual(parse('Bbb/1'), ('Bbb', ('1', '3', '5'), '1'))

        self.assertEqual(parse('A#:min'), ('A#', ('1', 'b3', '5'), None))
        self.assertEqual(parse('B:min'), ('B', ('1', 'b3', '5'), None))

        self.assertEqual(parse('Cb:maj'), ('Cb', ('1', '3', '5'), None))
        self.assertEqual(parse('D#:maj'), ('D#', ('1', '3', '5'), None))

        self.assertEqual(parse('A#:min/5'), ('A#', ('1', 'b3', '5'), '5'))
        self.assertEqual(parse('B:min/3'), ('B', ('1', 'b3', '5'), '3'))

        self.assertEqual(parse('Cb:maj(*3)/9'), ('Cb', ('1', '5'), '9'))
        self.assertEqual(parse('D#:maj(6)/10'), ('D#', ('1', '3', '5', '6'), '10'))

        self.assertEqual(parse('Cb:maj(*3,b9,3,3)/9'), ('Cb', ('1', '5', 'b9', '3'), '9'))
        self.assertEqual(parse('Cb:maj(*3,b9,3,3)/bb9'), ('Cb', ('1', '5', 'b9', '3'), 'bb9'))

        self.assertEqual(parse('C:maj(*3,*1)/bb9'), ('C', ('5',), 'bb9'))
        self.assertEqual(parse('C:maj(*3,*1)/bb9'), ('C', ('5',), 'bb9'))


class TestVector(unittest.TestCase):
    def test_dot(self):
        X = _Vector((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
        Y = _Vector((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(X @ Y, 12.)
        self.assertEqual(Y @ X, 12.)

        X = _Vector((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        Y = _Vector((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12))
        self.assertEqual(X @ Y, 144.)
        self.assertEqual(Y @ X, 144.)


class TestChord(unittest.TestCase):
    def test_dot(self):
        pass

    def test_vector(self):
        numpy.testing.assert_array_equal(Chord("C:maj")._vector, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("C:min")._vector, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("C:maj7")._vector, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])

        numpy.testing.assert_array_equal(Chord("C:maj(*5)")._vector, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("C:maj(*5,6)")._vector, [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
        numpy.testing.assert_array_equal(Chord("C:maj(*5,*1)")._vector, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        numpy.testing.assert_array_equal(Chord("C:(*5,*1)")._vector, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        numpy.testing.assert_array_equal(Chord("C##")._vector, [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])
        numpy.testing.assert_array_equal(Chord("D")._vector, [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])

        numpy.testing.assert_array_equal(Chord("C")._vector, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("Dbb")._vector, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

        numpy.testing.assert_array_equal(Chord("C:(b3,b5)")._vector, [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("Dbb:(b3,b5)")._vector, [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0])

        numpy.testing.assert_array_equal(Chord("Ebbbb:(8)")._vector, [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

        numpy.testing.assert_array_equal(Chord("D:min")._vector, [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        numpy.testing.assert_array_equal(Chord("B:min")._vector, [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        numpy.testing.assert_array_equal(Chord("B:maj")._vector, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        numpy.testing.assert_array_equal(Chord("B:maj(*1)")._vector, [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(Chord("B:maj(*1,6)")._vector, [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])

    def test_str(self):
        self.assertEqual(str(Chord("A:min(8)/7")), "A:min(8)/7")
        self.assertEqual(str(Chord("A:min")), "A:min")


if __name__ == '__main__':
    unittest.main()
