from typing import Tuple, Sequence, Union

import numpy
from lark.lark import Lark
from lark.tree import Tree
from lark.visitors import Transformer, v_args

_Parser = Lark(r"""
    chord: pitchname ":" shorthand components? bass?
        | pitchname ":" components bass?
        | pitchname bass?
        | NONE
    pitchname: NATURAL MODIFIER*
    NATURAL: "A" | "B" | "C" | "D" | "E" | "F" | "G"
    MODIFIER: "b" | "#"
    components: "(" [STAR] interval ("," [STAR] interval)* ")"
    bass: "/" interval
    interval: (DIGIT+ | DIGIT+ ZERO) | MODIFIER* (DIGIT+ | DIGIT+ ZERO)
    DIGIT: "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    ZERO: "0"
    !shorthand: "maj" | "min" | "dim" | "aug" | "maj7" | "min7" | "7" 
                | "dim7" | "hdim7" | "minmaj7" | "maj6" | "min6" | "9" 
                | "maj9" | "min9" | "sus2" | "sus4"
    NONE: "N"
    STAR: "*"
""", start='chord', parser='lalr')


class _Notation(Transformer):

    def __init__(self, visit_tokens=True):
        super().__init__(visit_tokens)
        self._shorthand = False
        self._components = False
        self._bass = False

    @v_args(inline=True)
    def chord(self, pitchname, shorthand=None, components=None, bass=None):
        if not self._shorthand and self._components:
            if self._bass:
                bass = components
            components = shorthand
            shorthand = None
        elif self._shorthand and not self._components and self._bass:
            bass = components
            components = None
        elif not self._components and self._bass:
            bass = shorthand
            shorthand = None

        def modify_shorthand(short=None, modifier=None, default_interval=('1', '3', '5')):
            if short is not None and modifier is None:
                return short
            if short is None and modifier is not None:
                short = default_interval + tuple(
                    s for s in modifier if s[0] != '*')  # no shorthand, add default interval
                modifier = tuple(s for s in modifier if s[0] == '*')
            if short is not None and modifier is not None:
                _short = ()

                _remove = tuple(s[1:] for s in modifier if s[0] == '*')
                for s in short:
                    if s not in _remove:
                        _short += (s,)
                for s in tuple(s for s in modifier if s[0] != '*'):
                    if s not in _short:
                        _short += (s,)
                return _short
            return default_interval  # no shorthand or component, return default interval

        _components = modify_shorthand(shorthand, components)
        _components = _components if bool(_components) else None
        if pitchname == 'N':
            return 'N', _components, bass
        return pitchname, _components, bass

    def pitchname(self, children):
        return ''.join(children)

    @v_args(inline=True)
    def shorthand(self, token):
        self._shorthand = True
        if token == 'maj':
            return "1", "3", "5"
        elif token == 'min':
            return "1", "b3", "5"
        elif token == 'dim':
            return "1", "b3", "b5"
        elif token == 'aug':
            return "1", "3", "#5"
        elif token == 'maj7':
            return "1", "3", "5", "7"
        elif token == 'min7':
            return "1", "b3", "5", "b7"
        elif token == '7':
            return "1", "3", "5", "b7"
        elif token == 'dim7':
            return "1", "b3", "b5", "bb7"
        elif token == 'hdim7':
            return "1", "b3", "b5", "b7"
        elif token == 'minmaj7':
            return "1", "b3", "5", "7"
        elif token == 'maj6':
            return "1", "3", "5", "6"
        elif token == 'min6':
            return "1", "b3", "5", "6"
        elif token == '9':
            return "1", "3", "5", "b7", "9"
        elif token == 'maj9':
            return "1", "3", "5", "7", "9"
        elif token == 'min9':
            return "1", "b3", "5", "b7", "9"
        elif token == 'sus2':
            return "1", "2", "5"
        elif token == 'sus4':
            return "1", "4", "5"
        raise ValueError

    @v_args(inline=True)
    def bass(self, tree):
        self._bass = True
        return ''.join(tree.children)

    def components(self, children):
        self._components = True
        star = False
        result = tuple()
        for elm in children:
            if isinstance(elm, Tree):
                result += (('*' + ''.join(elm.children)) if star else ''.join(elm.children),)
                star = False
            else:  # star
                if elm == '*':
                    star = True
                else:
                    raise ValueError
        return result


class _Vector(numpy.ndarray):

    def __new__(cls, vector: Tuple[float, float, float, float, float, float,
                                   float, float, float, float, float, float]):
        try:
            obj = numpy.zeros(12, numpy.float32)
            obj += vector
            return obj
        except ValueError:
            raise ValueError("Vector must have 12 columns.")

    def __matmul__(self, other):
        return numpy.dot(self, other)

    def __rmatmul__(self, other):
        return numpy.dot(other, self)


class Chord:
    _vector: _Vector
    _str: str

    def __new__(cls, string: str):
        pitchname, components, bass = _Notation().transform(_Parser.parse(string))

        def semitones(hop):
            return {0: 12, 1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}[hop % 8]

        if pitchname == 'N':
            raise ValueError("Illegal chord")
        _index = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B').index(pitchname[0])
        _raise = pitchname.count('#') - pitchname.count('b')

        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if components:
            for interval in components:
                _iindex = semitones(int(interval[(interval.count('#') + interval.count('b')):]))
                _iraise = interval.count('#') - interval.count('b')
                _ibase = abs(_index + _raise + _iindex + _iraise) % 12
                vector[_ibase] = 1

        obj = super().__new__(cls)

        obj.__dict__['_vector'] = vector
        obj.__dict__['_str'] = string

        return obj

    def __matmul__(self, other):
        return self._vector @ other

    def __rmatmul__(self, other):
        return self._vector @ other

    def __str__(self):
        return self._str

    def __eq__(self, other):
        return numpy.mean(self == other)

    def __ne__(self, other):
        return numpy.mean(self != other)


def parse(string: str) -> Tuple[str, Union[Sequence, None], Union[str, None]]:
    """ Returns pitchname, components, bass """
    return _Notation().transform(_Parser.parse(string))
