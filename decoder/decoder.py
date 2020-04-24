from abc import ABC, abstractmethod
from functools import wraps


def index_safe(func):
    @wraps(func)
    def wrapper(stream):
        try:
            return func(stream)
        except IndexError:
            return False

    return wrapper


class IntegerStack(ABC):

    @property
    @abstractmethod
    def top(self) -> int:
        pass

    @abstractmethod
    def push(self, value: int):
        pass

    @abstractmethod
    def pop(self) -> int:
        pass


class Stack:

    def __init__(self) -> None:
        super().__init__()
        self._content = []

    @property
    def top(self):
        return self.push(self.pop())

    def push(self, value):
        self._content.append(value)
        return value

    def pop(self):
        try:
            return self._content.pop()
        except IndexError:
            raise IndexError


class StreamPointer(Stack, IntegerStack):

    def __init__(self) -> None:
        super().__init__()
        self.push(0)


class Stream:

    def __init__(self, stream: str) -> None:
        super().__init__()
        self._pointer = StreamPointer()
        self._stream = stream

    def get(self):
        result = self._stream[self._pointer.top]
        self._pointer.push(1 + self._pointer.top)
        return result

    def slice(self):
        try:
            stop = self._pointer.pop()
        except IndexError:
            return ""

        try:
            start = self._pointer.pop()
        except IndexError:
            self._pointer.push(stop)
            return self._stream[0:stop]

        self._pointer.push(start)
        self._pointer.push(stop)

        return self._stream[start:stop]

    def match(self, with_char: str) -> int:

        if with_char is None:
            self._pointer.push(len(self._stream))
            return len(self._stream) - 1

        end = self._stream[self._pointer.top + 1:].find(with_char)

        if end == -1:
            raise IndexError

        end += self._pointer.top + 1
        if end > self._pointer.top:
            self._pointer.push(end)
            return end
        raise IndexError

    def undo(self):
        if self._pointer.top == 0:
            raise IndexError
        self._pointer.pop()


class TerminatedStream(Stream):

    def eof(self):
        try:
            self.get()
            self.undo()
            return False
        except IndexError:
            return True


class Chord:

    def __init__(self, root, bass, components) -> None:
        super().__init__()
        self.root = root
        self.bass = bass
        self.components = components

    def __str__(self) -> str:
        return "%s:(%s)/%s" % (self.root, self.components, self.bass)


def shorthand(stream):
    try:
        stream.match("(")
    except IndexError:
        try:
            stream.match("/")
        except IndexError:
            stream.match(None)

    interval_list = None
    _slice = stream.slice()
    if _slice == "maj":
        interval_list = ["1", "3", "5"]
    elif _slice == "min":
        interval_list = ["1", "b3", "5"]
    elif _slice == "dim":
        interval_list = ["1", "b3", "b5"]
    elif _slice == "aug":
        interval_list = ["1", "3", "#5"]
    elif _slice == "maj7":
        interval_list = ["1", "3", "5", "7"]
    elif _slice == "min7":
        interval_list = ["1", "b3", "5", "b7"]
    elif _slice == "7":
        interval_list = ["1", "3", "5", "b7"]
    elif _slice == "dim7":
        interval_list = ["1", "b3", "b5", "bb7"]
    elif _slice == "hdim7":
        interval_list = ["1", "b3", "b5", "b7"]
    elif _slice == "minmaj7":
        interval_list = ["1", "b3", "5", "7"]
    elif _slice == "maj6":
        interval_list = ["1", "3", "5", "6"]
    elif _slice == "min6":
        interval_list = ["1", "b3", "5", "6"]
    elif _slice == "9":
        interval_list = ["1", "3", "5", "b7", "9"]
    elif _slice == "maj9":
        interval_list = ["1", "3", "5", "7", "9"]
    elif _slice == "min9":
        interval_list = ["1", "b3", "5", "b7", "9"]
    elif _slice == "sus2":
        interval_list = ["1", "2", "5"]
    elif _slice == "sus4":
        interval_list = ["1", "4", "5"]

    if interval_list is not None:
        return interval_list
    stream.undo()
    return False


def natural(stream):
    _natural = stream.get()
    if _natural in ("C", "D", "E", "F", "G", "A", "B"):
        return _natural
    raise NameError


def modifier(stream):
    _modifier = stream.get()
    if _modifier in ("b", "#"):
        return _modifier
    raise NameError


def digit(stream):
    _digit = stream.get()
    if _digit != "0" and _digit.isnumeric():
        return _digit
    raise NameError


def pitchname(stream):
    _natural = natural(stream)
    while True:
        try:
            _modifier = modifier(stream)
            _natural = _natural + _modifier
        except IndexError:
            break
        except NameError:
            stream.undo()
            break
    return _natural


def degree(stream):
    def _zero(stream):
        try:
            _zero = stream.get()
            if _zero == "0":
                return _zero
            else:
                raise NameError
        except IndexError:
            raise NameError

    _digit = digit(stream)
    try:
        return _digit + degree(stream)
    except NameError:
        stream.undo()
        try:
            return _digit + _zero(stream)
        except NameError:
            stream.undo()
            return _digit
    except IndexError:
        return _digit


def __interval(_stream):
    try:
        try:
            return modifier(_stream) + __interval(_stream)
        except NameError:
            _stream.undo()
            return degree(_stream)
    except IndexError:
        raise NameError


def interval(stream):
    _next_char = stream.get()
    if _next_char == "/":
        try:
            stream.get()
        except IndexError:
            raise IndexError
        stream.undo()
        return __interval(stream)
    stream.undo()
    return False


def ilist(stream):
    _next_char = stream.get()

    if _next_char == "(":
        try:
            stream.match(")")
            stream1 = Stream(stream.slice())

            def _ilist(_stream):
                _result = []
                try:
                    _char = _stream.get()
                except IndexError:
                    raise NameError
                if _char == "*":
                    _char += __interval(_stream)
                else:
                    _stream.undo()
                    _char = __interval(_stream)
                _result.append(_char)
                try:
                    if _stream.get() == ",":
                        return _result + _ilist(_stream)
                    else:
                        raise NameError
                except IndexError:
                    return _result

            result = _ilist(stream1)
            if stream.get() == ")":
                return result
            raise IndexError
        except IndexError:
            raise IndexError
    stream.undo()
    return False


def erased(stream):
    if not stream.eof():
        raise NameError


def decrypt(stream: TerminatedStream):
    _shorthand = None
    _ilist = None
    _interval = None
    _pitchname = None
    _next_char = None

    if stream.eof():
        return None

    try:
        _pitchname = pitchname(stream)
    except NameError:
        stream.undo()
        if stream.get() == "N":
            try:
                stream.get()
            except IndexError:
                return "N", _shorthand, _ilist, _interval
            raise NameError
        raise NameError

    if stream.eof():
        return _pitchname, _shorthand, _ilist, _interval

    _next_char = stream.get()

    if _next_char == ":" and not stream.eof():

        _shorthand = shorthand(stream)
        if _shorthand is False:
            _ilist = ilist(stream)
            if _ilist:
                _interval = index_safe(interval)(stream)
                erased(stream)
            else:
                raise NameError
        else:
            _ilist = index_safe(ilist)(stream)
            _interval = index_safe(interval)(stream)
            erased(stream)
    else:
        stream.undo()
        _interval = interval(stream)
        erased(stream)

    return _pitchname, _shorthand, _ilist, _interval
