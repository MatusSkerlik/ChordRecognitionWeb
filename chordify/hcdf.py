from itertools import tee

import librosa
import numpy as np
from numpy.linalg import norm
from scipy.signal import find_peaks


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    next(b, None)
    return zip(a, b)


def hcdf(frames: np.ndarray):
    """ Harmonic change detect function """
    _tonnetz = librosa.feature.tonnetz(chroma=frames)

    l2_seq = list()
    for tv0, tv1 in _pairwise(_tonnetz.T):
        d = norm(tv0 - tv1, ord=2)  # euclidean distance
        l2_seq.append(d)
    return np.array(l2_seq)


def get_segments(frames: np.ndarray, prominence=None):
    _hcdf = hcdf(frames)
    _peaks, _ = find_peaks(_hcdf, prominence)
    return np.array_split(frames, _peaks, axis=1), \
           np.asarray(tuple(_peaks) + (np.size(frames, axis=1) - 1,))
