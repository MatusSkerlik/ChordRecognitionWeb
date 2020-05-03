import random


def random_str(length=48):
    random_set = 'qwertzuioopasdfghjklyxcvbnmQWERTZUIOPASDFGHJKLYXCVBNM1234567890'
    return ''.join(random.sample(random_set, length))
