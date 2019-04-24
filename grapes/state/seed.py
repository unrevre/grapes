# pylint: disable=missing-docstring,invalid-name


EMPTY =  0
BLACK =  1
WHITE = -1

ABBR = [' ', 'b', 'w']


def inverse(seed):
    return seed * -1


def abbr(seed):
    return ABBR[seed]
