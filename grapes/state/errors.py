# pylint: disable=missing-docstring,invalid-name


class GrapeError(Exception):
    pass


class IllegalMove(GrapeError):
    def __init__(self, point):
        message = 'Illegal move at point {}'.format(point)
        super().__init__(message)
