# pylint: disable=missing-docstring,invalid-name


class GrapeError(Exception):
    pass


class InvalidPoint(GrapeError):
    def __init__(self, point, size):
        message = 'Invalid point {} on board of size {}'.format(point, size)
        super().__init__(message)

        self.point = point
        self.size = size


class FilledPoint(GrapeError):
    def __init__(self, point, seed):
        message = 'Point {} on board filled by {}'.format(point, seed)
        super().__init__(message)

        self.point = point
        self.seed = seed


class EmptyPoint(GrapeError):
    def __init__(self, point):
        message = 'Point {} on board is empty'.format(point)
        super().__init__(message)

        self.point = point


class IllegalMove(GrapeError):
    def __init__(self, point):
        message = 'Illegal move at {}'.format(point)
        super().__init__(message)

        self.point = point
