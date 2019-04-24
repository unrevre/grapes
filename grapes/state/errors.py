# pylint: disable=missing-docstring,invalid-name


class GrapeError(Exception):
    pass


class InvalidPoint(GrapeError):
    def __init__(self, point, size):
        message = 'Invalid point {} on board of size {}'.format(point, size)
        super().__init__(message)


class FilledPoint(GrapeError):
    def __init__(self, point, seed):
        message = 'Point {} on board filled by {}'.format(point, seed)
        super().__init__(message)
