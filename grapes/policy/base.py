# pylint: disable=missing-docstring,invalid-name

from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def assign(self, grapevine, x, y):
        pass

    @abstractmethod
    def select(self, grapevine):
        pass
