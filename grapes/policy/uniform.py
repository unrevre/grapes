# pylint: disable=missing-docstring,invalid-name

import random

from grapes.policy.base import Policy


class Uniform(Policy):
    def __init__(self):
        super().__init__()

        self.desc = 'uniform'

    def assign(self, grapevine, x, y):
        return 1.0

    def select(self, grapevine):
        moves = list(grapevine.buds())
        return random.choice(moves)
