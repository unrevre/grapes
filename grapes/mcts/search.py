# pylint: disable=missing-docstring,invalid-name

import numpy as np

import grapes.mcts.tree as Tree


class Search:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.history = []

    def iter(self, node):
        policy, value = self.model.eval(node)

        node.expand(policy)
        node.update(value)

    def search(self, n, alpha, frac):
        if not self.root.children:
            self.iter(self.root)

        self.root.smear(alpha, frac)

        for _ in range(n):
            self.iter(self.root.descend())

    def next(self, temp):
        self.history.append(self.root.state)

        def select(n, temp):
            if temp is None:
                return np.argmax(n)

            prob = np.power(n, 1.0 / temp)
            prob = prob / np.sum(prob)

            return np.random.choice(n.size, p=prob)

        _, (_, n) = self.history[-1]

        self.root = self.root.children[select(n, temp)]
        self.root.inherit()
        self.root.parent = None

    def move(self, action):
        self.root = next(
            filter(lambda x: x.action == action, self.root.children)
        )
        self.root.inherit()
        self.root.parent = None

    def extract(self):
        s, p = zip(*self.history)
        r = self.root.vine.result
        l = len(self.history)

        size = self.root.vine.size
        data = self.root.vine.data

        def fill(data, i, n):
            p = np.zeros(data.size, dtype=float)
            p[i] = n

            return p

        state = np.stack(s).reshape(l, size, size, -1)
        pi = np.stack([fill(data, i, n) for i, n in p])

        z = np.empty((l, 1), dtype=float)
        z[0::2] = -r
        z[1::2] = r

        return state, pi, z
