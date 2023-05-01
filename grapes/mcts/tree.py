# pylint: disable=missing-docstring,invalid-name

import copy

import numpy as np

import grapes.state.errors as errors
import grapes.state.seed as seed


class Tree:
    C = 5.0

    def __init__(self, parent, action, vine):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

        self.vine = vine
        self.action = action
        self.parent = parent
        self.children = []

    def info(self, key):
        if key == 'side':
            return (
                '[{}]'.format(
                    seed.abbr(seed.inverse(self.vine.seed))
                )
                if self.action is not None
                else '[ ]'
            )

        if key == 'action':
            return (
                '[{: 2}][{: 2}]'.format(
                    self.action // self.vine.size,
                    self.action % self.vine.size,
                )
                if self.action is not None
                else '[  ][  ]'
            )

        if key == 'self':
            return '{} {} {: .4f} {: .4f}                 [{: 5}]'.format(
                self.info('side'),
                self.info('action'),
                self.p,
                self.q,
                self.n,
            )

        if key == 'info':
            q, p, n = zip(*((node.q, node.p, node.n) for node in self.children))
            u = Tree.C * np.asarray(p) * np.sqrt(self.n) / (1 + np.asarray(n))

            return '\n'.join(
                '    {} {: .4f} {: .4f} {: .4f} {: .4f} [{: 5}]'.format(
                    node.info('action'),
                    p[i],
                    q[i],
                    u[i],
                    q[i] + u[i],
                    n[i],
                )
                for i, node in enumerate(self.children)
            )

        if key == 'div':
            return '----------------------------------------------------'

    def __str__(self):
        return '\n'.join([self.info(key) for key in ['self', 'div', 'info']])

    @property
    def state(self):
        index, n = zip(*((node.action, node.n) for node in self.children))
        index, n = np.asarray(index, dtype=int), np.asarray(n, dtype=float)

        return np.swapaxes(self.vine.state, 0, -1), (index, n / np.sum(n))

    @property
    def next(self):
        q, p, n = zip(*((node.q, node.p, node.n) for node in self.children))
        q, p, n = np.asarray(q), np.asarray(p), np.asarray(n)

        index = np.argmax(q + Tree.C * p * np.sqrt(self.n) / (1 + n))

        return self.children[index]

    def descend(self):
        node = self

        while node.children:
            node = node.next

        return node

    def expand(self, policy):
        if self.vine.complete:
            return

        for index in self.vine.buds():
            vine = copy.copy(self.vine)

            if not vine.move(index):
                continue

            node = Tree(self, index, vine)
            node.p = policy[index]

            self.children.append(node)

        norm = sum(node.p for node in self.children)

        for node in self.children:
            node.p = node.p / norm

    def update(self, value):
        node = self

        while node is not None:
            node.n = node.n + 1
            node.w = node.w + value
            node.q = node.w / node.n

            node = node.parent

            value = -value

    def smear(self, alpha, frac):
        p = np.fromiter((node.p for node in self.children), dtype=float)
        noise = np.random.dirichlet(np.full(p.shape, alpha, dtype=float))

        prob = p * (1.0 - frac) + noise * frac

        for i, node in enumerate(self.children):
            node.p = prob[i]

    def trace(self):
        print(node.vine, '\n')

        node = self
        while node.children:
            node = node.next

            print(node.info('self'))

        print(node.vine, '\n')
