# pylint: disable=redefined-outer-name,missing-docstring,invalid-name

import pytest

@pytest.fixture
def grapevine():
    import grapes.state.vine as vine
    return vine.Vine(4)

def test_vine_insert_bounds(grapevine):
    import grapes.state.errors as errors
    with pytest.raises(errors.InvalidPoint):
        grapevine.insert(-1, -1)

def test_vine_insert_overlap(grapevine):
    grapevine.insert(0, 0)
    import grapes.state.errors as errors
    with pytest.raises(errors.FilledPoint):
        grapevine.insert(0, 0)

ref_vine_insert = """
|       |
|       |
|    b  |
|       |
"""

def test_vine_insert(grapevine):
    grapevine.insert(2, 2)
    assert grapevine.__str__() == ref_vine_insert[1:-1]

ref_vine_move = """
|w b    |
|b      |
|       |
|       |
"""

def test_vine_move(grapevine):
    grapevine.move(0, 1)
    grapevine.move(0, 0)
    grapevine.move(1, 0)
    assert grapevine.__str__() == ref_vine_move[1:-1]

def test_vine_neighbours(grapevine):
    assert sorted(grapevine.adjacent(0, 0)) == [(0, 1), (1, 0)]
    assert sorted(grapevine.adjacent(1, 0)) == [(0, 0), (1, 1), (2, 0)]
    assert sorted(grapevine.adjacent(1, 1)) == [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert sorted(grapevine.adjacent(3, 3)) == [(2, 3), (3, 2)]

def test_vine_group(grapevine):
    grapevine.insert(0, 0)
    grapevine.insert(0, 1)
    grapevine.insert(1, 2)
    assert grapevine.group(0, 0) == set([(0, 0), (0, 1)])
    assert grapevine.group(1, 2) == set([(1, 2)])
    grapevine.next()
    grapevine.insert(1, 1)
    assert grapevine.group(0, 0) == set([(0, 0), (0, 1)])
    assert grapevine.group(1, 1) == set([(1, 1)])
    grapevine.next()
    grapevine.insert(0, 2)
    assert grapevine.group(0, 0) == set([(0, 0), (0, 1), (0, 2), (1, 2)])

def test_vine_remove_empty(grapevine):
    import grapes.state.errors as errors
    with pytest.raises(errors.EmptyPoint):
        grapevine.remove(0, 0)

ref_vine_remove = """
|       |
|  b    |
|       |
|       |
"""

def test_vine_remove(grapevine):
    grapevine.insert(0, 0)
    grapevine.insert(1, 1)
    grapevine.remove(0, 0)
    assert grapevine.__str__() == ref_vine_remove[1:-1]
