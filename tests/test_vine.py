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
