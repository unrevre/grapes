# pylint: disable=redefined-outer-name,missing-docstring,invalid-name

import pytest

@pytest.fixture
def grapevine():
    import grapes.state.vine as vine
    return vine.Vine(4)

ref_vine_insert = """
|       |
|       |
|    b  |
|       |
"""

def test_vine_insert(grapevine):
    grapevine.insert(10)
    assert grapevine.__str__() == ref_vine_insert[1:-1]

ref_vine_move_0 = """
|  b    |
|b w    |
|       |
|       |
"""

def test_vine_move_0(grapevine):
    grapevine.move(1)
    grapevine.move(0)
    grapevine.move(4)
    grapevine.move(5)
    assert grapevine.__str__() == ref_vine_move_0[1:-1]

ref_vine_move_1 = """
|    w  |
|w   w  |
|  w b  |
|       |
"""

def test_vine_move_1(grapevine):
    grapevine.move(0)
    grapevine.move(2)
    grapevine.move(1)
    grapevine.move(4)
    grapevine.move(5)
    grapevine.move(6)
    grapevine.move(10)
    grapevine.move(9)
    assert grapevine.__str__() == ref_vine_move_1[1:-1]

ref_vine_move_2 = """
|  w b  |
|w b    |
|       |
|       |
"""

def test_vine_move_2(grapevine):
    grapevine.move(0)
    grapevine.move(1)
    grapevine.move(2)
    grapevine.move(4)
    grapevine.move(5)
    assert grapevine.__str__() == ref_vine_move_2[1:-1]

def test_vine_neighbours(grapevine):
    assert sorted(grapevine.adjacent(0)) == [1, 4]
    assert sorted(grapevine.adjacent(4)) == [0, 5, 8]
    assert sorted(grapevine.adjacent(5)) == [1, 4, 6, 9]
    assert sorted(grapevine.adjacent(15)) == [11, 14]

def test_vine_group(grapevine):
    grapevine.insert(0)
    grapevine.insert(1)
    grapevine.insert(6)
    assert sorted(grapevine.group(0)[0]) == [0, 1]
    assert sorted(grapevine.group(6)[0]) == [6]
    grapevine.next()
    grapevine.insert(5)
    assert sorted(grapevine.group(0)[0]) == [0, 1]
    assert sorted(grapevine.group(5)[0]) == [5]
    grapevine.next()
    grapevine.insert(2)
    assert sorted(grapevine.group(0)[0]) == [0, 1, 2, 6]

def test_vine_space(grapevine):
    grapevine.insert(0)
    grapevine.insert(1)
    grapevine.next()
    grapevine.insert(5)
    assert sorted(grapevine.group(0)[1]) == [2, 4]
    assert sorted(grapevine.group(5)[1]) == [4, 6, 9]

def test_vine_buds(grapevine):
    size = grapevine.size * grapevine.size
    assert len(list(grapevine.buds())) == size + 1
    grapevine.insert(1)
    grapevine.insert(4)
    assert len(list(grapevine.buds())) == size - 1

ref_vine_remove = """
|       |
|  b    |
|       |
|       |
"""

def test_vine_remove(grapevine):
    grapevine.insert(0)
    grapevine.insert(5)
    grapevine.remove(0)
    assert grapevine.__str__() == ref_vine_remove[1:-1]

def test_vine_move_illegal(grapevine):
    grapevine.move(0)
    grapevine.move(4)
    grapevine.move(5)
    grapevine.move(1)
    assert grapevine.move(0) == False

ref_vine_move_capture_priority = """
|b w    |
|  b    |
|b      |
|       |
"""

def test_vine_move_capture_priority(grapevine):
    grapevine.move(8)
    grapevine.move(4)
    grapevine.move(5)
    grapevine.move(1)
    grapevine.move(0)
    assert grapevine.__str__() == ref_vine_move_capture_priority[1:-1]

def test_vine_zhash(grapevine):
    ref0 = grapevine.hash.hash
    grapevine.insert(1)
    grapevine.insert(4)
    grapevine.remove(1)
    grapevine.remove(4)
    assert grapevine.hash.hash == ref0

def test_vine_ko_legality(grapevine):
    grapevine.move(1)
    grapevine.move(5)
    grapevine.move(4)
    grapevine.move(2)
    grapevine.move(6)
    grapevine.move(0)
    assert grapevine.move(1) == False

def test_vine_null(grapevine):
    size = grapevine.size * grapevine.size
    grapevine.move(0)
    grapevine.move(5)
    grapevine.move(16)
    assert grapevine.null == 1
    grapevine.move(1)
    assert grapevine.null == 0
    grapevine.move(16)
    assert len(list(grapevine.buds())) == size - 2
    grapevine.move(16)
    assert grapevine.null == 2
