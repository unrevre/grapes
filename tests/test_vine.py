# pylint: disable=redefined-outer-name,missing-docstring,invalid-name

import pytest

@pytest.fixture
def grapevine():
    import grapes.state.vine as vine
    return vine.Vine(4, 8, 0.5)

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

def test_vine_area_0(grapevine):
    grapevine.insert(0)
    grapevine.insert(5)
    grapevine.insert(10)
    grapevine.insert(15)
    assert grapevine.area() == 12
    grapevine.next()
    grapevine.insert(2)
    assert grapevine.area() == 6
    grapevine.insert(7)
    assert grapevine.area() == 5

def test_vine_area_1(grapevine):
    grapevine.insert(15)
    grapevine.next()
    grapevine.insert(1)
    grapevine.insert(4)
    grapevine.insert(6)
    grapevine.insert(9)
    assert grapevine.area() == -2
    grapevine.insert(12)
    assert grapevine.area() == -3

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

def test_vine_result(grapevine):
    grapevine.move(1)
    grapevine.move(14)
    grapevine.move(4)
    grapevine.move(11)
    grapevine.move(5)
    grapevine.move(10)
    grapevine.move(6)
    grapevine.move(9)
    grapevine.move(7)
    assert grapevine.result == 1
    grapevine.move(8)
    assert grapevine.result == -1

ref_vine_state_b = """[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]"""
ref_vine_state_w = """[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]"""
ref_vine_state_0 = """[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]"""
ref_vine_state_1 = """[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]"""
ref_vine_state_2 = """[0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]"""

def test_vine_state(grapevine):
    assert grapevine.state[-1].__str__() == ref_vine_state_b
    grapevine.move(1)
    grapevine.update()
    assert grapevine.state[-1].__str__() == ref_vine_state_w
    assert grapevine.state[8].__str__() == ref_vine_state_0
    grapevine.move(2)
    grapevine.update()
    assert grapevine.state[-1].__str__() == ref_vine_state_b
    assert grapevine.state[0].__str__() == ref_vine_state_0
    assert grapevine.state[8].__str__() == ref_vine_state_1
    grapevine.move(3)
    grapevine.update()
    assert grapevine.state[-1].__str__() == ref_vine_state_w
    assert grapevine.state[8].__str__() == ref_vine_state_2
    assert grapevine.state[0].__str__() == ref_vine_state_1
