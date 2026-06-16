import numpy as np

from wcp.model.standings import group_sortkey, rank_order


def _order_single(points, gd, gf):
    p = np.array(points)[:, None]
    d = np.array(gd)[:, None]
    f = np.array(gf)[:, None]
    return rank_order(p, d, f)[:, 0]


def test_points_dominate():
    # Team 2 has fewer goals but more points -> ranks first.
    order = _order_single(points=[3, 9, 6, 1], gd=[5, -2, 1, 0], gf=[8, 2, 4, 1])
    assert order[0] == 1  # 9 points


def test_goal_difference_breaks_points_tie():
    order = _order_single(points=[6, 6, 6, 0], gd=[1, 5, 3, 0], gf=[5, 5, 5, 0])
    assert list(order[:3]) == [1, 2, 0]  # by GD: 5 > 3 > 1


def test_goals_scored_breaks_gd_tie():
    order = _order_single(points=[6, 6, 0, 0], gd=[2, 2, 0, 0], gf=[7, 4, 0, 0])
    assert order[0] == 0 and order[1] == 1  # equal pts & GD, more GF first


def test_sortkey_strict_domination():
    # A one-point edge must beat any plausible GD/GF edge.
    hi = group_sortkey(np.array([3.0]), np.array([-30.0]), np.array([0.0]))
    lo = group_sortkey(np.array([2.0]), np.array([30.0]), np.array([30.0]))
    assert hi[0] > lo[0]


def test_full_group_ordering_is_permutation():
    order = _order_single(points=[4, 7, 7, 1], gd=[0, 3, 2, -5], gf=[3, 6, 5, 1])
    assert sorted(order.tolist()) == [0, 1, 2, 3]
