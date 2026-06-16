import numpy as np

from wcp.model.standings import best_n_indices, group_sortkey


def test_best_eight_selects_top_keys():
    # 12 groups, single sim. Keys increasing with index; best 8 = indices 4..11.
    keys = group_sortkey(
        points=np.arange(12, dtype=float),  # 0..11 points
        gd=np.zeros(12), gf=np.zeros(12))[:, None]
    mask = best_n_indices(keys, 8)[:, 0]
    chosen = set(np.where(mask)[0])
    assert chosen == set(range(4, 12))
    assert mask.sum() == 8


def test_best_eight_vectorised_across_sims():
    rng = np.random.default_rng(1)
    keys = rng.random((12, 50))
    mask = best_n_indices(keys, 8)
    # Exactly 8 chosen per sim.
    assert np.all(mask.sum(axis=0) == 8)
    # Chosen keys are all >= the max unchosen key per sim.
    for s in range(50):
        chosen = keys[mask[:, s], s]
        unchosen = keys[~mask[:, s], s]
        assert chosen.min() >= unchosen.max()


def test_goal_difference_orders_thirds():
    # Equal points, GD decides which third-placed teams are best.
    keys = group_sortkey(points=np.full(12, 3.0),
                         gd=np.arange(-6, 6, dtype=float),
                         gf=np.zeros(12))[:, None]
    mask = best_n_indices(keys, 8)[:, 0]
    # The four eliminated should be the four lowest GD (indices 0..3).
    assert set(np.where(~mask)[0]) == {0, 1, 2, 3}
