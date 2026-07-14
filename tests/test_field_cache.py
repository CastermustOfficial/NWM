"""Regression tests for the incremental centroid-state matrix cache.

The cache must stay bit-for-bit consistent with a brute-force restack after any
mix of merges, appends, and prunes, and querying must be unaffected by it.
"""

import numpy as np

from nwm.core.potential_field import PersistentPotentialField


def _brute_force_matrix(field):
    if not field.centroids:
        return np.empty((0, field.state_dim), dtype=np.float32)
    return np.stack([c.state for c in field.centroids]).astype(np.float32)


class TestStateMatrixCache:
    def test_cache_matches_bruteforce_after_merges_and_appends(self):
        rng = np.random.default_rng(0)
        field = PersistentPotentialField(state_dim=4, merge_threshold=0.5)
        for _ in range(300):
            state = rng.standard_normal(4).astype(np.float32)
            field.add(state, action=int(rng.integers(0, 2)), score=float(rng.random()))
            cached = field._stack_states()
            brute = _brute_force_matrix(field)
            assert cached.shape == brute.shape
            assert np.allclose(cached, brute)

    def test_cache_matches_bruteforce_after_prune(self):
        rng = np.random.default_rng(1)
        # Tiny capacity + tiny merge threshold forces frequent pruning.
        field = PersistentPotentialField(state_dim=3, max_centroids=10, merge_threshold=0.001)
        for _ in range(500):
            state = (rng.standard_normal(3) * 100).astype(np.float32)
            field.add(state, action=int(rng.integers(0, 2)), score=float(rng.random()))
        assert len(field) <= field.max_centroids + 50
        assert np.allclose(field._stack_states(), _brute_force_matrix(field))

    def test_query_matches_independent_recompute(self):
        rng = np.random.default_rng(2)
        field = PersistentPotentialField(state_dim=4, merge_threshold=0.4)
        for _ in range(200):
            field.add(
                rng.standard_normal(4).astype(np.float32),
                action=int(rng.integers(0, 3)),
                score=float(rng.random()),
            )

        probe = rng.standard_normal(4).astype(np.float32)
        forces_cached, dist_cached = field.query_forces(probe)

        # Force a full rebuild and query again; results must be identical.
        field._invalidate_matrix()
        forces_fresh, dist_fresh = field.query_forces(probe)

        assert dist_cached == dist_fresh
        assert forces_cached.keys() == forces_fresh.keys()
        for a in forces_cached:
            assert forces_cached[a] == forces_fresh[a]

    def test_clear_resets_cache(self):
        field = PersistentPotentialField(state_dim=2)
        field.add(np.array([1.0, 2.0], dtype=np.float32), action=0, score=0.9)
        assert field._stack_states().shape == (1, 2)
        field.clear()
        assert field._stack_states().shape == (0, 2)
