"""
Persistent Potential Field implementation for NWM framework.

The potential field stores a collection of centroids and provides
methods for querying attractive/repulsive forces based on spatial proximity.

Performance
-----------
The stacked matrix of centroid states is cached and maintained incrementally:
appending a centroid costs ``O(d)`` (a single row is stacked on), merging into
an existing centroid costs ``O(d)`` (a single row is overwritten), and only a
prune (rare) invalidates the whole cache. Because ``query_forces`` is called
every environment step while the field is *not* mutated during a rollout, the
cache stays valid across an entire episode of action selections, replacing the
previous per-call ``O(N*d)`` rebuild.
"""

from __future__ import annotations

import numpy as np

from nwm.core.centroid import PersistentCentroid


class PersistentPotentialField:
    """
    Spatial memory structure that stores centroids and computes forces.

    The potential field maintains a collection of centroids representing
    learned experiences. When queried, it returns forces based on:
    - Spatial proximity to nearby centroids
    - Historical success/failure of actions in those regions
    - Confidence (variance) and lock status of centroids

    Attributes
    ----------
    state_dim : int
        Dimensionality of the state space.
    max_centroids : int
        Maximum number of centroids to maintain.
    centroids : list[PersistentCentroid]
        Collection of memory centroids.

    Examples
    --------
    >>> field = PersistentPotentialField(state_dim=4, max_centroids=100)
    >>> field.add(state=np.array([0.1, 0.2, 0.3, 0.4]), action=1, score=0.9)
    >>> forces, min_dist = field.query_forces(np.array([0.1, 0.2, 0.3, 0.4]))
    >>> 1 in forces
    True
    """

    def __init__(
        self,
        state_dim: int,
        max_centroids: int = 500,
        merge_threshold: float = 0.3,
        distance_cutoff: float = 2.5,
        lock_min_visits: int = 8,
        lock_min_score: float = 0.80,
        lock_boost: float = 2.5,
        score_ema: float = 0.0,
        dynamic_unlock: bool = False,
        credit_propagation: float = 0.0,
    ) -> None:
        """
        Initialize a new potential field.

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state space.
        max_centroids : int
            Maximum centroids to maintain (older low-value ones are pruned).
        merge_threshold : float
            Distance threshold for merging experiences into existing centroids.
        distance_cutoff : float
            Maximum distance for a centroid to influence force calculations.
        lock_min_visits : int
            Minimum visits before a centroid can be locked.
        lock_min_score : float
            Minimum average score to allow locking.
        lock_boost : float
            Multiplier for locked centroid influence.
        score_ema : float
            If > 0, per-action centroid scores are exponential moving averages
            (recency-weighted) instead of all-time running sums.
        dynamic_unlock : bool
            If True, locked centroids whose recent scores collapse lose their
            lock instead of staying protected forever.
        """
        self.state_dim = state_dim
        self.max_centroids = max_centroids
        self.merge_threshold = merge_threshold
        self.distance_cutoff = distance_cutoff
        self.lock_min_visits = lock_min_visits
        self.lock_min_score = lock_min_score
        self.lock_boost = lock_boost
        self.score_ema = score_ema
        self.dynamic_unlock = dynamic_unlock
        self.credit_propagation = credit_propagation

        self.centroids: list[PersistentCentroid] = []

        # Successor graph for credit propagation: uid -> action -> {succ_uid:
        # transition count}, plus the propagated per-centroid values.
        self._next_uid = 0
        self.succ: dict[int, dict[int, dict[int, int]]] = {}
        self._prop_values: dict[int, float] = {}

        # Cached (N, state_dim) matrix of centroid states, maintained
        # incrementally. ``None`` marks the cache as invalid (rebuilt lazily).
        self._state_matrix: np.ndarray | None = None

        # Running state normalization statistics
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_var = np.ones(state_dim, dtype=np.float32)
        self.state_count = 0

        # Global statistics
        self.total_experiences = 0
        self.score_sum = 0.0

    # ------------------------------------------------------------------
    # State-matrix cache
    # ------------------------------------------------------------------
    def _stack_states(self) -> np.ndarray:
        """Return the cached (N, d) centroid-state matrix, rebuilding if stale."""
        if self._state_matrix is None:
            if self.centroids:
                self._state_matrix = np.stack([c.state for c in self.centroids]).astype(
                    np.float32, copy=False
                )
            else:
                self._state_matrix = np.empty((0, self.state_dim), dtype=np.float32)
        return self._state_matrix

    def _invalidate_matrix(self) -> None:
        """Mark the state-matrix cache as needing a full rebuild."""
        self._state_matrix = None

    def _update_state_stats(self, state: np.ndarray) -> None:
        """Update running mean and variance for state normalization."""
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_var += (delta * delta2 - self.state_var) / self.state_count

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        state = np.asarray(state, dtype=np.float32)
        if self.state_count < 10:
            return state
        std = np.sqrt(self.state_var + 1e-8)
        return np.asarray((state - self.state_mean) / std, dtype=np.float32)

    def _new_centroid(self, state: np.ndarray, score: float, action: int) -> PersistentCentroid:
        centroid = PersistentCentroid(state, score, action, self.score_ema)
        centroid.uid = self._next_uid
        self._next_uid += 1
        return centroid

    def add(self, state: np.ndarray, action: int, score: float) -> int:
        """
        Add a new experience to the potential field.

        If a nearby centroid exists (within merge_threshold), the experience
        is merged. Otherwise, a new centroid is created.

        Parameters
        ----------
        state : np.ndarray
            Observed state.
        action : int
            Action taken.
        score : float
            Resulting score (0-1 scale, higher is better).

        Returns
        -------
        int
            The uid of the centroid that absorbed this experience.
        """
        norm_state = self._normalize_state(state)

        self.total_experiences += 1
        self.score_sum += score

        if not self.centroids:
            centroid = self._new_centroid(norm_state, score, action)
            self.centroids.append(centroid)
            self._invalidate_matrix()
            return centroid.uid

        # Find nearest centroid using the cached state matrix.
        matrix = self._stack_states()
        dists = np.linalg.norm(matrix - norm_state, axis=1)
        nearest_idx = int(np.argmin(dists))
        min_dist = float(dists[nearest_idx])

        if min_dist < self.merge_threshold:
            centroid = self.centroids[nearest_idx]
            centroid.merge(
                norm_state,
                score,
                action,
                self.lock_min_visits,
                self.lock_min_score,
                self.dynamic_unlock,
            )
            # Only one centroid's state changed: patch that row in place.
            if self._state_matrix is not None:
                self._state_matrix[nearest_idx] = centroid.state
            return centroid.uid
        new_centroid = self._new_centroid(norm_state, score, action)
        self.centroids.append(new_centroid)
        if self._state_matrix is not None:
            self._state_matrix = np.vstack([self._state_matrix, new_centroid.state[None, :]])
        if len(self.centroids) > self.max_centroids + 50:
            self._prune()
        return new_centroid.uid

    # ------------------------------------------------------------------
    # Credit propagation (successor graph)
    # ------------------------------------------------------------------
    def record_transition(self, prev_uid: int, action: int, next_uid: int) -> None:
        """Record that taking ``action`` at centroid ``prev_uid`` led to
        ``next_uid`` (consecutive steps of one trajectory). Self-loops are
        skipped: they carry no propagation information."""
        if prev_uid == next_uid:
            return
        actions = self.succ.setdefault(prev_uid, {})
        counts = actions.setdefault(action, {})
        counts[next_uid] = counts.get(next_uid, 0) + 1

    def propagate(self, sweeps: int = 3) -> None:
        """Run ``sweeps`` rounds of value propagation over the successor graph.

        Values live in score space (0-1, 0.5 neutral): each centroid's value is
        a blend of its own recent score and the count-weighted mean value of
        its successors. Good outcomes flow backwards along trajectories, so a
        centroid whose own episodes failed can still learn that it leads to a
        good region (trajectory stitching across episodes).
        """
        beta = self.credit_propagation
        if beta <= 0.0 or not self.centroids:
            return
        base = {c.uid: c.score_ema_val for c in self.centroids}
        values = dict(base)
        for _ in range(sweeps):
            nxt: dict[int, float] = {}
            for uid, own in base.items():
                actions = self.succ.get(uid)
                if not actions:
                    nxt[uid] = own
                    continue
                acc = 0.0
                weight = 0
                for counts in actions.values():
                    for succ_uid, cnt in counts.items():
                        val = values.get(succ_uid)
                        if val is not None:
                            acc += cnt * val
                            weight += cnt
                nxt[uid] = (1.0 - beta) * own + beta * (acc / weight) if weight else own
            values = nxt
        self._prop_values = values

    def _successor_value(self, uid: int, action: int) -> float | None:
        """Count-weighted propagated value of the successors of (uid, action)."""
        counts = self.succ.get(uid, {}).get(action)
        if not counts:
            return None
        acc = 0.0
        weight = 0
        for succ_uid, cnt in counts.items():
            val = self._prop_values.get(succ_uid)
            if val is not None:
                acc += cnt * val
                weight += cnt
        return acc / weight if weight else None

    def _prune(self) -> None:
        """Remove low-value centroids when capacity is exceeded."""
        if len(self.centroids) <= self.max_centroids:
            return

        # Preserve locked centroids
        locked = [c for c in self.centroids if c.locked]
        unlocked = [c for c in self.centroids if not c.locked]

        remaining_slots = self.max_centroids - len(locked)
        if remaining_slots < 0:
            # Too many locked: keep most valuable
            self.centroids.sort(
                key=lambda c: c.count * ((c.score_sum / c.count - 0.5) ** 2 + 0.1),
                reverse=True,
            )
            self.centroids = self.centroids[: self.max_centroids]
            self._invalidate_matrix()
            self._prune_graph()
            return

        # Keep most valuable unlocked centroids
        unlocked.sort(
            key=lambda c: c.count * ((c.score_sum / c.count - 0.5) ** 2 + 0.1),
            reverse=True,
        )
        kept_unlocked = unlocked[:remaining_slots]
        self.centroids = locked + kept_unlocked
        self._invalidate_matrix()
        self._prune_graph()

    def _prune_graph(self) -> None:
        """Drop successor-graph entries that reference pruned centroids."""
        if not self.succ:
            return
        alive = {c.uid for c in self.centroids}
        self.succ = {
            uid: pruned
            for uid, actions in self.succ.items()
            if uid in alive
            and (
                pruned := {
                    action: kept
                    for action, counts in actions.items()
                    if (kept := {u: n for u, n in counts.items() if u in alive})
                }
            )
        }
        self._prop_values = {u: v for u, v in self._prop_values.items() if u in alive}

    def query_forces(self, state: np.ndarray, k: int = 20) -> tuple[dict[int, float], float]:
        """
        Query forces for all actions from nearby centroids.

        Returns a dictionary mapping action IDs to force values:
        - Positive values indicate attraction (good outcomes)
        - Negative values indicate repulsion (bad outcomes)

        Parameters
        ----------
        state : np.ndarray
            Current state to query from.
        k : int
            Number of nearest centroids to consider.

        Returns
        -------
        tuple[dict[int, float], float]
            (forces_dict, min_distance) where forces_dict maps actions
            to their net force and min_distance is distance to nearest centroid.
        """
        if not self.centroids:
            return {}, float("inf")

        norm_state = self._normalize_state(state)

        matrix = self._stack_states()
        all_dists = np.linalg.norm(matrix - norm_state, axis=1)
        nearest_indices = np.argsort(all_dists)[:k]

        forces: dict[int, float] = {}
        total_weights: dict[int, float] = {}
        min_dist = float(all_dists[nearest_indices[0]])

        for idx in nearest_indices:
            dist = float(all_dists[idx])
            if dist > self.distance_cutoff:
                continue

            centroid = self.centroids[idx]

            for action in centroid.action_votes:
                if self.credit_propagation > 0.0:
                    succ_value = self._successor_value(centroid.uid, action)
                    force = centroid.get_force(action, succ_value, self.credit_propagation)
                else:
                    force = centroid.get_force(action)
                var = centroid.get_action_variance(action)
                confidence = 1.0 / (1.0 + var * 2.0)

                # Repulsion uses stronger spatial decay (inverse square)
                spatial_weight = 1.0 / (dist**2 + 0.1) if force < 0 else 1.0 / (dist + 0.1)

                # Boost for locked centroids
                lock_boost = self.lock_boost if centroid.locked else 1.0

                impact = (
                    force
                    * spatial_weight
                    * confidence
                    * float(np.log(1 + centroid.count))
                    * lock_boost
                )

                if action not in forces:
                    forces[action] = 0.0
                    total_weights[action] = 0.0

                forces[action] += impact
                total_weights[action] += spatial_weight

        # Normalize by total weights
        final_forces: dict[int, float] = {}
        for a in forces:
            if total_weights[a] > 0:
                final_forces[a] = forces[a] / total_weights[a]

        return final_forces, min_dist

    def get_stats(self) -> dict[str, float]:
        """
        Get statistics about the potential field.

        Returns
        -------
        dict[str, float]
            Statistics including centroid count, locked count, and scores.
            Count entries (``num_centroids``, ``locked_centroids``) are integers.
        """
        locked_count = sum(1 for c in self.centroids if c.locked)
        avg_state_variance = (
            float(np.mean([c.get_state_variance() for c in self.centroids]))
            if self.centroids
            else 0.0
        )
        return {
            "num_centroids": len(self.centroids),
            "locked_centroids": locked_count,
            "avg_score": self.score_sum / max(1, self.total_experiences),
            "avg_state_variance": avg_state_variance,
        }

    def clear(self) -> None:
        """Clear all centroids and reset statistics."""
        self.centroids = []
        self._invalidate_matrix()
        self._next_uid = 0
        self.succ = {}
        self._prop_values = {}
        self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
        self.state_var = np.ones(self.state_dim, dtype=np.float32)
        self.state_count = 0
        self.total_experiences = 0
        self.score_sum = 0.0

    def __len__(self) -> int:
        """Return number of centroids."""
        return len(self.centroids)

    def __repr__(self) -> str:
        """String representation."""
        locked = sum(1 for c in self.centroids if c.locked)
        return (
            f"PersistentPotentialField(centroids={len(self.centroids)}, "
            f"locked={locked}, experiences={self.total_experiences})"
        )
