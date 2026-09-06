from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CollisionResult:
    is_candidate:      bool
    min_distance:      float        # to nearest different-outcome neighbor
    nearest_action:    int | None   # that neighbor's optimal action
    threshold_used:    float        # q_near distance quantile value


class CollisionAnalyzer:
    """§10 bottleneck collision analysis.

    Tests whether missed harmful events (Z_u=1, JSD below threshold)
    are enriched near states with different optimal actions that
    were mapped close together by the shared representation.
    """

    def __init__(
        self,
        instrumented,    # SimulusInstrumented
        q_near: float = 0.25,
        q_far:  float = 0.75,
        max_db: int   = 500,
    ) -> None:
        self._instr  = instrumented
        self.q_near  = q_near
        self.q_far   = q_far
        # (rep: np.ndarray, optimal_action: int, q_p_max: float)
        self._db: list[tuple[np.ndarray, int, float]] = []
        self._max_db = max_db

    def add_to_database(
        self,
        b_ua: np.ndarray,
        optimal_action: int,
        q_p_max: float,
    ) -> None:
        """Add one reference point to the collision database."""
        if len(self._db) >= self._max_db:
            self._db.pop(0)
        self._db.append((b_ua, optimal_action, q_p_max))

    def extract_b_ua(
        self,
        instrumented_output,  # InstrumentedActionOutput for one action
    ) -> np.ndarray | None:
        """Extract b_{u,a} — §0.5 item 2.

        Read backend/soma/instrumentation.py to find where b_ua is stored
        in InstrumentedActionOutput. If it is None (hook not yet wired),
        read /tmp/simulus/src to find the pre-bottleneck activation name,
        add the hook, and return the captured value.
        Do not return a zero vector as a substitute — None is correct
        when the hook is absent.
        """
        return instrumented_output.b_ua  # may be None — check before use

    def is_collision_candidate(
        self,
        b_missed: np.ndarray,
        a_missed_optimal: int,
    ) -> CollisionResult:
        """Test one missed harmful event against the database.

        A collision candidate: the nearest database entry with a DIFFERENT
        optimal action is within the q_near quantile of all pairwise distances.
        Same-action neighbors are excluded — they are not collisions.
        """
        if len(self._db) < 10:
            return CollisionResult(False, float('inf'), None, 0.0)

        reps     = np.stack([e[0] for e in self._db])
        actions  = np.array([e[1] for e in self._db])
        dists    = np.linalg.norm(reps - b_missed, axis=1)

        # Only different-outcome neighbors count
        diff_mask = actions != a_missed_optimal
        if not diff_mask.any():
            return CollisionResult(False, float('inf'), None, 0.0)

        diff_dists   = dists[diff_mask]
        min_dist     = float(diff_dists.min())
        # Find the index within the filtered array, then map back
        nearest_local_idx = int(diff_dists.argmin())
        # Map back to original db index
        original_indices = np.where(diff_mask)[0]
        nearest_idx      = int(original_indices[nearest_local_idx])
        nearest_act      = int(self._db[nearest_idx][1])
        threshold        = float(np.quantile(dists, self.q_near))

        return CollisionResult(
            is_candidate   = min_dist <= threshold,
            min_distance   = min_dist,
            nearest_action = nearest_act,
            threshold_used = threshold,
        )

    def compute_enrichment(
        self,
        missed_results:  list[CollisionResult],  # Z_u=1, JSD missed
        null_results:    list[CollisionResult],  # everything else
    ) -> dict:
        """§10 enrichment test.

        Returns verdict dict written to h2_result.json.
        """
        if len(missed_results) < 5:
            return {
                'verdict':               'INSUFFICIENT_DATA',
                'n_missed':              len(missed_results),
                'n_null':                len(null_results),
                'collision_rate_missed': None,
                'collision_rate_null':   None,
                'enrichment_ratio':      None,
                'finding':               f'Only {len(missed_results)} missed harmful events. Need ≥5.',
            }

        rate_missed = sum(r.is_candidate for r in missed_results) / len(missed_results)
        rate_null   = sum(r.is_candidate for r in null_results)   / max(len(null_results), 1)
        ratio       = rate_missed / max(rate_null, 1e-6)

        if ratio >= 1.5:
            verdict = 'ENRICHED'
            finding = (
                f'Missed harmful events show {ratio:.2f}x collision enrichment '
                f'({rate_missed:.1%} vs {rate_null:.1%} null). '
                f'Bottleneck aliasing is associated with JSD detection failure.'
            )
        elif ratio <= 0.75:
            verdict = 'NOT_ENRICHED'
            finding = (
                f'Missed harmful events NOT enriched for collisions (ratio={ratio:.2f}). '
                f'Bottleneck aliasing is not the primary explanation for JSD misses.'
            )
        else:
            verdict = 'INCONCLUSIVE'
            finding = f'Enrichment ratio={ratio:.2f} — inconclusive. Need more missed events.'

        return {
            'verdict':               verdict,
            'n_missed':              len(missed_results),
            'n_null':                len(null_results),
            'collision_rate_missed': rate_missed,
            'collision_rate_null':   rate_null,
            'enrichment_ratio':      ratio,
            'finding':               finding,
            'q_near_used':           missed_results[0].threshold_used if missed_results else None,
        }