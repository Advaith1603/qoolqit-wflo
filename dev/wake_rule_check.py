"""Phase 1 probe: wake QUBO reimplemented from Hancock et al. arXiv:2312.13123, Eqs. (3)-(5).

Not derived from the Plymouth code. The paper does not state the wake-membership rule precisely;
the rule below reproduces the published R0 values and is symmetric under the grid's mirror symmetries:
  site j is in the wake of i for wind d  iff  0.001 < s <= x + 0.001  and  t < r * s - 1e-9,
  (cells exactly on the cone edge are outside; a bare `t < r * s` is rounding-dependent and breaks mirror symmetry)
  with s, t the downstream / lateral offsets of j's centre from i's centre.
Wind angle is counted clockwise from west (row index grows southwards).
"75% from west" = p(0 deg) = 0.75, remaining 0.25 spread evenly over the other 35 directions.
"""
import itertools
import numpy as np

A_IND, RT, V, TOL = 0.1, 0.33, 12.0, 1e-3


def regime(kind):
    p = np.full(36, 1 / 36) if kind == "uniform" else np.r_[0.75, np.full(35, 0.25 / 35)]
    return np.arange(36) * 10.0, p


def wake_matrix(L, x, r, kind):
    rc = np.array([(k // L, k % L) for k in range(L * L)], float)
    aT = (r - RT) / x
    W = np.zeros((L * L, L * L))
    for ang, pd in zip(*regime(kind)):
        th = np.deg2rad(ang)
        e = np.array([np.sin(th), np.cos(th)])  # direction of travel in (row, col)
        for i, j in itertools.permutations(range(L * L), 2):
            dv = rc[j] - rc[i]
            s, t = dv @ e, abs(dv[0] * e[1] - dv[1] * e[0])
            if TOL < s <= x + TOL and t < r * s - 1e-9:
                u = V * (1 - 2 * A_IND / (1 + aT * (np.hypot(*dv) / r) ** 2) ** 2)
                W[i, j] += pd * (V**3 - u**3) / 3
    return W + W.T  # symmetric pairwise penalty for an unordered pair


def optimum(W, L, m=4):
    combos = np.array(list(itertools.combinations(range(L * L), m)))
    e = m * V**3 / 3 - sum(W[combos[:, a], combos[:, b]] for a, b in itertools.combinations(range(m), 2))
    best = e.max()
    return best, combos[np.abs(e - best) < 1e-6]


if __name__ == "__main__":
    W0 = wake_matrix(4, 1, 1.5, "uniform")
    assert np.isclose(V**3 / 3, 576)
    assert np.isclose(W0[0, 1], 83.728, atol=1e-3) and np.isclose(W0[0, 5], 17.582, atol=1e-3)
    assert W0[0, [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]].max() == 0
    rungs = {"R0": (1, 1.5, "uniform", 2304.0), "R1": (3, 1, "uniform", 2229.4),
             "R2": (1, 1.5, "west75", 2304.0), "R3": (3, 1, "west75", 2242.7)}
    for name, (x, r, kind, target) in rungs.items():
        best, sets = optimum(wake_matrix(4, x, r, kind), 4)
        print(name, round(best, 2), len(sets), sets[0])
        assert round(best, 1) == target
    _, s0 = optimum(W0, 4)
    king = lambda a, b: max(abs(a // 4 - b // 4), abs(a % 4 - b % 4)) == 1
    indep = [c for c in itertools.combinations(range(16), 4) if not any(king(a, b) for a, b in itertools.combinations(c, 2))]
    assert len(s0) == 79 and {tuple(c) for c in s0} == set(indep)
    print("all oracle assertions pass")
