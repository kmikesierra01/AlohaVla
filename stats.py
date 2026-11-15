"""Success rate, 95% CI and same-seed comparison from lerobot-eval output dirs (stdlib only).

Usage: python stats.py <eval_dir> [<eval_dir_to_compare>]   (compare only runs that used the same seeds)
"""
import json
import math
import sys
from collections import Counter


def load(d):
    m = json.load(open(f"{d}/eval_info.json"))["per_task"][0]["metrics"]
    return [bool(s) for s in m["successes"]], [int(r) for r in m["max_rewards"]]


def wilson(k, n, z=1.96):
    """95% confidence interval of a success rate k/n (Wilson score interval)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar(only_a, only_b):
    """Exact McNemar p: chance of a split at least this lopsided if both models were equally good."""
    n = only_a + only_b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)


if len(sys.argv) < 2:  # self-check, then usage
    lo, hi = wilson(160, 200)
    assert abs(lo - 0.739) < 1e-3 and abs(hi - 0.850) < 1e-3
    assert abs(mcnemar(16, 34) - 0.015) < 1e-3
    sys.exit(__doc__)

runs = [load(d) for d in sys.argv[1:3]]
for d, (s, r) in zip(sys.argv[1:], runs):
    k, n = sum(s), len(s)
    lo, hi = wilson(k, n)
    print(f"{d}: {k}/{n} = {k / n:.1%}  95% CI {lo:.1%} - {hi:.1%}  max_reward counts {dict(sorted(Counter(r).items()))}")
    # two_cubes.py: max_reward 2-4 = commanded cube lifted first, 10+ = the other cube lifted first
    ok, wrong = sum(2 <= x < 10 for x in r), sum(x >= 10 for x in r)
    print(f"  first lift: commanded cube {ok}, other cube {wrong}, none {n - ok - wrong}"
          + (f" -> right colour {ok}/{ok + wrong} = {ok / (ok + wrong):.1%}" if wrong else ""))
if len(runs) == 2:
    a, b = runs[0][0], runs[1][0]
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    both = sum(x and y for x, y in zip(a, b))
    print(f"same seeds: both {both}, only 1st {only_a}, only 2nd {only_b}, neither {len(a) - both - only_a - only_b}"
          f" -> McNemar p = {mcnemar(only_a, only_b):.3f}")
