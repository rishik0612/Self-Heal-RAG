import math
import random
from typing import List, Tuple, Dict

Pair = Tuple[bool, bool]  # (baseline_outcome, variant_outcome)


def _counts(pairs: List[Pair]) -> Tuple[int, int, int, int]:
    #                   baseline True | baseline False
    # variant True      a             | c
    # variant False     b             | d
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    return a, b, c, d


def mcnemar_test(pairs: List[Pair]) -> Dict:
    """McNemar exact binomial test for matched binary outcomes.

    H0: baseline and variant perform equally on discordant pairs.
    Returns counts, rates, delta and a two-sided exact p-value.
    """
    a, b, c, d = _counts(pairs)
    n = len(pairs)
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        x = min(b, c)
        p_value = 2.0 * sum(
            math.comb(discordant, k) * (0.5 ** discordant)
            for k in range(x + 1)
        )
        p_value = min(1.0, p_value)
    return {
        "n": n,
        "a": a, "b": b, "c": c, "d": d,
        "discordant": discordant,
        "baseline_rate": (a + b) / n if n else 0.0,
        "variant_rate": (a + c) / n if n else 0.0,
        "delta": ((a + c) - (a + b)) / n if n else 0.0,
        "p_value": p_value,
    }


def bootstrap_diff_ci(pairs: List[Pair], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Dict:
    """Bootstrap 1-alpha CI for the change in outcome rate (variant - baseline)."""
    n = len(pairs)
    if n == 0:
        return {}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a, b, c, d = _counts(sample)
        base_rate = (a + b) / n
        var_rate = (a + c) / n
        diffs.append(var_rate - base_rate)
    diffs.sort()
    lo = diffs[max(0, int((alpha / 2) * n_boot))]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)]

    a, b, c, d = _counts(pairs)
    base_rate = (a + b) / n
    var_rate = (a + c) / n
    return {
        "n": n,
        "baseline_rate": base_rate,
        "variant_rate": var_rate,
        "diff": var_rate - base_rate,
        "ci_low": lo,
        "ci_high": hi,
        "significant_alpha": alpha,
        "ci_excludes_zero": not (lo <= 0 <= hi),
    }