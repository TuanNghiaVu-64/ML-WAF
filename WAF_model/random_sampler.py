"""
random_sampler.py
=================
Implements the RAN (random attack generation) baseline from the paper.
Imports the grammar from grammar_definition.py — contains NO grammar data.

PUBLIC API
----------
    generate_attack(start_rule="start")
        → (attack_string, derivation)

    attack_string : str
        The raw SQLi payload ready to inject, e.g. "' or 1=1 #"

    derivation : list of (rule_name, chosen_alternative) tuples
        Full record of every expansion decision made during generation.
        Each entry is:
            rule_name        : str   — the non-terminal that was expanded
            chosen_alternative : list — the alternative that was selected
        This trace is consumed by the slice extractor in the next step.

ALGORITHM (paper Section 3.2)
------------------------------
1. Start at <start>
2. For each non-terminal encountered, collect all its alternatives and
   assign each a weight = product of descendant counts of its symbols.
   ("each production rule is selected with a probability proportional
    to the number of distinct production rules descending from it")
3. Pick one alternative via weighted random choice.
4. For each symbol in the chosen alternative:
     - terminal string  → emit it directly
     - non-terminal     → recurse (go to step 2)
     - OPT tuple        → include with 50 % probability, then recurse/emit
5. Concatenate all emitted terminals into the final attack string.

WHY WEIGHTED?
-------------
Without weighting, random.choice would strongly favour short alternatives
(fewer symbols = less chance of going wrong), producing a biased and
repetitive set. Weighting by descendant count spreads sampling more
evenly across the full space the grammar defines.
"""

import random
from grammar_definition import GRAMMAR

# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

# Module-level memo so descendant counts are computed only once per session
_descendant_memo: dict = {}


def _count_descendants(symbol: object) -> int:
    """
    Recursively count how many distinct terminal strings are reachable
    from *symbol*.

    Parameters
    ----------
    symbol : str | tuple
        Either a rule name / terminal string, or an ("OPT", ...) tuple.

    Returns
    -------
    int  ≥ 1
    """
    # ── OPT tuple ────────────────────────────────────────────────────────
    if isinstance(symbol, tuple):
        _, inner = symbol          # ("OPT", inner)
        if isinstance(inner, list):
            count = sum(_count_descendants(s) for s in inner)
        else:
            count = _count_descendants(inner)
        return count + 1           # +1 for the "skip" option

    # ── Terminal literal ──────────────────────────────────────────────────
    if symbol not in GRAMMAR:
        return 1

    # ── Non-terminal (cached) ─────────────────────────────────────────────
    if symbol in _descendant_memo:
        return _descendant_memo[symbol]

    # Seed with 1 before recursing to break any potential cycle
    _descendant_memo[symbol] = 1
    total = 0
    for alternative in GRAMMAR[symbol]:
        alt_weight = 1
        for sym in alternative:
            alt_weight *= _count_descendants(sym)
        total += alt_weight

    _descendant_memo[symbol] = max(total, 1)
    return _descendant_memo[symbol]


def _alternative_weight(alternative: list) -> int:
    """
    Weight for one alternative = product of descendant counts of its symbols.
    Minimum weight is 1 so random.choices never gets a zero-weight list.
    """
    weight = 1
    for sym in alternative:
        weight *= _count_descendants(sym)
    return max(weight, 1)


def _expand_symbol(symbol: object, derivation: list) -> str:
    """
    Expand a single symbol and return its terminal string contribution.

    Parameters
    ----------
    symbol     : str | tuple  — the symbol to expand
    derivation : list         — accumulator for (rule, alternative) pairs

    Returns
    -------
    str  — zero or more characters to concatenate into the attack string
    """
    # ── OPT tuple ─────────────────────────────────────────────────────────
    if isinstance(symbol, tuple):
        kind, inner = symbol
        assert kind == "OPT", f"Unknown tuple kind: {kind}"
        if random.random() < 0.5:
            return ""              # skip the optional element
        if isinstance(inner, list):
            return "".join(_expand_symbol(s, derivation) for s in inner)
        return _expand_symbol(inner, derivation)

    # ── Non-terminal ──────────────────────────────────────────────────────
    if symbol in GRAMMAR:
        return _sample(symbol, derivation)

    # ── Terminal literal ──────────────────────────────────────────────────
    return symbol


def _sample(rule_name: str, derivation: list) -> str:
    """
    Expand a non-terminal rule, record the decision, return the string.

    Parameters
    ----------
    rule_name  : str   — must be a key in GRAMMAR
    derivation : list  — accumulates (rule_name, chosen_alternative)

    Returns
    -------
    str  — the concatenated terminal output of this subtree
    """
    alternatives = GRAMMAR[rule_name]

    # Weighted random selection among alternatives
    weights = [_alternative_weight(alt) for alt in alternatives]
    chosen  = random.choices(alternatives, weights=weights, k=1)[0]

    # Record this expansion step for later slice extraction
    derivation.append((rule_name, chosen))

    # Expand each symbol in the chosen alternative
    return "".join(_expand_symbol(sym, derivation) for sym in chosen)


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_attack(start_rule: str = "start") -> tuple[str, list]:
    """
    Generate one random SQLi attack string.

    Parameters
    ----------
    start_rule : str
        Grammar rule to start from. Default is "start" (the entry point).
        Can be set to any rule name for targeted generation, e.g.
        "unionAttack" to generate only union-based attacks.

    Returns
    -------
    attack : str
        The raw attack payload, e.g. "' or 1=1 #"

    derivation : list of (rule_name, chosen_alternative) tuples
        Full derivation trace. Each tuple records:
          [0] rule_name         : str  — the non-terminal expanded
          [1] chosen_alternative: list — the symbols selected for it
        Consumed by the slice extractor (Step 2).
    """
    derivation: list = []
    attack = _sample(start_rule, derivation)
    return attack, derivation


def generate_unique_attacks(n: int,
                             start_rule: str = "start",
                             max_attempts: int = 100_000
                             ) -> list[tuple[str, list]]:
    """
    Generate *n* distinct attack strings.

    Parameters
    ----------
    n            : int  — how many unique attacks to produce
    start_rule   : str  — grammar entry point
    max_attempts : int  — safety limit to avoid infinite loops

    Returns
    -------
    list of (attack_string, derivation) tuples, length ≤ n
    """
    seen:    set  = set()
    results: list = []
    attempts = 0

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        attack, derivation = generate_attack(start_rule)
        if attack not in seen:
            seen.add(attack)
            results.append((attack, derivation))

    if len(results) < n:
        print(f"[WARN] Only generated {len(results)}/{n} unique attacks "
              f"after {attempts} attempts — grammar space may be exhausted.")
    return results


def regenerate_subtree(rule_name: str) -> tuple[str, list]:
    """
    Generate a new random sub-derivation starting from a specific rule.
    Used by the mutation engine to swap out parts of an attack.

    Parameters
    ----------
    rule_name : str   grammar rule to start from

    Returns
    -------
    (text, derivation)
    """
    derivation: list = []
    text = _sample(rule_name, derivation)
    return text, derivation


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Random SQLi Attack Sampler — 20 unique attacks")
    print("=" * 65)

    attacks = generate_unique_attacks(20)

    for i, (attack, derivation) in enumerate(attacks, 1):
        context = derivation[0][1][0]   # first symbol of start alternative
        # Print with |...| delimiters instead of repr() quotes so it is
        # visually clear that the payload boundaries are | not ' or "
        # This prevents accidentally copying the Python string delimiters
        # when pasting a payload into test_waf.py or Burp.
        print(f"\n[{i:02d}] |{attack}|")
        print(f"      context : {context}")

    print("\n" + "=" * 65)
    print(f"Total generated : {len(attacks)}")

    # ── Full derivation trace for attack #1 ──────────────────────────────
    print("\n--- Full derivation trace for attack #1 ---")
    attack, derivation = attacks[0]
    print(f"Attack string : {repr(attack)}\n")
    print(f"{'Step':<6} {'Rule':<25} {'Chosen alternative'}")
    print("-" * 65)
    for step, (rule, alt) in enumerate(derivation, 1):
        symbols = " , ".join(
            f"OPT({a[1]})" if isinstance(a, tuple) else str(a)
            for a in alt
        )
        print(f"{step:<6} {rule:<25} {symbols}")

    # ── Descendant count spot-check ───────────────────────────────────────
    print("\n--- Descendant counts (grammar richness) ---")
    for rule in ["start", "booleanAttack", "unionAttack",
                 "wsp", "blank", "terSQuote"]:
        print(f"  {rule:<20} : {_count_descendants(rule):>8,} paths")