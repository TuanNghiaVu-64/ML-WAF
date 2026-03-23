"""
slice_extractor.py
==================
Component 2 of ML-Driven: turns a derivation trace into slices,
then encodes a collection of labelled attacks into a binary feature matrix.

WHAT IS A SLICE? (paper Definition 1 & 2)
------------------------------------------
Given the derivation tree of an attack, a SLICE is any subtree that
covers a STRICT SUBSET of the attack's leaves (terminal symbols).

Concretely: the derivation trace records every rule expansion as
    (rule_name, chosen_alternative)
Each such expansion node has:
  - a ROOT  : the rule that was expanded          e.g. "unaryTrue"
  - LEAVES  : all terminal strings produced by that subtree
              e.g. ["not", " ", "0"]  →  string "not 0"

A subtree is a valid slice if and only if its leaves are a STRICT subset
of the full attack's leaves — i.e. it does not cover the whole attack.

A MINIMAL SLICE (Definition 2) has exactly one leaf — it maps directly
to one terminal token (e.g. rule "opOr" → "or").

We extract ALL slices (not just minimal), because combinations of slices
are what the decision-tree classifier learns patterns from.

PIPELINE
--------
    derivation                       (from random_sampler)
        │
        ▼
    build_derivation_tree()          internal tree of DerivationNode objects
        │
        ▼
    extract_slices()                 set of Slice objects for one attack
        │
        ▼                            (repeat for every attack in the corpus)
    SliceRegistry.encode_corpus()    binary matrix  (n_attacks × n_slices)
        │
        ▼
    ready for ML classifier          (Component 3)

DATA STRUCTURES
---------------
DerivationNode
    rule      : str          rule name that was expanded
    children  : list         child nodes (DerivationNode) or terminal strings
    leaves    : tuple[str]   all terminal strings under this node (cached)

Slice
    root_rule : str          the rule name at the root of this subtree
    text      : str          concatenated terminal string  e.g. "not 0"
    id        : int          globally unique integer ID (assigned by registry)

SliceRegistry
    Maintains the global slice-ID mapping across the whole corpus.
    encode_corpus() returns:
        matrix : list[list[int]]   shape (n_attacks, n_slices), values 0/1
        labels : list[str]         "P" or "B" per attack
        slice_index : dict         slice_text → column index
"""

from __future__ import annotations
from dataclasses import dataclass, field
from grammar_definition import GRAMMAR


# ── DerivationNode ────────────────────────────────────────────────────────────

@dataclass
class DerivationNode:
    """
    One node in the reconstructed derivation tree.

    Attributes
    ----------
    rule     : str   — the non-terminal rule that was expanded here
    children : list  — mix of DerivationNode (non-terminal children)
                       and str (terminal leaves)
    """
    rule:     str
    children: list = field(default_factory=list)

    @property
    def leaves(self) -> tuple:
        """
        All terminal strings produced by this subtree, left-to-right.
        Cached on first access via _leaves.
        """
        if not hasattr(self, "_leaves"):
            result = []
            for child in self.children:
                if isinstance(child, str):
                    result.append(child)
                else:
                    result.extend(child.leaves)
            self._leaves = tuple(result)
        return self._leaves

    @property
    def text(self) -> str:
        """The concatenated terminal string of this subtree."""
        return "".join(self.leaves)

    def __repr__(self):
        return f"DerivationNode(rule={self.rule!r}, text={self.text!r})"


# ── Slice ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Slice:
    """
    A slice — one subtree of the derivation tree that covers a strict
    subset of the attack's leaves.

    Attributes
    ----------
    root_rule : str   the grammar rule at the root of this subtree
    text      : str   concatenated terminal string  e.g. " not"  "or"  "/**/"
    """
    root_rule: str
    text:      str

    def __repr__(self):
        return f"Slice({self.root_rule!r} → {self.text!r})"


# ── Tree builder ──────────────────────────────────────────────────────────────

def build_derivation_tree(derivation: list) -> DerivationNode:
    """
    Reconstruct a DerivationNode tree from the flat derivation trace
    produced by random_sampler.generate_attack().

    The derivation is a pre-order list:
        [(rule, chosen_alternative), ...]
    where chosen_alternative is a list of symbols (str or OPT tuple).

    We replay the trace using a stack, consuming one entry per non-terminal
    expansion in the same order the sampler produced them.

    Parameters
    ----------
    derivation : list of (rule_name, chosen_alternative) tuples

    Returns
    -------
    DerivationNode   — root of the reconstructed tree
    """
    # Work on a copy so we can pop from the front
    trace = list(derivation)

    def _build() -> DerivationNode:
        """Pop the next expansion from trace and build its subtree."""
        rule, alternative = trace.pop(0)
        node = DerivationNode(rule=rule)

        for symbol in alternative:
            _attach(node, symbol)

        return node

    def _attach(parent: DerivationNode, symbol) -> None:
        """
        Attach one symbol from an alternative to parent.children.

        symbol can be:
          str matching GRAMMAR key  → non-terminal: recurse (_build)
          str NOT in GRAMMAR        → terminal: attach as plain string
          ("OPT", inner)            → was either skipped (no trace entry)
                                      or expanded (trace entry present)
        """
        if isinstance(symbol, tuple):
            # OPT symbol — only present in trace if it was NOT skipped.
            # We detect inclusion by peeking: if the next trace entry's
            # rule matches the inner rule (or any rule in an inner list),
            # consume it; otherwise it was skipped.
            kind, inner = symbol
            assert kind == "OPT"
            inner_rules = (
                [inner] if isinstance(inner, str)
                else [s for s in inner if isinstance(s, str) and s in GRAMMAR]
            )
            if trace and inner_rules and trace[0][0] in inner_rules:
                # It was included — recurse normally for each inner symbol
                if isinstance(inner, str):
                    _attach(parent, inner)
                else:
                    for s in inner:
                        _attach(parent, s)
            # else: it was skipped, attach nothing

        elif symbol in GRAMMAR:
            # Non-terminal — build its subtree
            child = _build()
            parent.children.append(child)

        else:
            # Terminal literal — attach directly as a string
            parent.children.append(symbol)

    root = _build()
    return root


# ── Slice extractor ───────────────────────────────────────────────────────────

def extract_slices(root: DerivationNode) -> set[Slice]:
    """
    Extract all valid slices from a derivation tree.

    A slice is valid when its leaves are a STRICT SUBSET of root.leaves
    (paper Definition 1).  We collect slices from every node in the tree
    by a depth-first traversal.

    Parameters
    ----------
    root : DerivationNode   root of the derivation tree

    Returns
    -------
    set[Slice]   all distinct slices found in this attack's tree
    """
    all_root_leaves = set(root.leaves)   # full leaf set for strict-subset check
    slices: set[Slice] = set()

    def _visit(node: DerivationNode) -> None:
        if isinstance(node, str):
            return  # plain terminal, not a node

        node_leaves = set(node.leaves)

        # Definition 1: strict subset — node must NOT cover all root leaves
        if node_leaves < all_root_leaves:   # '<' means proper subset
            slices.add(Slice(root_rule=node.rule, text=node.text))

        # Recurse into children
        for child in node.children:
            if isinstance(child, DerivationNode):
                _visit(child)

    _visit(root)
    return slices


# ── Slice registry & feature encoder ─────────────────────────────────────────

class SliceRegistry:
    """
    Maintains a global mapping from Slice → integer column index.
    Encodes a labelled corpus of attacks into a binary feature matrix.

    Usage
    -----
        registry = SliceRegistry()
        matrix, labels, index = registry.encode_corpus(labelled_attacks)

    Where labelled_attacks is a list of:
        {
            "attack"     : str,        the raw payload string
            "label"      : "P" | "B",  bypass or blocked
            "derivation" : list,       from generate_attack()
        }
    """

    def __init__(self):
        # slice.text → column index   (insertion-order preserved in Python 3.7+)
        self._index: dict[str, int] = {}

    def _get_or_add(self, sl: Slice) -> int:
        """Return the column index for a slice, adding it if new."""
        key = f"{sl.root_rule}::{sl.text}"
        if key not in self._index:
            self._index[key] = len(self._index)
        return self._index[key]

    @property
    def slice_index(self) -> dict[str, int]:
        """Read-only view: slice_key → column index."""
        return dict(self._index)

    @property
    def n_slices(self) -> int:
        return len(self._index)

    def encode_corpus(
        self,
        labelled_attacks: list[dict],
    ) -> tuple[list[list[int]], list[str], dict[str, int]]:
        """
        Encode a list of labelled attacks into a binary feature matrix.

        Algorithm (paper Section 3.3.2 — Training Set Preparation)
        -----------------------------------------------------------
        For each attack:
          1. Build its derivation tree
          2. Extract all slices
          3. Register any new slices (assign new column IDs)
        Then build the matrix:
          4. For each attack row, set column j = 1 if slice j is present,
             0 otherwise.

        Parameters
        ----------
        labelled_attacks : list of dicts, each with keys:
            "attack"     : str        raw payload
            "label"      : "P"|"B"   ground truth from WAF
            "derivation" : list       from generate_attack()

        Returns
        -------
        matrix      : list[list[int]]   shape (n_attacks, n_slices)
                      binary presence/absence, 0 or 1
        labels      : list[str]         "P" or "B", one per row
        slice_index : dict[str, int]    slice_key → column index
                      (column index matches matrix columns)
        """
        # ── Pass 1: build trees, extract slices, register all slice IDs ──
        attack_slices: list[set[Slice]] = []
        labels: list[str] = []

        for entry in labelled_attacks:
            tree   = build_derivation_tree(entry["derivation"])
            slices = extract_slices(tree)

            # Register every slice found in this attack
            for sl in slices:
                self._get_or_add(sl)

            attack_slices.append(slices)
            labels.append(entry["label"])

        # ── Pass 2: build binary matrix now that all columns are known ───
        # Build reverse lookup: column_index → slice_key
        col_to_key = {v: k for k, v in self._index.items()}
        n_cols = len(self._index)

        matrix: list[list[int]] = []
        for slices in attack_slices:
            # Set of slice keys present in this attack
            present_keys = {f"{sl.root_rule}::{sl.text}" for sl in slices}
            row = [1 if col_to_key[j] in present_keys else 0
                   for j in range(n_cols)]
            matrix.append(row)

        return matrix, labels, self.slice_index


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from random_sampler import generate_attack

    # ── 1. Generate a handful of attacks and mock-label them ─────────────
    #   In the real system these labels come from the WAF response.
    #   Here we mock: anything containing "union" or "sleep" → "P", else "B"
    print("=" * 65)
    print("Component 2 — Slice Extraction & Feature Encoding Demo")
    print("=" * 65)

    def mock_waf(attack: str) -> str:
        """Simulated WAF: naive keyword check, intentionally incomplete."""
        blocked_patterns = ["union", "sleep", "select"]
        low = attack.lower()
        for p in blocked_patterns:
            if p in low:
                return "B"
        return "P"

    # Generate 10 attacks
    corpus = []
    seen   = set()
    while len(corpus) < 10:
        attack, derivation = generate_attack()
        if attack in seen:
            continue
        seen.add(attack)
        label = mock_waf(attack)
        corpus.append({"attack": attack, "label": label, "derivation": derivation})

    # ── 2. Show the derivation tree for attack #1 ────────────────────────
    print(f"\n── Attack #1 ──────────────────────────────────────────────")
    ex = corpus[0]
    print(f"  Payload : {repr(ex['attack'])}")
    print(f"  Label   : {ex['label']}")

    tree = build_derivation_tree(ex["derivation"])

    def _print_tree(node, indent=0):
        prefix = "  " * indent
        if isinstance(node, str):
            print(f"{prefix}TERMINAL: {repr(node)}")
        else:
            print(f"{prefix}[{node.rule}]  →  {repr(node.text)}")
            for child in node.children:
                _print_tree(child, indent + 1)

    print("\n  Derivation tree (first 3 levels):")
    def _print_tree_limited(node, indent=0, max_depth=3):
        if indent > max_depth:
            return
        prefix = "  " * indent
        if isinstance(node, str):
            print(f"{prefix}└─ TERMINAL {repr(node)}")
        else:
            print(f"{prefix}└─ [{node.rule}] → {repr(node.text)}")
            for child in node.children:
                _print_tree_limited(child, indent + 1, max_depth)

    _print_tree_limited(tree)

    # ── 3. Extract slices from attack #1 ────────────────────────────────
    slices = extract_slices(tree)
    print(f"\n  Slices extracted ({len(slices)} total):")
    for sl in sorted(slices, key=lambda s: s.text):
        print(f"    {sl.root_rule:<22} → {repr(sl.text)}")

    # ── 4. Encode the full corpus ────────────────────────────────────────
    print(f"\n── Full corpus encoding ({len(corpus)} attacks) ──────────────")
    registry = SliceRegistry()
    matrix, labels, slice_index = registry.encode_corpus(corpus)

    print(f"\n  Unique slices found across corpus : {registry.n_slices}")
    print(f"  Matrix shape                      : "
          f"{len(matrix)} rows × {len(matrix[0])} cols")

    # Show the matrix as a table (truncated to first 15 slices for readability)
    MAX_SHOW = 15
    slice_keys = list(slice_index.keys())[:MAX_SHOW]
    header_labels = [k.split("::")[1][:8] for k in slice_keys]

    print(f"\n  Binary matrix (first {MAX_SHOW} slices, truncated text):")
    print(f"  {'Attack':>6}  {'Label'}  " +
          "  ".join(f"{h:>8}" for h in header_labels))
    print("  " + "-" * (16 + 10 * MAX_SHOW))

    for i, (row, lbl) in enumerate(zip(matrix, labels)):
        vals = "  ".join(f"{row[j]:>8}" for j in range(min(MAX_SHOW, len(row))))
        payload_short = corpus[i]["attack"][:30]
        print(f"  {i:>6}   {lbl}    {vals}   |{payload_short}|")

    # ── 5. Summary ───────────────────────────────────────────────────────
    n_bypass  = labels.count("P")
    n_blocked = labels.count("B")
    print(f"\n  Label distribution: {n_bypass} bypass (P), {n_blocked} blocked (B)")
    print(f"\n  Sample slice keys (rule::text):")
    for k, v in list(slice_index.items())[:8]:
        print(f"    col {v:>3} : {k}")