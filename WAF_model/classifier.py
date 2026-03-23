"""
classifier.py
=============
Component 3 of ML-Driven: RandomTree and RandomForest classifiers.

Implements the paper's Section 3.3.3 exactly — no external ML libraries,
pure Python from scratch so every line maps directly to the paper.

WHAT THIS COMPONENT DOES
-------------------------
Input  : binary feature matrix (n_attacks × n_slices) + labels ("P"/"B")
         — produced by SliceRegistry.encode_corpus() in slice_extractor.py

Output : for each attack in a new set:
           • bypassing_probability : float  0.0–1.0
           • path_condition        : list of (slice_key, value) pairs
                                     e.g. [("opOr::or", 1), ("wsp:: ", 0)]

These two outputs drive the evolutionary algorithm:
  - probability  → fitness score → which attacks to mutate
  - path_condition → which slices to preserve/avoid during mutation

─────────────────────────────────────────────────────────────────────────────
RANDOMTREE  (paper Section 3.3.3)
─────────────────────────────────────────────────────────────────────────────

A decision tree where at each node we pick the best split from a RANDOM
SUBSET of features (not all features). This is the key difference from
a classical C4.5 tree.

Why random subset?
  The dataset has many columns (one per unique slice, can be hundreds).
  Evaluating every column at every node is slow and risks overfitting.
  Randomly subsetting scales linearly and adds implicit regularisation.

Split criterion: Gini impurity
  Gini(node) = 1 - Σ p_c²
  where p_c = fraction of class c (P or B) at this node.

  We pick the feature+threshold that minimises the weighted average Gini
  of the two child nodes. For binary features (0/1) there is only one
  possible threshold: split on value == 1.

Node types:
  Internal node : feature_index, threshold, left_child, right_child
  Leaf node     : class_label ("P" or "B"), confidence (fraction of class)

Path condition (Definition 3 from paper):
  The sequence of (feature, value) tests from root to a LEAF labelled "P".
  It is a conjunction:  slice_j=1 ∧ slice_k=0 ∧ ...
  Each path condition describes one pattern that predicts bypass.

─────────────────────────────────────────────────────────────────────────────
RANDOMFOREST  (paper Section 3.3.3)
─────────────────────────────────────────────────────────────────────────────

An ensemble of n_trees RandomTree instances, each trained on a bootstrap
sample (random sample WITH replacement) of the training data.

Classification:
  Each tree votes with its confidence score.
  Final probability = average confidence across all trees.

Path condition:
  Computed separately from each tree that predicts "P".
  Overall path condition = CONJUNCTION of all individual path conditions.
  (paper: "the overall path condition for the entire RandomForest is the
   conjunction of the path conditions computed from the trees")

Why ensemble?
  Individual trees are unstable — small data changes flip predictions.
  Averaging many trees cancels out individual biases.
  Cost: ~n_trees × training time. Paper shows the accuracy gain is worth
  it for path condition discovery but NOT for raw bypass count (the
  overhead reduces the number of EA generations possible in a time budget).

─────────────────────────────────────────────────────────────────────────────
PUBLIC API
─────────────────────────────────────────────────────────────────────────────

    from classifier import RandomTree, RandomForest

    # Train
    tree = RandomTree()
    tree.fit(matrix, labels, slice_index)

    forest = RandomForest(n_trees=10)
    forest.fit(matrix, labels, slice_index)

    # Predict probability for a new attack's feature row
    prob = tree.predict_proba(row)      # float 0.0–1.0

    # Get path condition for a new attack
    pc = tree.path_condition(row)
    # → [("opOr::or", 1), ("wsp::/**/", 1), ("blank:: ", 0), ...]

    # Predict + path condition together
    prob, pc = tree.predict(row)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal tree node structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeafNode:
    """
    Terminal node of a decision tree.

    Attributes
    ----------
    label      : "P" | "B"   majority class at this node
    confidence : float        fraction of training samples that are `label`
                              used as the bypassing probability estimate
    n_samples  : int          how many training samples reached this node
    n_bypass   : int          how many of those were "P"
    """
    label:      str
    confidence: float
    n_samples:  int
    n_bypass:   int


@dataclass
class SplitNode:
    """
    Internal (decision) node.

    Attributes
    ----------
    feature_idx  : int          column index in the feature matrix
    feature_key  : str          human-readable slice key "rule::text"
    threshold    : int          always 0 for binary features
                                left  child = feature <= threshold  (absent)
                                right child = feature >  threshold  (present)
    left         : node         subtree when feature == 0 (slice absent)
    right        : node         subtree when feature == 1 (slice present)
    """
    feature_idx: int
    feature_key: str
    threshold:   int
    left:        object   # LeafNode or SplitNode
    right:       object


# ─────────────────────────────────────────────────────────────────────────────
# Gini impurity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gini(labels: list[str]) -> float:
    """
    Gini impurity of a label list.

    Gini = 1 - Σ p_c²

    Pure node  (all same class) → Gini = 0.0
    50/50 split                 → Gini = 0.5  (maximum for binary)

    Parameters
    ----------
    labels : list of "P" or "B"

    Returns
    -------
    float in [0.0, 0.5]
    """
    n = len(labels)
    if n == 0:
        return 0.0
    p_bypass = labels.count("P") / n
    p_block  = 1.0 - p_bypass
    return 1.0 - (p_bypass ** 2 + p_block ** 2)


def _weighted_gini(left_labels: list[str], right_labels: list[str]) -> float:
    """
    Weighted average Gini of two child nodes.

    weighted_gini = (n_left/n_total)*Gini(left) + (n_right/n_total)*Gini(right)

    Lower is better — a perfect split sends all P to one side and all B
    to the other, giving weighted_gini = 0.
    """
    n_total = len(left_labels) + len(right_labels)
    if n_total == 0:
        return 0.0
    w_left  = len(left_labels)  / n_total
    w_right = len(right_labels) / n_total
    return w_left * _gini(left_labels) + w_right * _gini(right_labels)


# ─────────────────────────────────────────────────────────────────────────────
# RandomTree
# ─────────────────────────────────────────────────────────────────────────────

class RandomTree:
    """
    A single decision tree with randomised feature selection at each node.

    Paper reference: Section 3.3.3
    "When selecting an attribute for a tree node, the algorithm chooses
     the best attribute amongst a randomly selected subset of attributes."

    Parameters
    ----------
    max_features  : int | None
        Number of features to consider at each split.
        None → use sqrt(n_features), matching Weka's RandomTree default.
    max_depth     : int | None
        Maximum tree depth. None → grow until pure or min_samples_leaf.
    min_samples_leaf : int
        Minimum samples required at a leaf. Prevents overfitting on tiny
        datasets. Default 1 (same as Weka default).
    random_state  : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_features:     Optional[int] = None,
        max_depth:        Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state:     Optional[int] = None,
    ):
        self.max_features      = max_features
        self.max_depth         = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.random_state      = random_state
        self._root:  object    = None
        self._slice_keys: list = []   # column_index → slice_key mapping

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        matrix:      list[list[int]],
        labels:      list[str],
        slice_index: dict[str, int],
    ) -> "RandomTree":
        """
        Train the tree on the binary feature matrix.

        Parameters
        ----------
        matrix      : list[list[int]]   shape (n_samples, n_features)
                      binary 0/1, one row per attack
        labels      : list[str]         "P" or "B", one per row
        slice_index : dict[str,int]     slice_key → column_index
                      used to map column indices back to human-readable keys

        Returns
        -------
        self  (for chaining)
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        n_features = len(matrix[0]) if matrix else 0

        # Build reverse index: column_idx → slice_key
        self._slice_keys = [""] * n_features
        for key, idx in slice_index.items():
            if idx < n_features:
                self._slice_keys[idx] = key

        # Set max_features default: sqrt(n_features) per Weka RandomTree
        max_feat = self.max_features
        if max_feat is None:
            max_feat = max(1, int(math.sqrt(n_features)))

        # Convert to row indices list for recursive splitting
        indices = list(range(len(matrix)))

        self._root = self._build(
            matrix, labels, indices, depth=0, max_feat=max_feat
        )
        return self

    def _build(
        self,
        matrix:  list[list[int]],
        labels:  list[str],
        indices: list[int],
        depth:   int,
        max_feat: int,
    ) -> object:
        """
        Recursively build the tree.

        ALGORITHM
        ---------
        1. Collect labels for current node's samples
        2. Check stopping conditions:
             a. All labels same           → leaf
             b. max_depth reached         → leaf
             c. Too few samples for split → leaf
        3. Randomly sample max_feat features (without replacement)
        4. For each candidate feature, compute weighted Gini of the split
        5. Pick the feature with lowest weighted Gini
        6. If no improvement over parent Gini → leaf
        7. Split samples into left (feature=0) and right (feature=1)
        8. Recurse on left and right
        """
        node_labels = [labels[i] for i in indices]
        n = len(node_labels)

        # ── stopping conditions ───────────────────────────────────────────
        def make_leaf() -> LeafNode:
            n_bypass = node_labels.count("P")
            if n == 0:
                return LeafNode("B", 0.0, 0, 0)
            confidence = n_bypass / n
            label = "P" if n_bypass >= n / 2 else "B"
            return LeafNode(label, confidence, n, n_bypass)

        # Pure node
        if len(set(node_labels)) == 1:
            return make_leaf()

        # Depth limit
        if self.max_depth is not None and depth >= self.max_depth:
            return make_leaf()

        # Too small to split
        if n < 2 * self.min_samples_leaf:
            return make_leaf()

        n_features = len(matrix[0])

        # ── random feature subset ─────────────────────────────────────────
        candidate_features = random.sample(
            range(n_features), min(max_feat, n_features)
        )

        # ── find best split ───────────────────────────────────────────────
        best_gini    = _gini(node_labels)   # current node's impurity
        best_feature = None

        for feat_idx in candidate_features:
            # Binary feature: split on feat == 0 vs feat == 1
            left_labels  = [labels[i] for i in indices if matrix[i][feat_idx] == 0]
            right_labels = [labels[i] for i in indices if matrix[i][feat_idx] == 1]

            # Skip if split would create too-small leaves
            if (len(left_labels)  < self.min_samples_leaf or
                    len(right_labels) < self.min_samples_leaf):
                continue

            wg = _weighted_gini(left_labels, right_labels)
            if wg < best_gini:
                best_gini    = wg
                best_feature = feat_idx

        # No improvement found → leaf
        if best_feature is None:
            return make_leaf()

        # ── split and recurse ─────────────────────────────────────────────
        left_indices  = [i for i in indices if matrix[i][best_feature] == 0]
        right_indices = [i for i in indices if matrix[i][best_feature] == 1]

        left_child  = self._build(matrix, labels, left_indices,
                                  depth + 1, max_feat)
        right_child = self._build(matrix, labels, right_indices,
                                  depth + 1, max_feat)

        return SplitNode(
            feature_idx = best_feature,
            feature_key = self._slice_keys[best_feature],
            threshold   = 0,
            left        = left_child,    # feature == 0  (slice absent)
            right       = right_child,   # feature == 1  (slice present)
        )

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, row: list[int]) -> float:
        """
        Return the bypass probability for one attack's feature row.

        Walks the tree from root to leaf, following:
          feature == 0  →  left  child
          feature == 1  →  right child

        Returns the leaf's confidence score (fraction of "P" in training
        samples that reached this leaf).

        Parameters
        ----------
        row : list[int]   feature vector, length = n_slices

        Returns
        -------
        float  0.0–1.0   estimated bypass probability
        """
        node = self._root
        while isinstance(node, SplitNode):
            if row[node.feature_idx] <= node.threshold:
                node = node.left   # feature absent
            else:
                node = node.right  # feature present
        return node.confidence  # type: ignore[union-attr]

    def path_condition(self, row: list[int]) -> list[tuple[str, int]]:
        """
        Extract the path condition for one attack.

        The path condition is the sequence of (slice_key, value) decisions
        made from the root to the leaf reached by this row.

        Paper Definition 3:
        "A path condition represents a set of slices that the machine
         learning technique deems to be relevant for the attack's
         classification into blocked or bypassing."
         Represented as a conjunction: ∧_i (s_i = val_i)

        Parameters
        ----------
        row : list[int]   feature vector

        Returns
        -------
        list of (slice_key, value) tuples
        e.g. [("opOr::or", 1), ("wsp::/**/", 1), ("blank:: ", 0)]

        Only returned for paths that end at a "P" leaf — these are the
        conditions associated with BYPASSING.
        Returns [] if the leaf predicts "B".
        """
        path: list[tuple[str, int]] = []
        node = self._root

        while isinstance(node, SplitNode):
            if row[node.feature_idx] <= node.threshold:
                path.append((node.feature_key, 0))   # slice absent
                node = node.left
            else:
                path.append((node.feature_key, 1))   # slice present
                node = node.right

        # Only return path conditions for predicted bypasses
        if isinstance(node, LeafNode) and node.label == "P":
            return path
        return []

    def predict(self, row: list[int]) -> tuple[float, list]:
        """
        Return (bypass_probability, path_condition) together.
        Convenience method used by the EA loop.
        """
        return self.predict_proba(row), self.path_condition(row)

    # ── tree stats ────────────────────────────────────────────────────────────

    def depth(self) -> int:
        """Return the maximum depth of the built tree."""
        def _depth(node) -> int:
            if isinstance(node, LeafNode):
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self._root)

    def n_leaves(self) -> int:
        """Count leaf nodes."""
        def _count(node) -> int:
            if isinstance(node, LeafNode):
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self._root)

    def n_bypass_leaves(self) -> int:
        """Count leaves labelled P (bypass)."""
        def _count(node) -> int:
            if isinstance(node, LeafNode):
                return 1 if node.label == "P" else 0
            return _count(node.left) + _count(node.right)
        return _count(self._root)

    def extract_all_path_conditions(self) -> list[list[tuple[str, int]]]:
        """
        Extract ALL path conditions in the tree that lead to a "P" leaf.

        Used by RQ4 in the paper: counting how many distinct attack patterns
        have been discovered. Each path condition = one pattern.

        Returns
        -------
        list of path conditions, each a list of (slice_key, value) tuples
        """
        results = []

        def _traverse(node, current_path):
            if isinstance(node, LeafNode):
                if node.label == "P":
                    results.append(list(current_path))
                return
            # Go left (feature absent)
            current_path.append((node.feature_key, 0))
            _traverse(node.left, current_path)
            current_path.pop()
            # Go right (feature present)
            current_path.append((node.feature_key, 1))
            _traverse(node.right, current_path)
            current_path.pop()

        _traverse(self._root, [])
        return results


# ─────────────────────────────────────────────────────────────────────────────
# RandomForest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForest:
    """
    Ensemble of RandomTree instances, each trained on a bootstrap sample.

    Paper reference: Section 3.3.3
    "Instead of using only one RandomTree, we extend our technique to make
     use of ensembles of trees produced by RandomForest."

    Bootstrap sampling:
        Each tree is trained on n_samples rows drawn WITH replacement from
        the original training set. On average each bootstrap sample contains
        ~63.2% unique rows (the rest are duplicates). This diversity means
        each tree sees slightly different data, producing different trees
        whose errors are partially uncorrelated — averaging cancels noise.

    Parameters
    ----------
    n_trees          : int    number of trees in the ensemble (default 10)
                              paper uses Weka default; more trees = more
                              path conditions but slower retraining
    max_features     : int|None  passed to each RandomTree (default sqrt)
    max_depth        : int|None  passed to each RandomTree
    min_samples_leaf : int       passed to each RandomTree
    random_state     : int|None  master seed (each tree gets a derived seed)
    """

    def __init__(
        self,
        n_trees:          int = 10,
        max_features:     Optional[int] = None,
        max_depth:        Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state:     Optional[int] = None,
    ):
        self.n_trees          = n_trees
        self.max_features     = max_features
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.trees: list[RandomTree] = []
        self._slice_keys: list = []

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        matrix:      list[list[int]],
        labels:      list[str],
        slice_index: dict[str, int],
    ) -> "RandomForest":
        """
        Train the forest: build n_trees RandomTree instances, each on a
        bootstrap sample of the training data.

        BOOTSTRAP SAMPLING ALGORITHM
        ------------------------------
        Given n training samples, draw n samples WITH replacement:
            bootstrap_indices = [random.choice(range(n)) for _ in range(n)]
        This means some rows appear multiple times, some not at all.
        Each tree therefore sees a different, slightly varied dataset.

        Parameters
        ----------
        matrix      : list[list[int]]   full training matrix
        labels      : list[str]         full label list
        slice_index : dict[str,int]     passed through to each tree

        Returns
        -------
        self
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        n_samples = len(matrix)
        n_features = len(matrix[0]) if matrix else 0

        # Build reverse index for slice key lookup
        self._slice_keys = [""] * n_features
        for key, idx in slice_index.items():
            if idx < n_features:
                self._slice_keys[idx] = key

        self.trees = []

        for t in range(self.n_trees):
            # Bootstrap sample — draw n_samples WITH replacement
            boot_indices = [random.randrange(n_samples)
                            for _ in range(n_samples)]
            boot_matrix = [matrix[i] for i in boot_indices]
            boot_labels = [labels[i] for i in boot_indices]

            # Each tree gets a deterministic seed derived from master seed
            tree_seed = (
                None if self.random_state is None
                else self.random_state + t * 31   # 31 is arbitrary prime
            )

            tree = RandomTree(
                max_features     = self.max_features,
                max_depth        = self.max_depth,
                min_samples_leaf = self.min_samples_leaf,
                random_state     = tree_seed,
            )
            tree.fit(boot_matrix, boot_labels, slice_index)
            self.trees.append(tree)

        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, row: list[int]) -> float:
        """
        Average bypass probability across all trees.

        Paper: "all individual classifications are consolidated by computing
        the average of the prediction confidence values for each class."

        Parameters
        ----------
        row : list[int]   feature vector

        Returns
        -------
        float  0.0–1.0   ensemble bypass probability
        """
        if not self.trees:
            return 0.0
        probs = [tree.predict_proba(row) for tree in self.trees]
        return sum(probs) / len(probs)

    def path_condition(self, row: list[int]) -> list[tuple[str, int]]:
        """
        Compute the CONJUNCTION of path conditions from all trees that
        predict "P" (bypass) for this row.

        Paper: "the overall path condition for the entire RandomForest is
        the conjunction of the path conditions computed from the trees."

        Algorithm:
          1. For each tree, walk root→leaf and record path if leaf = "P"
          2. Collect all (slice_key, value) pairs from all "P" trees
          3. Keep only conditions that are CONSISTENT across trees
             (same slice_key → same value in all trees that mention it)
          4. Return as a deduplicated sorted list

        Parameters
        ----------
        row : list[int]   feature vector

        Returns
        -------
        list of (slice_key, value) tuples — the conjunction
        Returns [] if no tree predicts bypass, or no consistent conditions.
        """
        # Gather path conditions from every tree that predicts "P"
        all_conditions: list[list[tuple[str, int]]] = []
        for tree in self.trees:
            pc = tree.path_condition(row)
            if pc:   # non-empty means the tree predicted "P"
                all_conditions.append(pc)

        if not all_conditions:
            return []

        # Build a dict: slice_key → set of values seen across trees
        condition_votes: dict[str, set[int]] = {}
        for pc in all_conditions:
            for key, val in pc:
                condition_votes.setdefault(key, set()).add(val)

        # Keep only conditions where ALL trees agree on the value
        # (paper: conjunction = only conditions present in ALL trees)
        conjunction = [
            (key, next(iter(vals)))
            for key, vals in condition_votes.items()
            if len(vals) == 1   # unanimous across all trees that mention it
        ]

        # Sort for deterministic output: present slices first, then absent
        conjunction.sort(key=lambda x: (x[1] == 0, x[0]))
        return conjunction

    def predict(self, row: list[int]) -> tuple[float, list]:
        """Return (bypass_probability, path_condition) together."""
        return self.predict_proba(row), self.path_condition(row)

    def extract_all_path_conditions(self) -> list[list[tuple[str, int]]]:
        """
        Extract ALL path conditions from ALL trees in the forest that
        lead to "P" leaves.

        Used for RQ4: how many attack patterns have been discovered?
        More trees → more path conditions (paper Fig 15 shows ~200 for
        RandomForest vs ~50 for RandomTree after 10 generations).

        Returns
        -------
        list of path conditions (deduplicated by frozenset of conditions)
        """
        seen  = set()
        all_pcs = []
        for tree in self.trees:
            for pc in tree.extract_all_path_conditions():
                key = frozenset(pc)
                if key not in seen:
                    seen.add(key)
                    all_pcs.append(pc)
        return all_pcs

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics about the trained forest."""
        if not self.trees:
            return {}
        depths  = [t.depth()   for t in self.trees]
        leaves  = [t.n_leaves() for t in self.trees]
        p_leaves = [t.n_bypass_leaves() for t in self.trees]
        return {
            "n_trees":           len(self.trees),
            "avg_depth":         sum(depths)   / len(depths),
            "avg_leaves":        sum(leaves)   / len(leaves),
            "avg_bypass_leaves": sum(p_leaves) / len(p_leaves),
            "total_path_conditions": len(self.extract_all_path_conditions()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from random_sampler import generate_unique_attacks
    from slice_extractor import SliceRegistry, build_derivation_tree, extract_slices
    from waf_connector import MockWafConnector

    print("=" * 65)
    print("Component 3 — RandomTree & RandomForest Classifier Demo")
    print("=" * 65)

    # ── 1. Generate and label a training corpus ───────────────────────────
    print("\n[1] Generating 60 attacks and labelling with mock WAF...")
    waf    = MockWafConnector()
    raw    = generate_unique_attacks(60)
    corpus = [{"attack": a, "derivation": d} for a, d in raw]
    corpus = waf.label_corpus(corpus)

    n_bypass  = sum(1 for e in corpus if e["label"] == "P")
    n_blocked = sum(1 for e in corpus if e["label"] == "B")
    print(f"    bypass (P): {n_bypass}   blocked (B): {n_blocked}")

    # ── 2. Encode into binary matrix ─────────────────────────────────────
    print("\n[2] Encoding into binary feature matrix...")
    registry = SliceRegistry()
    matrix, labels, slice_index = registry.encode_corpus(corpus)
    print(f"    matrix shape : {len(matrix)} × {len(matrix[0])}")
    print(f"    unique slices: {registry.n_slices}")

    # ── 3. Train RandomTree ───────────────────────────────────────────────
    print("\n[3] Training RandomTree...")
    rt = RandomTree(random_state=42)
    rt.fit(matrix, labels, slice_index)
    print(f"    tree depth   : {rt.depth()}")
    print(f"    total leaves : {rt.n_leaves()}")
    print(f"    bypass leaves: {rt.n_bypass_leaves()}")

    all_pcs_rt = rt.extract_all_path_conditions()
    print(f"    path conditions (P leaves): {len(all_pcs_rt)}")

    # ── 4. Train RandomForest ─────────────────────────────────────────────
    print("\n[4] Training RandomForest (10 trees)...")
    rf = RandomForest(n_trees=10, random_state=42)
    rf.fit(matrix, labels, slice_index)
    stats = rf.stats()
    print(f"    n_trees           : {stats['n_trees']}")
    print(f"    avg depth         : {stats['avg_depth']:.1f}")
    print(f"    avg leaves        : {stats['avg_leaves']:.1f}")
    print(f"    total path conds  : {stats['total_path_conditions']}")

    # ── 5. Predict on training data (sanity check) ────────────────────────
    print("\n[5] Prediction sanity check (first 10 attacks):")
    print(f"\n    {'#':<4} {'True':>6} {'RT prob':>8} {'RF prob':>8} {'RT label':>9} {'RF label':>9}")
    print("    " + "-" * 50)

    correct_rt = 0
    correct_rf = 0

    for i, (row, true_label) in enumerate(zip(matrix, labels)):
        rt_prob = rt.predict_proba(row)
        rf_prob = rf.predict_proba(row)
        rt_pred = "P" if rt_prob >= 0.5 else "B"
        rf_pred = "P" if rf_prob >= 0.5 else "B"

        if rt_pred == true_label: correct_rt += 1
        if rf_pred == true_label: correct_rf += 1

        if i < 10:
            print(f"    {i:<4} {true_label:>6} {rt_prob:>8.3f} {rf_prob:>8.3f} "
                  f"{rt_pred:>9} {rf_pred:>9}")

    print(f"\n    Training accuracy — RandomTree : {correct_rt}/{len(labels)} "
          f"({100*correct_rt/len(labels):.1f}%)")
    print(f"    Training accuracy — RandomForest: {correct_rf}/{len(labels)} "
          f"({100*correct_rf/len(labels):.1f}%)")

    # ── 6. Show path conditions for a bypass attack ───────────────────────
    print("\n[6] Path conditions for first bypass attack:")
    for i, (row, label) in enumerate(zip(matrix, labels)):
        if label == "P":
            rt_prob, rt_pc = rt.predict(row)
            rf_prob, rf_pc = rf.predict(row)

            print(f"\n    Attack   : |{corpus[i]['attack']}|")
            print(f"    RT prob  : {rt_prob:.3f}")
            print(f"    RF prob  : {rf_prob:.3f}")

            print(f"\n    RandomTree path condition ({len(rt_pc)} conditions):")
            for key, val in rt_pc[:8]:
                rule, text = key.split("::", 1)
                presence = "PRESENT" if val == 1 else "absent"
                print(f"      {presence:>8}  {rule:<22} → {repr(text)}")

            print(f"\n    RandomForest path condition ({len(rf_pc)} conditions):")
            for key, val in rf_pc[:8]:
                rule, text = key.split("::", 1)
                presence = "PRESENT" if val == 1 else "absent"
                print(f"      {presence:>8}  {rule:<22} → {repr(text)}")
            break

    # ── 7. All path conditions summary ────────────────────────────────────
    print(f"\n[7] All discovered path conditions:")
    print(f"    RandomTree  : {len(all_pcs_rt)} path conditions")
    print(f"    RandomForest: {len(rf.extract_all_path_conditions())} path conditions")
    print(f"\n    Sample RT path condition #1:")
    if all_pcs_rt:
        for key, val in all_pcs_rt[0]:
            rule, text = key.split("::", 1)
            print(f"      {'∧ ' if key != all_pcs_rt[0][0][0] else '  '}"
                  f"{rule}={val}  ({repr(text)})")