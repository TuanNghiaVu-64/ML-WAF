"""
ea_loop.py
==========
Component 4 of ML-Driven: the (µ+λ) Evolutionary Algorithm loop.

Implements Algorithm 2 (main loop), Algorithm 3 (standard offspring
generation for ML-Driven B and D), and Algorithm 4 (adaptive offspring
generation for ML-Driven E) from the paper exactly.

─────────────────────────────────────────────────────────────────────────────
FULL PIPELINE POSITION
─────────────────────────────────────────────────────────────────────────────

  grammar_definition  ←─── defines the search space
         │
  random_sampler      ←─── generates initial population (RAN)
         │
  waf_connector       ←─── labels each attack P or B
         │
  slice_extractor     ←─── derivation → slices → binary feature matrix
         │
  classifier          ←─── RandomTree / RandomForest → bypass probability
         │                  + path conditions per attack
  ┌──────┴──────────────────────────────────────────────────────────────┐
  │  ea_loop  (THIS FILE)                                                │
  │                                                                      │
  │  Population (µ attacks ranked by bypass probability)                 │
  │       │                                                              │
  │       ▼                                                              │
  │  OffspringGen: select parents → mutate slices → λ new attacks        │
  │       │                                                              │
  │       ▼                                                              │
  │  Execute offspring against WAF → new P/B labels                     │
  │       │                                                              │
  │       ▼                                                              │
  │  Archive update → retrain classifier → re-rank population           │
  │       │                                                              │
  │       └──── repeat until time/generation budget exhausted            │
  └──────────────────────────────────────────────────────────────────────┘
         │
  Archive of all bypassing attacks + path conditions
         │
  WAF repair (Component 5)

─────────────────────────────────────────────────────────────────────────────
OFFSPRING GENERATION — HOW MUTATION WORKS
─────────────────────────────────────────────────────────────────────────────

Mutation replaces one slice in a parent attack with an ALTERNATIVE slice
that shares the same grammar root rule but derives a different terminal.

Example:
  Parent attack : "' or/**/ 0=1 #"
  Parent slices : [wsp::"/**/", wsp::" ", opOr::"or", ...]
  Path condition: (opOr::or = 1)   ← opOr must be present

  Pick slice to mutate: wsp::"/**/"
  wsp has alternatives: [blank, inlineCmt]
    blank → " " | "+" | "%20" | "%09" | "%0a" | "%0b" | "%0c" | "%0d" | "%a0"
    inlineCmt → "/**/"   (already chosen — skip)
  Alternative: wsp::"%20"

  Mutant: "' or%20 0=1 #"  (/**/ replaced with %20)

The path condition constrains which slices CAN be replaced:
  - A slice present in the path condition with value=1 must stay present
    (it is a required bypass pattern — swapping it out may lose bypass)
  - A slice present with value=0 must stay absent
  - Slices NOT in the path condition are free to mutate

─────────────────────────────────────────────────────────────────────────────
THREE VARIANTS
─────────────────────────────────────────────────────────────────────────────

  ML-Driven D  (deep / exploitation)
      MAXM = 100 → many mutants per parent, few parents selected
      Good early when few high-probability attacks exist

  ML-Driven B  (broad / exploration)
      MAXM = 10  → few mutants per parent, many parents selected
      Good later when many high-probability attacks exist

  ML-Driven E  (enhanced / adaptive)  ← RECOMMENDED
      Budget per parent ∝ bypass probability (Equation 1 from paper)
      Automatically balances D and B behaviour over time

─────────────────────────────────────────────────────────────────────────────
PUBLIC API
─────────────────────────────────────────────────────────────────────────────

    from ea_loop import MLDrivenEA, EAConfig

    config = EAConfig(
        mu           = 500,      # population size
        lam          = 4000,     # offspring per generation
        max_gen      = 10,       # number of generations
        variant      = "E",      # "B", "D", or "E"
        classifier   = "tree",   # "tree" or "forest"
        phpsessid    = "abc...", # DVWA session
    )

    ea = MLDrivenEA(config)
    results = ea.run()

    # results["bypass_attacks"]   : list of {"attack", "label", "derivation"}
    # results["path_conditions"]  : list of path condition lists
    # results["history"]          : per-generation stats
"""

from __future__ import annotations

import time
import random
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# ── sibling module imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grammar_definition  import GRAMMAR
from random_sampler      import generate_attack, generate_unique_attacks
from slice_extractor     import (SliceRegistry, build_derivation_tree,
                                  extract_slices, Slice)
from classifier          import RandomTree, RandomForest
from waf_connector       import WafConnector, MockWafConnector, DvwaConfig
import mutation


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EAConfig:
    """
    All tunable parameters for the EA loop in one place.

    Paper parameter values (Section 4.4):
        mu           = 500
        lam          = 4000
        archive_cap  = 6000 per class
        maxm_d       = 100   (ML-Driven D)
        maxm_b       = 10    (ML-Driven B)
        sigma        = 0.80  (ML-Driven E bypass threshold)
        n_trees      = 10    (RandomForest)

    For quick local testing use smaller values:
        mu=50, lam=200, max_gen=5
    """
    # ── EA parameters ─────────────────────────────────────────────────────
    mu:          int   = 50       # population size (paper: 500)
    lam:         int   = 200      # offspring per generation (paper: 4000)
    max_gen:     int   = 5        # number of generations to run
    variant:     str   = "E"      # "B" | "D" | "E"
    maxm_b:      int   = 10       # MAXM for ML-Driven B
    maxm_d:      int   = 100      # MAXM for ML-Driven D
    sigma:       float = 0.80     # bypass prob threshold for E variant

    # ── Archive parameters ─────────────────────────────────────────────────
    archive_cap_bypass:  int = 6000   # max P entries in archive (paper: 6000)
    archive_cap_blocked: int = 6000   # max B entries in archive (paper: 6000)

    # ── Classifier parameters ──────────────────────────────────────────────
    classifier:  str  = "tree"    # "tree" | "forest"
    n_trees:     int  = 10        # trees in RandomForest

    # ── Initial population ─────────────────────────────────────────────────
    init_size:   int  = 100       # random attacks to seed population with

    # ── WAF connector ─────────────────────────────────────────────────────
    use_mock:    bool = False
    phpsessid:   str  = "f59690908df4860b89bddd3eaba6922c"
    host:        str  = "http://localhost:9003"

    # ── Output ────────────────────────────────────────────────────────────
    verbose:     bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Mutation engine — offspring generation
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Archive management
# ─────────────────────────────────────────────────────────────────────────────

class Archive:
    """
    Keeps track of ALL attacks generated across all generations.

    Paper Section 3.3.4:
    "The archive is a second population used to keep track of all tests
     being generated across the generations."

    Capped at archive_cap_bypass P entries and archive_cap_blocked B
    entries — if over cap, oldest entries are dropped (FIFO).
    The archive is the training set for the classifier at each generation.
    """

    def __init__(self, cap_bypass: int = 6000, cap_blocked: int = 6000):
        self.cap_bypass  = cap_bypass
        self.cap_blocked = cap_blocked
        self._bypass:  list[dict] = []   # P entries (most recent last)
        self._blocked: list[dict] = []   # B entries (most recent last)

    def update(self, entries: list[dict]) -> None:
        """Add new entries, maintaining the cap (FIFO eviction)."""
        for e in entries:
            if e["label"] == "P":
                self._bypass.append(e)
            else:
                self._blocked.append(e)
        # Keep only the most recent entries within cap
        if len(self._bypass)  > self.cap_bypass:
            self._bypass  = self._bypass[-self.cap_bypass:]
        if len(self._blocked) > self.cap_blocked:
            self._blocked = self._blocked[-self.cap_blocked:]

    def as_training_set(self) -> tuple[list[dict], list[str]]:
        """Return all archive entries as (corpus, labels)."""
        all_entries = self._bypass + self._blocked
        return all_entries

    @property
    def n_bypass(self)  -> int: return len(self._bypass)
    @property
    def n_blocked(self) -> int: return len(self._blocked)
    @property
    def n_total(self)   -> int: return len(self._bypass) + len(self._blocked)

    def all_bypass_attacks(self) -> list[dict]:
        """Return all confirmed bypass attacks — the final output."""
        return list(self._bypass)


# ─────────────────────────────────────────────────────────────────────────────
# Main EA class
# ─────────────────────────────────────────────────────────────────────────────

class MLDrivenEA:
    """
    The ML-Driven evolutionary algorithm.

    Implements Algorithm 2 from the paper (Section 3.3.4) with all three
    variants (B, D, E) and both classifiers (RandomTree, RandomForest).

    GENERATION LOOP (Algorithm 2)
    ──────────────────────────────
    Init:
      1. Generate init_size random attacks (RAN)
      2. Send to WAF → get P/B labels
      3. Add to archive
      4. Encode archive → binary matrix
      5. Train classifier DT on matrix
      6. Rank population by bypass probability from DT

    Each generation:
      7.  Generate λ offspring via OFFSPRINGSGEN (Alg 3 or 4)
      8.  Send offspring to WAF → P/B labels
      9.  Add to archive (update)
      10. Retrain classifier DT on updated archive
      11. Select µ fittest from population ∪ offspring → new population
      12. Log generation stats
      13. Repeat until max_gen reached
    """

    def __init__(self, config: EAConfig):
        self.cfg = config

        # ── WAF connector ─────────────────────────────────────────────────
        if config.use_mock:
            self.waf = MockWafConnector()
        else:
            self.waf = WafConnector(DvwaConfig(
                host      = config.host,
                phpsessid = config.phpsessid,
            ))

        # ── Slice registry (persistent across generations) ────────────────
        self.registry = SliceRegistry()

        # ── Archive ───────────────────────────────────────────────────────
        self.archive = Archive(
            cap_bypass  = config.archive_cap_bypass,
            cap_blocked = config.archive_cap_blocked,
        )

        # ── Classifier (rebuilt each generation) ─────────────────────────
        self.classifier = None   # set after first training

        # ── Population: list of attack dicts with "prob" added ────────────
        self.population: list[dict] = []

        # ── History for plotting / analysis ──────────────────────────────
        self.history: list[dict] = []

    # ── internal: encode + train ──────────────────────────────────────────────

    def _train_classifier(self) -> None:
        """
        Encode the current archive into a binary matrix and train the
        classifier. Called at the start and end of each generation.

        Paper Algorithm 2 lines 6-7 (init) and lines 14-15 (per generation):
            trainData ← transform(archive)
            DT        ← learnClassifier(trainData)
        """
        corpus = self.archive.as_training_set()

        if not corpus:
            return

        # Need at least one of each class to train meaningfully
        labels_raw = [e["label"] for e in corpus]
        if "P" not in labels_raw or "B" not in labels_raw:
            return   # cannot train on single-class data

        matrix, labels, slice_index = self.registry.encode_corpus(corpus)

        if self.cfg.classifier == "forest":
            clf = RandomForest(
                n_trees      = self.cfg.n_trees,
                random_state = 42,
            )
        else:
            clf = RandomTree(random_state=42)

        clf.fit(matrix, labels, slice_index)
        self.classifier = clf

    # ── internal: rank population ─────────────────────────────────────────────

    def _rank_population(self) -> None:
        """
        Assign bypass probability to each individual in the population.

        Paper Algorithm 2 line 8: rankTests(P, DT)

        Probability is assigned by encoding each attack's slices and
        running predict_proba() on the trained classifier.
        The population is then sorted descending by probability.
        """
        if self.classifier is None:
            # No classifier yet — assign 0.5 to everyone
            for entry in self.population:
                entry["prob"] = 0.5
            return

        # We need the feature row for each attack.
        # encode_corpus handles new attacks by extending the registry —
        # but here we just need to predict, so we build rows manually
        # from whatever slices are in the registry already.
        slice_index = self.registry.slice_index
        n_cols      = len(slice_index)

        for entry in self.population:
            try:
                tree_obj = build_derivation_tree(entry["derivation"])
                slices   = extract_slices(tree_obj)
                present  = {f"{s.root_rule}::{s.text}" for s in slices}
                row = [1 if k in present else 0
                       for k in sorted(slice_index, key=slice_index.get)]
                entry["feature_row"] = row
                entry["prob"] = self.classifier.predict_proba(row)
            except Exception:
                entry["prob"]        = 0.0
                entry["feature_row"] = None

        # Sort descending by bypass probability
        self.population.sort(key=lambda e: e["prob"], reverse=True)

    # ── internal: offspring generation ───────────────────────────────────────

    def _get_path_condition(self, entry: dict) -> list[tuple[str, int]]:
        """
        Get path condition for one attack from the current classifier.
        Returns [] if no classifier or attack has no bypass path.
        """
        if self.classifier is None:
            return []

        slice_index = self.registry.slice_index
        try:
            tree_obj = build_derivation_tree(entry["derivation"])
            slices   = extract_slices(tree_obj)
            present  = {f"{s.root_rule}::{s.text}" for s in slices}
            row = [1 if k in present else 0
                   for k in sorted(slice_index, key=slice_index.get)]
            return self.classifier.path_condition(row)
        except Exception:
            return []

    def _generate_offspring(self) -> list[dict]:
        """
        Produce λ offspring by calling the mutation engine (Component 7).
        """
        # Ensure entries have "prob" and "derivation" which mutation.py expects
        # and "feature_row" if available.
        # mutation.adaptive_offspring_gen(population, lambda_val, classifier, sigma)
        
        # We need to make sure 'prob' is set for all individuals
        # (already done by _rank_population at line 471)
        
        offspring = mutation.adaptive_offspring_gen(
            population       = self.population,
            lambda_val       = self.cfg.lam,
            classifier_model = self.classifier,
            sigma            = self.cfg.sigma
        )
        return offspring

    def _execute(self, attacks: list[dict]) -> list[dict]:
        """
        Send attacks to WAF and add "label" to each dict.
        Deduplicates by attack string before sending.
        """
        # Deduplicate
        seen     = set()
        to_send  = []
        for e in attacks:
            if e["attack"] not in seen:
                seen.add(e["attack"])
                to_send.append(e)

        labelled = self.waf.label_corpus(to_send)
        return labelled

    # ── internal: elitist selection ───────────────────────────────────────────

    def _select(self, parents: list[dict], offspring: list[dict]) -> list[dict]:
        """
        Elitist (µ+λ) selection: keep top µ from parents ∪ offspring.

        Paper Algorithm 2 line 17: P ← SELECT(P ∪ O)

        Both lists must already have "prob" assigned.
        """
        combined = parents + offspring
        combined.sort(key=lambda e: e.get("prob", 0.0), reverse=True)
        return combined[:self.cfg.mu]

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full ML-Driven EA loop.

        Returns
        -------
        dict with keys:
            "bypass_attacks"   : list[dict]  all confirmed bypass attacks
            "path_conditions"  : list        all discovered path conditions
            "history"          : list[dict]  per-generation statistics
        """
        cfg = self.cfg
        log = self._log

        log("=" * 65)
        log(f"ML-Driven EA  variant={cfg.variant}  "
            f"classifier={cfg.classifier}  "
            f"µ={cfg.mu}  λ={cfg.lam}  generations={cfg.max_gen}")
        log("=" * 65)

        # ── INIT: generate initial population ──────────────────────────────
        log(f"\n[INIT] Generating {cfg.init_size} initial attacks...")
        init_raw    = generate_unique_attacks(cfg.init_size)
        init_corpus = [{"attack": a, "derivation": d} for a, d in init_raw]

        log(f"[INIT] Sending {len(init_corpus)} attacks to WAF...")
        init_corpus = self._execute(init_corpus)

        # Add to archive (Algorithm 2 line 4)
        self.archive.update(init_corpus)
        log(f"[INIT] Archive: {self.archive.n_bypass} bypass, "
            f"{self.archive.n_blocked} blocked")

        # Train initial classifier (Algorithm 2 lines 6-7)
        log("[INIT] Training initial classifier...")
        self._train_classifier()

        # Seed population: keep top µ from initial corpus by prob
        self.population = init_corpus
        self._rank_population()
        self.population = self.population[:cfg.mu]

        log(f"[INIT] Population seeded: {len(self.population)} individuals")
        log(f"[INIT] Top-5 bypass probs: "
            f"{[round(e['prob'],3) for e in self.population[:5]]}")

        # ── GENERATION LOOP ────────────────────────────────────────────────
        for gen in range(1, cfg.max_gen + 1):
            gen_start = time.time()
            log(f"\n{'─'*65}")
            log(f"Generation {gen}/{cfg.max_gen}")

            # Step 1: generate offspring (Algorithms 3 or 4)
            log(f"  Generating offspring (variant={cfg.variant})...")
            offspring_unlabelled = self._generate_offspring()
            log(f"  Generated {len(offspring_unlabelled)} candidate mutants")

            if not offspring_unlabelled:
                log("  [WARN] No offspring generated — population may be exhausted")
                break

            # Step 2: execute offspring against WAF (Algorithm 2 line 11)
            log(f"  Sending offspring to WAF...")
            offspring = self._execute(offspring_unlabelled)

            n_new_bypass  = sum(1 for e in offspring if e["label"] == "P")
            n_new_blocked = sum(1 for e in offspring if e["label"] == "B")
            log(f"  Offspring results: {n_new_bypass} bypass, "
                f"{n_new_blocked} blocked")

            # Step 3: update archive (Algorithm 2 line 12)
            self.archive.update(offspring)
            log(f"  Archive: {self.archive.n_bypass} bypass total, "
                f"{self.archive.n_blocked} blocked total")

            # Step 4: retrain classifier (Algorithm 2 lines 14-15)
            log(f"  Retraining classifier...")
            self._train_classifier()

            # Assign probabilities to offspring for selection
            for entry in offspring:
                try:
                    slice_index = self.registry.slice_index
                    tree_obj    = build_derivation_tree(entry["derivation"])
                    slices      = extract_slices(tree_obj)
                    present     = {f"{s.root_rule}::{s.text}" for s in slices}
                    row = [1 if k in present else 0
                           for k in sorted(slice_index, key=slice_index.get)]
                    entry["prob"] = self.classifier.predict_proba(row) \
                        if self.classifier else 0.0
                except Exception:
                    entry["prob"] = 0.0

            # Step 5: elitist selection (Algorithm 2 line 17)
            self.population = self._select(self.population, offspring)
            self._rank_population()

            # ── generation stats ───────────────────────────────────────────
            elapsed = time.time() - gen_start
            top_prob = self.population[0]["prob"] if self.population else 0.0

            # Count path conditions from current classifier
            n_pcs = 0
            if self.classifier is not None:
                try:
                    n_pcs = len(self.classifier.extract_all_path_conditions())
                except Exception:
                    n_pcs = 0

            gen_stats = {
                "generation":    gen,
                "n_bypass":      self.archive.n_bypass,
                "n_blocked":     self.archive.n_blocked,
                "new_bypass":    n_new_bypass,
                "new_blocked":   n_new_blocked,
                "top_prob":      top_prob,
                "n_pcs":         n_pcs,
                "elapsed_sec":   elapsed,
            }
            self.history.append(gen_stats)

            log(f"  Top bypass prob : {top_prob:.3f}")
            log(f"  Path conditions : {n_pcs}")
            log(f"  Generation time : {elapsed:.1f}s")

        # ── collect final results ──────────────────────────────────────────
        bypass_attacks = self.archive.all_bypass_attacks()

        path_conditions = []
        if self.classifier is not None:
            try:
                path_conditions = self.classifier.extract_all_path_conditions()
            except Exception:
                pass

        log(f"\n{'='*65}")
        log(f"DONE — {len(bypass_attacks)} distinct bypass attacks found")
        log(f"       {len(path_conditions)} path conditions discovered")

        return {
            "bypass_attacks":  bypass_attacks,
            "path_conditions": path_conditions,
            "history":         self.history,
        }

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Usage:
    #   python ea_loop.py                    → real DVWA, variant E, tree
    #   python ea_loop.py mock               → mock WAF
    #   python ea_loop.py mock B forest      → mock WAF, variant B, forest
    #   python ea_loop.py <sessid>           → real DVWA with session id
    #   python ea_loop.py <sessid> E forest  → real DVWA, variant E, forest

    args     = sys.argv[1:]
    use_mock = len(args) > 0 and args[0] == "mock"
    sessid   = args[0] if (args and not use_mock) \
               else "f59690908df4860b89bddd3eaba6922c"
    variant  = args[1].upper() if len(args) > 1 else "E"
    clf_type = args[2].lower() if len(args) > 2 else "tree"

    if variant not in ("B", "D", "E"):
        print(f"Unknown variant {variant!r}, defaulting to E")
        variant = "E"
    if clf_type not in ("tree", "forest"):
        print(f"Unknown classifier {clf_type!r}, defaulting to tree")
        clf_type = "tree"

    config = EAConfig(
        # Small values for quick demo — increase for real runs
        mu         = 20,
        lam        = 50,
        max_gen    = 3,
        init_size  = 30,
        variant    = variant,
        classifier = clf_type,
        use_mock   = use_mock,
        phpsessid  = sessid,
        verbose    = True,
    )

    ea      = MLDrivenEA(config)
    results = ea.run()

    # ── print final bypass attacks ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Bypass attacks ({len(results['bypass_attacks'])} total):")
    for i, e in enumerate(results["bypass_attacks"][:15], 1):
        print(f"  [{i:02d}] |{e['attack']}|")
    if len(results["bypass_attacks"]) > 15:
        print(f"  ... and {len(results['bypass_attacks'])-15} more")

    # ── print path conditions ──────────────────────────────────────────────
    print(f"\nPath conditions ({len(results['path_conditions'])} total):")
    for i, pc in enumerate(results["path_conditions"][:5], 1):
        conditions = "  ∧  ".join(
            f"{k.split('::')[0]}={'PRESENT' if v==1 else 'absent'}"
            for k, v in pc[:4]
        )
        print(f"  [{i}] {conditions}")

    # ── print generation history ───────────────────────────────────────────
    print(f"\nGeneration history:")
    print(f"  {'Gen':>4}  {'Bypass':>8}  {'Blocked':>8}  "
          f"{'New P':>7}  {'PCs':>6}  {'Time':>7}")
    print("  " + "-" * 50)
    for h in results["history"]:
        print(f"  {h['generation']:>4}  {h['n_bypass']:>8}  "
              f"{h['n_blocked']:>8}  {h['new_bypass']:>7}  "
              f"{h['n_pcs']:>6}  {h['elapsed_sec']:>6.1f}s")