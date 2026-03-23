"""
mutation.py
===========
Component 7: Adaptive Offspring Generation and Mutation.

This module implements the mutation engine for the ML-WAF project. 
It follows Algorithm 4 (Adaptive) where the mutation budget per parent
is dynamically proportional to its bypassing probability.
"""

import random
import copy
from typing import Optional, List, Dict, Any

import random
import copy
from typing import Any, List, Dict, Optional, Union, Set

from slice_extractor import (
    Slice, 
    DerivationNode, 
    build_derivation_tree, 
    extract_slices
)
from random_sampler import regenerate_subtree
from grammar_definition import GRAMMAR
from classifier import RandomTree, RandomForest

def _find_all_occurrences(root: DerivationNode, target_rule: str, target_text: str, results: List[DerivationNode]):
    """Recursively find all nodes matching the slice signature."""
    if root.rule == target_rule and root.text == target_text:
        results.append(root)
    for child in root.children:
        if isinstance(child, DerivationNode):
            _find_all_occurrences(child, target_rule, target_text, results)

def _tree_to_derivation(node: DerivationNode) -> List[tuple]:
    """Reconstruct a derivation trace from a DerivationNode tree."""
    trace = []
    def _visit(n: DerivationNode):
        if isinstance(n, DerivationNode):
            symbols = []
            for child in n.children:
                if isinstance(child, str):
                    symbols.append(child)
                else:
                    symbols.append(child.rule)
            trace.append((n.rule, symbols))
            for child in n.children:
                if isinstance(child, DerivationNode):
                    _visit(child)
    _visit(node)
    return trace

def calculate_mutation_budgets(top_attacks: List[Dict[str, Any]], lambda_val: int) -> List[int]:
    """Distribute offspring budget proportional to bypass probability."""
    total_prob = sum(a.get("prob", 0.0) for a in top_attacks)
    
    if total_prob == 0:
        # Fallback to equal distribution if no one has probability
        avg = lambda_val // len(top_attacks)
        return [avg] * len(top_attacks)

    budgets = []
    for a in top_attacks:
        # Equation 1: mt = (P(t) / sum P(x)) * lambda
        mt = int((a.get("prob", 0.0) / total_prob) * lambda_val)
        budgets.append(max(mt, 1)) # Ensure at least 1 if selected

    # Adjust for rounding errors to match exactly lambda_val
    diff = lambda_val - sum(budgets)
    if diff > 0:
        for i in range(diff):
            budgets[i % len(budgets)] += 1
    elif diff < 0:
        for i in range(abs(diff)):
            idx = i % len(budgets)
            if budgets[idx] > 1:
                budgets[idx] -= 1

    return budgets

def _invalidate_cache(node: DerivationNode):
    """Clear cached leaves in the node and all its subnodes recursively."""
    if hasattr(node, "_leaves"):
        delattr(node, "_leaves")
    for child in node.children:
        if isinstance(child, DerivationNode):
            _invalidate_cache(child)

def _perform_mutation_on_node(node: DerivationNode):
    """In-place replacement of a node's content with a new random derivation."""
    _, new_sub_trace = regenerate_subtree(node.rule)
    new_subtree = build_derivation_tree(new_sub_trace)
    node.children = new_subtree.children
    # Cache invalidation is handled by the caller on the root node

def adaptive_offspring_gen(population: List[Dict[str, Any]], 
                            lambda_val: int, 
                            classifier_model: Union[RandomTree, RandomForest], 
                            sigma: float = 0.8) -> List[Dict[str, Any]]:
    """
    Generate lambda_val offspring from the current population using 
    adaptive mutation (Algorithm 4 from the paper).
    
    Inputs:
    - population: List of payload dicts with 'derivation' and 'probability'
    - lambda_val: Total number of new offspring to generate
    - classifier_model: The trained RandomForest to get path conditions
    - sigma: Selection threshold for parents (default 0.8)
    """
    offsprings = []
    
    # 1. Selection (T set): Filter parents with high bypassing probability
    T = [a for a in population if a.get("prob", 0.0) >= sigma]
    if not T:
        # Fallback: if no one meets sigma, take top 10%
        sorted_pop = sorted(population, key=lambda x: x.get("prob", 0.0), reverse=True)
        T = sorted_pop[:max(1, len(population) // 10)]

    # 2. Assign mutation budget per parent
    budgets = calculate_mutation_budgets(T, lambda_val)
    
    for parent, mt in zip(T, budgets):
        # Build tree once for this parent
        root_orig = build_derivation_tree(parent["derivation"])
        
        # Get all slices (original function returns a set)
        current_slices = list(extract_slices(root_orig))
        
        # Get Path Condition (features that the AI thinks lead to bypass)
        # We assume the caller or the entry has the row or we rebuild it.
        # In ea_loop.py, classifier has a path_condition(row) method.
        row = parent.get("feature_row")
        
        # Get Path Condition (features that the AI thinks lead to bypass)
        if row is not None:
             # val == 1 means this slice IS required for bypass.
             # We should avoid mutating these.
             path_cond = classifier_model.path_condition(row)
             required_slices = {key for key, val in path_cond if val == 1}
        else:
             required_slices = set()

        # Filter slices that are candidates for mutation (NOT in required path condition)
        candidate_slices = [s for s in current_slices if f"{s.root_rule}::{s.text}" not in required_slices]
        if not candidate_slices:
             # Fallback: if all slices are required (rare), allow any mutation to avoid getting stuck
             candidate_slices = current_slices

        current_mt = int(mt)
        
        # Track seen texts in this generation to avoid duplicates
        # (Alternatively, pass in a global seen set from ea_loop.py)
        seen_in_batch = {o["attack"] for o in offsprings}
        seen_in_batch.add(parent["attack"])
        
        # Safety limit to avoid infinite attempts if parent is too "rigid"
        max_attempts = current_mt * 10
        attempts = 0

        while current_mt > 0 and candidate_slices and attempts < max_attempts:
            attempts += 1
            # Pick a slice type to mutate
            s = random.choice(candidate_slices)
            
            root_copy = copy.deepcopy(root_orig)
            
            # Resolve ambiguity: find all occurrences of this slice type
            occurrences = []
            _find_all_occurrences(root_copy, s.root_rule, s.text, occurrences)
            
            if not occurrences:
                candidate_slices.remove(s)
                continue
            
            # Pick a random occurrence to mutate
            target_node = random.choice(occurrences)
            _perform_mutation_on_node(target_node)
            _invalidate_cache(root_copy) 
            
            new_text = root_copy.text
            if new_text not in seen_in_batch:
                offsprings.append({
                    "attack": new_text,
                    "parent_prob": parent.get("prob", 0.0),
                    "derivation": _tree_to_derivation(root_copy)
                })
                seen_in_batch.add(new_text)
                current_mt -= 1
            else:
                # If this specific slice type always produces duplicates, maybe try another
                # But we don't remove it yet to preserve budget attempts
                pass
    
    return offsprings

if __name__ == "__main__":
    # Small test for reversibility
    from random_sampler import generate_attack
    print("Testing reversibility: build_tree -> to_derivation -> build_tree")
    for _ in range(5):
        orig_attack, orig_trace = generate_attack()
        tree1 = build_derivation_tree(orig_trace)
        trace2 = _tree_to_derivation(tree1)
        tree2 = build_derivation_tree(trace2)
        
        if tree1.text != tree2.text:
            print(f"FAILED: |{tree1.text}| != |{tree2.text}|")
        else:
            print(f"SUCCESS: {len(orig_trace)} nodes, text match.")
    
    print("\nComponent 7: Adaptive Mutation Engine Loaded.")
