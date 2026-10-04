"""
Objective functions for scenario selection.

Given a component-involvement profile (list of per-step chromosome counts),
return a scalar score to minimise.
"""
from __future__ import annotations

from typing import List


def cost_max(profile: List[int]) -> int:
    """Minimise the maximum number of chromosomes touched in any single step."""
    return max(profile) if profile else 0


def cost_sum(profile: List[int]) -> int:
    """Minimise the total number of chromosome-step involvements."""
    return sum(profile)


def cost_max_then_sum(profile: List[int]) -> tuple:
    """Lexicographic: minimise max first, then sum among ties."""
    return (cost_max(profile), cost_sum(profile))
