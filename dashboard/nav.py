"""Navigation registry — single source of truth for page labels and order.

Both render_sidebar() and render_selected_page() in pages.py consume this
enum so that adding / renaming / reordering a page only requires a change
here.
"""

from __future__ import annotations

from enum import Enum


class Page(str, Enum):
    """Each member's *value* is the human-readable label shown in the sidebar."""

    # RAW_DATA         = "Raw Data"
    # PROCESSING       = "Processing Pipeline"
    # TRAIN_EVALUATE   = "Train & Evaluate"
    ONE_STEP         = "Chapter 1: One-Step Anticipation"
    MULTI_STEP       = "Chapter 2: Multi-Step Averaging"
    # SYMBOLIC         = "Symbolic Regression"
    # TRANSFER         = "Transfer Learning"

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def labels(cls) -> list[str]:
        """Ordered list of sidebar labels — preserves enum declaration order."""
        return [p.value for p in cls]

    @classmethod
    def from_label(cls, label: str) -> "Page":
        """Look up a Page by its sidebar label string."""
        return cls(label)
