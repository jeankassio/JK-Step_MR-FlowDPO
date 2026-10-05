"""Fisher Information + Spectral Analysis for adaptive LoRA rank assignment.

Public API:
    run_fisher_analysis  -- run the full analysis pipeline
    load_fisher_map      -- load a saved fisher_map.json
    save_fisher_map      -- persist analysis results to JSON
"""

from jk_engine.analysis.fisher.analysis import run_fisher_analysis
from jk_engine.analysis.fisher.io import load_fisher_map, save_fisher_map

__all__ = ["run_fisher_analysis", "load_fisher_map", "save_fisher_map"]
