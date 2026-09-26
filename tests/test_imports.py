#!/usr/bin/env python
"""Import validation: Verify all active production modules import cleanly."""
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

modules = [
    "src",
    "src.models",
    "src.smr_core",
    "experiments.run_final_experiments",
    "experiments.extract_empirical_activations",
    "plots.generate_award_figures",
]

def test_modules_import():
    failed_mods = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            failed_mods.append((mod, str(e)))
    assert not failed_mods, f"Failed imports: {failed_mods}"

if __name__ == "__main__":
    test_modules_import()
    print("ALL SMR MODULES IMPORT CLEANLY!")

