"""
Experiment package for code translation experiments.

This package provides modules for:
1. Dataset management (collecting examples from Rosetta Code)
2. Test case generation
3. Pseudocode generation
4. Code translation across languages
5. Evaluation of translations
6. Experiment reporting
"""

# Import main Experiment class
import logging

from experiment.core import Experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__all__ = ["Experiment"]
