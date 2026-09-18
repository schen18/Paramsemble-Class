"""
Core module for ELR package.

Contains the main ELRClassifier class and feature sampling functionality.
"""

from paramsemble_class.core.elr_classifier import ELRClassifier
from paramsemble_class.core.feature_sampler import FeatureSampler
from paramsemble_class.core.baseline_model import BaselineModel
from paramsemble_class.core.constituent_model import ConstituentModel

__all__ = [
    "ELRClassifier",
    "FeatureSampler",
    "BaselineModel",
    "ConstituentModel",
]
