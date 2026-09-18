"""
Paramsemble-Class - Ensemble Logistic Regression

A scikit-learn compatible library for advanced classification using
ensemble methods based on combinatorial feature selection.
"""

__version__ = "0.6.0"

from paramsemble_class.core.elr_classifier import ELRClassifier
from paramsemble_class.core.feature_sampler import FeatureSampler
from paramsemble_class.metrics.performance import PerformanceMetrics
from paramsemble_class.scoring.scorer import ModelScorer
from paramsemble_class.sql.generator import SQLGenerator

__all__ = [
    "ELRClassifier",
    "FeatureSampler",
    "PerformanceMetrics",
    "ModelScorer",
    "SQLGenerator",
]
