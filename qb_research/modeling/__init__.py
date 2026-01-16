"""
Modeling module for QB research prediction models.

This module contains functions for:
- Wins prediction using ridge regression
- Payment prediction using logistic regression
- Enhanced payment prediction with confusion matrix analysis
- Payment probability surfaces and KNN-based models
"""

from .prediction_models import (
    ridge_regression_payment_prediction,
    wins_prediction_linear_ridge,
    payment_prediction_logistic_ridge,
    compare_injury_projection_predictiveness
)

from .surface_models import (
    create_payment_probability_surface,
    create_simple_knn_payment_surface,
    run_all_simple_knn_surfaces
)
