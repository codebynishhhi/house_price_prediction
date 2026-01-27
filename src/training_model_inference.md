# Models Trained

The following regression models were trained and evaluated on the Ames Housing dataset:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor

Each model was evaluated using:

1. Train R²
2. Test R²
3. RMSE
4. Generalization gap (|Train R² − Test R²|)

# Model Performance Summary

| Model             | Train R²  | Test R²   | Generalization Gap |
| ----------------- | --------- | --------- | ------------------ |
| Linear Regression | 0.928     | 0.868     | 0.060              |
| Ridge Regression  | 0.928     | 0.872     | 0.056              |
| Lasso Regression  | 0.928     | 0.868     | 0.060              |
| **Random Forest** | **0.984** | **0.926** | **0.058**          |


# Model Selection Logic

The best model was selected based on:
1. Highest Test R² score
2. Controlled generalization gap to avoid overfitting

Although Random Forest has a slightly higher generalization gap compared to linear models, it provides a significantly higher Test R², indicating superior predictive performance on unseen data.

Linear models provide strong baselines but underperform on complex feature interactions.

Tree-based models capture non-linear patterns better in housing price data.

A structured selection strategy ensures reproducible and explainable model choice.

# Final Selected Model

Random Forest Regressor
- Test R²: 0.926
- Lowest RMSE among all models
- Strong performance on non-linear relationships
- Robust to feature interactions and noise

# Artifacts Generated

Trained model saved at:
- artifacts/model/best_model.pkl

Evaluation metrics saved at:
- artifacts/reports/model_metrics.json