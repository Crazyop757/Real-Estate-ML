import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Prices"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="teal")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price (Lakh ₹)")
    plt.ylabel("Predicted Price (Lakh ₹)")
    plt.title(f"📌 {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residual_distribution(y_true, y_pred, title="Residual Error Distribution"):
    """Plot distribution of residuals (prediction errors)."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color='orange')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"📊 {title}")
    plt.xlabel("Prediction Error (Lakh ₹)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importances"):
    """
    Plot top N feature importances for tree-based models (e.g. RandomForest, XGBoost).
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("❌ This model does not support feature_importances_")
        return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title(f"📌 Top {top_n} {title}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
