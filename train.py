"""
Скрипт обучения модели скоринга.
Запуск: python train.py
"""
from src.model import train


if __name__ == "__main__":
    print("=" * 55)
    print("  AgriScore KZ — Training Pipeline v2")
    print("  Ensemble (XGBoost + LightGBM) + Rule-based scoring")
    print("=" * 55)
    models, encoders, oblast_stats, auc = train()
    print(f"\nDone. Ensemble ROC-AUC = {auc:.4f}")
    print(f"Models: {list(models.keys())}")
