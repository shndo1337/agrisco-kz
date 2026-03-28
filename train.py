"""
Скрипт обучения модели скоринга.
Запуск: py train.py
"""
from src.model import train


if __name__ == "__main__":
    print("=" * 50)
    print("  AgriScore KZ — Training Pipeline")
    print("=" * 50)
    model, encoders, auc = train()
    print(f"\nDone. ROC-AUC = {auc:.4f}")
