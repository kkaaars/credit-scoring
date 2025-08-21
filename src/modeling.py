from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import json
import os


def train_models(X_train, X_test, y_train, y_test):
    """
    Обучает несколько моделей и возвращает их метрики.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            solver="liblinear", random_state=42, class_weight="balanced"
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(random_state=42, class_weight="balanced"),
    }

    results = {}
    best_model = None
    best_roc_auc = 0
    best_model_name = ""

    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models")
    )
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for name, model in models.items():
        print(f"\nНачинаем обучение {name}...")
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        y_pred = model.predict(X_test)

        results[name] = {
            "roc_auc_score": roc_auc,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        print(f"ROC-AUC для {name}: {roc_auc:.4f}")

        model_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Модель {name} сохранена.")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_model_name = name

    if best_model:
        best_model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        print(f"\nЛучшая модель ({best_model_name}) сохранена как best_model.pkl")

    return results


def save_results(results, output_path):
    """
    Сохраняет результаты обучения в JSON-файл.
    """

    models_dir = os.path.dirname(output_path)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Результаты обучения сохранены в {output_path}")
