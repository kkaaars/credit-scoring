import pandas as pd
from src.data_processing import process_data
from src.modeling import train_models, save_results
import os


def main():
    """
    Основная функция для запуска всего пайплайна.
    """
    raw_data_path = "data/raw/cs-training.csv"
    processed_data_path = "data/processed"
    results_path = "models/training_results.json"

    process_data(raw_data_path, processed_data_path)

    X_train = pd.read_csv(os.path.join(processed_data_path, "X_train_processed.csv"))
    X_test = pd.read_csv(os.path.join(processed_data_path, "X_test_processed.csv"))
    y_train = pd.read_csv(os.path.join(processed_data_path, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_data_path, "y_test.csv")).squeeze()

    results = train_models(X_train, X_test, y_train, y_test)

    save_results(results, results_path)


if __name__ == "__main__":
    main()
