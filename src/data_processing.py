import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_data(input_path, output_path):
    """
    Загружает, обрабатывает и разделяет данные на тренировочную и тестовую выборки.

    Args:
        input_path (str): Путь к исходному CSV-файлу.
        output_path (str): Путь для сохранения обработанных данных.
    """
    print("Начинаем предобработку данных...")

    df = pd.read_csv(input_path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    df = df[df["RevolvingUtilizationOfUnsecuredLines"] < 1]
    df = df[df["age"] > 0]
    df = df[df["DebtRatio"] < 1]

    df["NumberOfTime30-59DaysPastDueNotWorse"] = df[
        "NumberOfTime30-59DaysPastDueNotWorse"
    ].apply(lambda x: 96 if x > 95 else x)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df[
        "NumberOfTime60-89DaysPastDueNotWorse"
    ].apply(lambda x: 96 if x > 95 else x)
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].apply(
        lambda x: 96 if x > 95 else x
    )

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    X_train.to_csv(os.path.join(output_path, "X_train_processed.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test_processed.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    print("Предобработка завершена. Данные сохранены.")


if __name__ == "__main__":
    process_data("../data/raw/cs-training.csv", "../data/processed")
