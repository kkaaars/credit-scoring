import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load("models/best_model.pkl")
    logger.info("Модель успешно загружена.")
except FileNotFoundError:
    logger.error("Ошибка: файл модели 'best_model.pkl' не найден.")
    model = None


app = FastAPI()


class DataInput(BaseModel):
    age: int
    monthly_income: float
    number_of_open_credit_lines_and_loans: int
    number_of_times_30_59_days_past_due_not_worse: int
    number_of_times_90_days_late: int
    number_real_estate_loans_or_lines: int
    number_of_times_60_89_days_past_due_not_worse: int
    number_of_dependents: float
    number_of_times_late: int
    utilization_of_unsecured_lines: float


@app.post("/predict")
def predict(data: DataInput):
    if model is None:
        return {
            "error": "Модель не загружена. Пожалуйста, убедитесь, что 'best_model.pkl' существует."
        }

    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict_proba(input_df)[:, 1]

    return {"prediction": prediction[0].tolist()}


@app.get("/")
def read_root():
    return {"message": "API для предсказания кредитного скоринга успешно запущено!"}
