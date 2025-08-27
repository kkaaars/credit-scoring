# Проект "Кредитный скоринг"

## Описание проекта
Этот проект представляет собой полный пайплайн машинного обучения для **прогнозирования кредитоспособности заёмщиков**. Он демонстрирует, как создать, обучить и развернуть модель машинного обучения в Docker-контейнерах, используя Docker Compose.

Основные задачи проекта:
- **Предобработка данных**: Очистка и подготовка данных для обучения моделей.
- **Обучение моделей**: Сравнение нескольких алгоритмов машинного обучения.
- **Развертывание**: Упаковка пайплайна обучения и API-сервиса в Docker-контейнеры.
- **Автоматизация**: Использование Docker Compose для оркестрации сервисов.

## Структура проекта
Проект имеет следующую структуру:
```

.
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.api
├── README.md
├── requirements.txt
├── train.py
├── app/
│   └── api.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
└── src/
    ├── __init__.py
    ├── data_processing.py
    └── modeling.py

````

## Технологии
- **Python**: Основной язык программирования.
- **Pandas, Scikit-learn, LightGBM**: Библиотеки для обработки данных и машинного обучения.
- **FastAPI, Uvicorn**: Для создания и запуска RESTful API.
- **Docker, Docker Compose**: Для контейнеризации и оркестрации сервисов.
- **Matplotlib, Seaborn**: Для визуализации данных и результатов.

## Результаты
### Обучение моделей
В рамках проекта были обучены и протестированы три модели. Лучшую производительность показала модель **LightGBM**.

| Модель                   | ROC-AUC Score |
|--------------------------|---------------|
| Logistic Regression      | 0.7760        |
| Random Forest Classifier | 0.8126        |
| LightGBM                 | **0.8430** |


## Как запустить проект
1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/kkaaars/credit-scoring
    cd credit-scoring
    ```

2.  **Запустите Docker Compose:**
    Эта команда соберёт Docker-образы, запустит контейнер `train` для обучения моделей и, после его завершения, запустит контейнер `api` с готовым API-сервисом.
    ```bash
    docker-compose up --build
    ```

3.  **Проверьте работу API:**
    - Откройте [http://localhost:8000/](http://localhost:8000/) в браузере, чтобы увидеть приветственное сообщение.
    - Откройте [http://localhost:8000/docs](http://localhost:8000/docs), чтобы использовать интерактивную документацию FastAPI (Swagger UI) для отправки тестовых запросов.

### Пример запроса к API
Отправьте `POST`-запрос на эндпоинт `/predict` с данными в формате JSON.

```json
{
  "age": 45,
  "monthly_income": 9120.0,
  "number_of_open_credit_lines_and_loans": 13,
  "number_of_times_30_59_days_past_due_not_worse": 2,
  "number_of_times_90_days_late": 0,
  "number_real_estate_loans_or_lines": 6,
  "number_of_times_60_89_days_past_due_not_worse": 0,
  "number_of_dependents": 2.0,
  "number_of_times_late": 0,
  "utilization_of_unsecured_lines": 0.5
}
````

**Ответ:**

```json
{"prediction": 0.86758}
```

-----

## Примечания

  - Все данные и обученные модели исключены из Git с помощью `.gitignore`.
  - Для запуска проекта необходимо иметь установленный и запущенный **Docker Desktop**.

<!-- end list -->

