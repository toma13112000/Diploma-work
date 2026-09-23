# Diploma Work — Классификация диагноза рака молочной железы

Дипломная работа: сравнение алгоритмов машинного обучения для бинарной классификации диагноза (доброкачественная / злокачественная опухоль) на основе Wisconsin Breast Cancer Dataset.

## Файлы репозитория

| Файл | Описание |
|---|---|
| `diploma_project.ipynb` | Jupyter/Colab-ноутбук с полным пайплайном: EDA, предобработка, обучение и сравнение моделей |
| `Diploma Ibrayeva M., Ibrayeva T.pdf` | Текст дипломной работы |
| `Report.pdf` | Отчёт по проекту |

## Данные

Используется набор данных `data.csv` (Wisconsin Breast Cancer Dataset) — 569 наблюдений, 30+ числовых признаков (radius, texture, perimeter, area, smoothness, compactness, concavity и др.) и целевая переменная `diagnosis` (M — malignant / B — benign).

## Пайплайн

1. **Загрузка и разведочный анализ данных (EDA)** — проверка пропусков, распределение классов, `pairplot`, корреляционная матрица признаков.
2. **Предобработка** — удаление пустых столбцов, кодирование целевой переменной (`LabelEncoder`), отбор признаков (`radius_mean`, `perimeter_mean`, `area_mean`, `symmetry_mean`, `compactness_mean`, `concave points_mean`), масштабирование (`StandardScaler`), разбиение на train/test (67/33).
3. **Классические модели ML** — обучение и сравнение:
   - Logistic Regression
   - Random Forest Classifier
   - Decision Tree Classifier
   - SVC
   - K-Nearest Neighbors
   
   Метрики: accuracy, precision, recall, F1-score, confusion matrix, кросс-валидация (`KFold`, `cross_validate`).
4. **Подбор гиперпараметров** — `GridSearchCV` для Decision Tree, KNN, SVC, Random Forest.
5. **Нейронная сеть (Keras)** — полносвязная сеть (Dense + Dropout, 19 скрытых слоёв по 16 нейронов, `relu`, выходной слой `sigmoid`), обучение 100 эпох, итоговая точность на тестовой выборке **~98.2%**.
6. **Сохранение модели** — сериализация лучшей модели через `pickle`.

## Стек технологий

Python, pandas, NumPy, scikit-learn, Keras/TensorFlow, matplotlib, seaborn, plotly.

## Запуск

Ноутбук рассчитан на Google Colab (пути вида `/content/data.csv`). Для локального запуска:

```bash
pip install pandas numpy scikit-learn keras tensorflow matplotlib seaborn plotly
```

Поместите `data.csv` в рабочую директорию и обновите путь к файлу в ноутбуке, затем выполните ячейки по порядку.
