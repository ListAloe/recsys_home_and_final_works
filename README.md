# Рекомендательная система книг

Персонализированное ранжирование книг на основе оценок пользователей.

## Структура проекта

```
data/          ratings.csv, books.csv, tags.csv, book_tags.csv
src/           data_loader, preprocessing, eda, popularity_model, content_model
               collaborative_filtering, matrix_factorization, neural_model
               evaluation, hybrid_system, pipeline, utils
notebooks/     rec_sys_final.ipynb
```

## Модели

- **Popularity**: рекомендация популярных книг
- **Content-Based**: TF-IDF на основе описаний и тегов
- **Item-Based CF**: сходство профилей пользователей
- **SVD/Matrix Factorization**: TruncatedSVD факторизация
- **Two-Tower Neural Network**: эмбединги для пользователей и книг
- **Hybrid System**: взвешенное объединение всех моделей

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/rec_sys_final.ipynb
```
## Python API

```python
from src.pipeline import RecommendationPipeline

pipeline = RecommendationPipeline('data')
results = pipeline.run_full_pipeline(neural_epochs=5, k_values=[5, 10])
# results['models_comparison'] - сравнение моделей
# results['hybrid_results'] - результаты гибридной системы
```

## Метрики

- **Precision@K**: доля релевантных в top-K
- **Recall@K**: доля найденных релевантных
- **nDCG@K**: качество ранжирования

Temporal split: 80% train / 20% test. Threshold: rating >= 4.

## Результаты

| Model | Precision@5 | Recall@5 | nDCG@5 | MAP@5 | Precision@10 | Recall@10 | nDCG@10 | MAP@10 |
|---|---|---|---|---|---|---|---|---|
| Popularity | 0.0014 | 0.0004 | 0.0015 | 0.0031 | 0.0011 | 0.0007 | 0.0012 | 0.0036 |
| Item-Based CF | 0.0062 | 0.0018 | 0.0069 | 0.0156 | 0.0052 | 0.0028 | 0.0060 | 0.0167 |
| SVD | 0.0482 | 0.0157 | 0.0509 | 0.1012 | 0.0404 | 0.0270 | 0.0454 | 0.1077 |
| Neural | 0.0044 | 0.0014 | 0.0047 | 0.0113 | 0.0039 | 0.0025 | 0.0043 | 0.0129 |
| Hybrid | 0.0256 | 0.0082 | 0.0290 | 0.0667 | 0.0221 | 0.0143 | 0.0257 | 0.0715 |

## Технические решения

- Sparse matrices (CSR) для эффективности
- TruncatedSVD (50 компонент) для SVD
- Batch processing для нейросети
- Vectorized operations (NumPy)