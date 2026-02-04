"""Evaluation metrics для рекомендательных систем."""

from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
import numpy as np
from functools import lru_cache
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class EvaluationMetrics:
    """Класс с статическими методами для вычисления метрик оценки рекомендаций.
    
    Включает стандартные метрики, используемые в информационном поиске
    и рекомендательных системах.
    """
    
    @staticmethod
    def precision_at_k(recommendations: List[int], relevant_items: Set[int],
                       k: int = 10) -> float:
        """
        Precision@K - доля релевантных предметов в top-K рекомендациях.
        
        Precision@K = (кол-во релевантных в top-K) / K
        
        Интерпретация: "Каков процент выданных рекомендаций оказались правильными?"
        
        Args:
            recommendations: Список рекомендаций, отсортированный по релевантности
            relevant_items: Множество релевантных предметов для пользователя
            k: Размер топ-K списка
            
        Returns:
            Precision@K (от 0 до 1, где 1 = все рекомендации релевантны)
        """
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        # Считаем пересечение рекомендаций с релевантными
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(recommendations: List[int], relevant_items: Set[int],
                    k: int = 10) -> float:
        """
        Recall@K - доля найденных релевантных предметов от всех релевантных.
        
        Recall@K = (кол-во релевантных в top-K) / (всего релевантных)
        
        Интерпретация: "Какой процент всех релевантных предметов мы нашли?"
        
        Args:
            recommendations: Список рекомендаций, отсортированный по релевантности
            relevant_items: Множество релевантных предметов для пользователя
            k: Размер топ-K списка
            
        Returns:
            Recall@K (от 0 до 1, где 1 = найдены все релевантные)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommendations: List[int], relevant_items: Set[int],
                  k: int = 10) -> float:
        """
        nDCG@K - Normalized Discounted Cumulative Gain.
        
        Учитывает не только релевантность, но и позицию рекомендации
        (релевантные в начале списка вносят больший вклад).
        
        DCG@K = sum(rel_i / log2(i+1)) для i=1..K
        nDCG@K = DCG@K / IDCG@K (нормализованный DCG)
        
        Интерпретация: "Насколько хорошо ранжированы релевантные редметы?"
        
        Args:
            recommendations: Список рекомендаций, отсортированный по релевантности
            relevant_items: Множество релевантных предметов для пользователя
            k: Размер топ-K списка
            
        Returns:
            nDCG@K (от 0 до 1, где 1 = идеальное ранжирование)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        dcg = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in relevant_items:
                # Дисконт по позиции: log2(i+2) чтобы первый элемент был log2(2)=1
                dcg += 1.0 / np.log2(i + 2)
        
        # Идеально все релевантные элементы в начале списка
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        # Нормализация
        ndcg = dcg / idcg
        
        return float(ndcg)
    
    @staticmethod
    def map_at_k(recommendations: List[int], relevant_items: Set[int],
                 k: int = 10) -> float:
        """
        MAP@K - Mean Average Precision.
        
        Args:
            recommendations: список рекомендаций
            relevant_items: множество релевантных элементов
            k: количество топ элементов
            
        Returns:
            MAP@K (значение от 0 до 1)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        
        precisions = []
        relevant_count = 0
        
        for i, item in enumerate(top_k):
            if item in relevant_items:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        map_value = np.mean(precisions)
        
        return float(map_value)


class ModelEvaluator:
    """Класс для оценки и сравнения рекомендательных моделей.
    
    Поддерживает вычисление метрик Precision@K, Recall@K, nDCG@K, MAP@K
    с параллелизацией и кэшированием релевантных элементов.
    """
    
    def __init__(self, test_df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None,
                 rating_threshold: int = 4) -> None:
        """
        Инициализация.
        
        Args:
            test_df: тестовые данные
            train_df: обучающие данные (для исключения)
            rating_threshold: порог релевантности
        """
        self.test_df = test_df
        self.train_df = train_df
        self.rating_threshold = rating_threshold
        
        # Кэш релевантных элементов (user_id -> Set[book_id])
        self._relevant_cache = {}
    
    def get_relevant_items(self, user_id: int, rating_col: str = 'rating') -> Set[int]:
        """
        Получить релевантные книги для пользователя в тесте (с кэшированием).
        
        Args:
            user_id: ID пользователя
            rating_col: название колонки с рейтингом
            
        Returns:
            Множество релевантных book_ids
        """
        if user_id in self._relevant_cache:
            return self._relevant_cache[user_id]
        
        user_test = self.test_df[self.test_df['user_id'] == user_id]
        relevant = user_test[user_test[rating_col] >= self.rating_threshold]['book_id']
        
        result = set(relevant.values)
        self._relevant_cache[user_id] = result
        
        return result
    
    def _evaluate_user(self, user_id: int, model_recommendations: Dict[int, List[int]],
                       k_values: List[int]) -> Dict[int, Tuple[List[float], List[float], List[float], List[float]]]:
        """Вычислить метрики для одного пользователя (вспомогательная функция для параллелизма)."""
        relevant = self.get_relevant_items(user_id)
        
        if len(relevant) == 0:
            return None
        
        recommendations = model_recommendations.get(user_id, [])
        
        metrics = {}
        for k in k_values:
            metrics[k] = (
                EvaluationMetrics.precision_at_k(recommendations, relevant, k),
                EvaluationMetrics.recall_at_k(recommendations, relevant, k),
                EvaluationMetrics.ndcg_at_k(recommendations, relevant, k),
                EvaluationMetrics.map_at_k(recommendations, relevant, k)
            )
        
        return metrics
    
    def evaluate_model(self, model_recommendations: Dict[int, List[int]],
                      k_values: List[int] = None, n_jobs: int = -1) -> Dict[str, float]:
        """
        Оценить модель по всем пользователям (параллелизм + кэширование).
        
        Args:
            model_recommendations: словарь {user_id: [book_ids]}
            k_values: значения K для метрик (например [5, 10, 20])
            n_jobs: количество процессов (-1 = все, 1 = без параллелизма)
            
        Returns:
            Словарь метрик
        """
        if k_values is None:
            k_values = [5, 10]
        
        user_ids = self.test_df['user_id'].unique()
        
        # Параллельная оценка
        if HAS_JOBLIB and n_jobs != 1:
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_user)(user_id, model_recommendations, k_values)
                for user_id in user_ids
            )
            results_list = [r for r in results_list if r is not None]
        else:
            # Последовательная обработка
            results_list = []
            for user_id in user_ids:
                result = self._evaluate_user(user_id, model_recommendations, k_values)
                if result is not None:
                    results_list.append(result)
        
        if len(results_list) == 0:
            return {}
        
        # Агрегация метрик
        results = {}
        for k in k_values:
            precisions = [r[k][0] for r in results_list]
            recalls = [r[k][1] for r in results_list]
            ndcgs = [r[k][2] for r in results_list]
            maps = [r[k][3] for r in results_list]
            
            results[f'Precision@{k}'] = np.mean(precisions)
            results[f'Recall@{k}'] = np.mean(recalls)
            results[f'nDCG@{k}'] = np.mean(ndcgs)
            results[f'MAP@{k}'] = np.mean(maps)
        
        return results
    
    def compare_models(self, models_dict: Dict[str, Dict[int, List[int]]],
                      k_values: List[int] = None, n_jobs: int = -1) -> pd.DataFrame:
        """
        Сравнить несколько моделей (параллелизм + кэширование).
        
        Args:
            models_dict: словарь {model_name: {user_id: [book_ids]}}
            k_values: значения K для метрик
            n_jobs: количество процессов для параллелизма
            
        Returns:
            DataFrame с результатами
        """
        if k_values is None:
            k_values = [5, 10]
        
        results_list = []
        
        for model_name, recommendations in models_dict.items():
            print(f"  Оценка {model_name}...")
            metrics = self.evaluate_model(recommendations, k_values, n_jobs=n_jobs)
            metrics['Model'] = model_name
            results_list.append(metrics)
        
        df = pd.DataFrame(results_list)
        df = df[['Model'] + [col for col in df.columns if col != 'Model']]
        
        return df.round(4)
