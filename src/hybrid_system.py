"""Гибридная рекомендательная система."""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


class HybridRecommender:
    """
    Гибридная рекомендательная система.
    
    Объединяет несколько моделей рекомендаций с взвешиванием для получения
    более робастных и точных рекомендаций.
    
    Поддерживаемые модели:
    - Popularity (базовая модель)
    - Content-based (на основе текстов)
    - Collaborative Filtering (item-based CF)
    - Matrix Factorization (SVD)
    - Neural model (двухбашенная нейросеть)
    """
    
    def __init__(self):
        """Инициализация пустой гибридной системы."""
        self.models = {}          # Словарь моделей
        self.weights = {}         # Веса для каждой модели
        self.fallback_model = None  # Модель для cold-start пользователей
    
    def add_model(self, name: str, model, weight: float = 1.0) -> 'HybridRecommender':
        """
        Добавление модели в гибридную систему.
        
        Args:
            name: Уникальное название модели для идентификации
            model: Объект модели с методом get_recommendations(user_id, top_n, **kwargs)
            weight: Вес модели при объединении рекомендаций (нормализуется автоматически)
            
        Returns:
            self (для цепочки вызовов add_model().add_model()...)
        """
        self.models[name] = model
        self.weights[name] = weight
        
        return self
    
    def set_fallback_model(self, model) -> 'HybridRecommender':
        """
        Установка fallback модели для обработки cold-start пользователей.
        
        Используется когда основные модели не могут выдать рекомендации
        (например, новый пользователь, не встречавшийся во время обучения).
        
        Args:
            model: Объект модели для использования в качестве fallback (обычно Popularity)
            
        Returns:
            self (для цепочки вызовов)
        """
        self.fallback_model = model
        
        return self
    
    def normalize_weights(self) -> None:
        """Нормализация весов моделей к сумме, равной 1.
        
        Преобразует абсолютные веса в относительные доли.
        """
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {
                name: weight / total_weight
                for name, weight in self.weights.items()
            }
    
    def get_model_recommendations(self, user_id: int, top_n: int = 10,
                                 **kwargs) -> Dict[str, List[int]]:
        """
        Получение рекомендаций от каждой модели отдельно.
        
        Полезно для отладки и анализа вклада каждой модели.
        
        Args:
            user_id: ID пользователя
            top_n: Количество рекомендаций от каждой модели
            **kwargs: Дополнительные параметры для метода get_recommendations
            
        Returns:
            Словарь {model_name: [book_ids]}
        """
        recommendations = {}
        
        for name, model in self.models.items():
            try:
                recs = model.get_recommendations(user_id, top_n=top_n, **kwargs)
                recommendations[name] = recs
            except Exception as e:
                print(f"Предупреждение: Модель '{name}' ошибка для пользователя {user_id}: {e}")
                recommendations[name] = []
        
        return recommendations
    
    def get_recommendations(self, user_id: int, top_n: int = 10,
                           exclude_seen: bool = True,
                           train_df: pd.DataFrame = None) -> List[int]:
        """
        Получение гибридных рекомендаций для пользователя.
        
        Алгоритм объединения:
        1. Получить рекомендации от каждой модели (в 2x количестве)
        2. Каждой модели присвоить вес согласно self.weights
        3. Для каждой книги вычислить взвешенный score (с дисконтом по позиции)
        4. Отранжировать по суммарному score
        5. Вернуть top_n книг
        
        Args:
            user_id: ID пользователя
            top_n: Количество рекомендаций
            exclude_seen: Исключать ли уже оц оцениваемые книги
            train_df: Обучающие данные для исключения
            
        Returns:
            Список ID книг, отранжированных по релевантности
        """
        # Получение рекомендаций от каждой модели
        model_recs = self.get_model_recommendations(
            user_id, top_n=top_n*2,
            exclude_seen=exclude_seen
        )
        
        # Fallback, если нет рекомендаций
        if not model_recs or all(len(recs) == 0 for recs in model_recs.values()):
            if self.fallback_model:
                return self.fallback_model.get_recommendations(
                    user_id, top_n=top_n, exclude_seen=exclude_seen
                )
            return []
        
        # Score = sum(weight * (1 / log2(rank + 1))) для каждой модели
        candidate_scores = {}
        
        self.normalize_weights()
        
        for model_name, recs in model_recs.items():
            weight = self.weights.get(model_name, 1.0)
            
            for rank, book_id in enumerate(recs, start=1):
                # Дисконтирование по позиции (логарифмическое)
                score = weight / np.log2(rank + 1)
                
                if book_id not in candidate_scores:
                    candidate_scores[book_id] = 0.0
                
                candidate_scores[book_id] += score
        
        # Сортировка по суммарному score (убывание)
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = [bid for bid, _ in sorted_candidates[:top_n]]
        
        return recommendations
    
    def get_all_recommendations(self, test_df: pd.DataFrame, top_n: int = 10,
                               train_df: pd.DataFrame = None) -> Dict[int, List[int]]:
        """
        Получить гибридные рекомендации для всех пользователей в тесте.
        
        Args:
            test_df: тестовые данные
            top_n: количество рекомендаций на пользователя
            train_df: обучающие данные
            
        Returns:
            Словарь {user_id: [book_ids]}
        """
        recommendations = {}
        
        for user_id in test_df['user_id'].unique():
            recs = self.get_recommendations(
                user_id, top_n=top_n,
                exclude_seen=True,
                train_df=train_df
            )
            recommendations[user_id] = recs
        
        return recommendations
    
    def optimize_weights(self, test_df: pd.DataFrame, train_df: pd.DataFrame,
                        rating_threshold: int = 4, k: int = 10,
                        learning_rate: float = 0.05, iterations: int = 50,
                        verbose: bool = False, use_scipy: bool = True) -> Dict[str, float]:
        """
        Оптимизировать веса моделей на основе Recall@K.
        
        Использует scipy.optimize.minimize для эффективной оптимизации вместо перебора.
        
        Args:
            test_df: тестовые данные
            train_df: обучающие данные
            rating_threshold: порог релевантности
            k: K для Recall@K
            learning_rate: не используется (для совместимости)
            iterations: количество итераций (макс для BFGS)
            verbose: выводить прогресс
            use_scipy: использовать scipy.optimize (по умолчанию True)
            
        Returns:
            Оптимизированные веса
        """
        from src.evaluation import ModelEvaluator, EvaluationMetrics

        evaluator = ModelEvaluator(test_df, train_df, rating_threshold)
        model_names = list(self.models.keys())
        unique_users = test_df['user_id'].unique()
        
        cached_model_recs = {}
        for model_name, model in self.models.items():
            model_recs = {}
            for user_id in unique_users:
                try:
                    recs = model.get_recommendations(user_id, top_n=k*2, exclude_seen=True)
                    model_recs[user_id] = recs
                except:
                    model_recs[user_id] = []
            cached_model_recs[model_name] = model_recs
        
        # Предварительно вычислить релевантные запросы для каждого пользователя
        relevant_items = {}
        for user_id in unique_users:
            rel = evaluator.get_relevant_items(user_id)
            if len(rel) > 0:
                relevant_items[user_id] = rel
        
        if verbose:
            print(f"Оптимизация весов на {len(relevant_items)} релевантных пользователях...")
        
        def compute_score(weights_array):
            """Вычисление Recall@K для текущих весов (для минимизации используем отрицание)."""
            # Присвоение весов
            for i, name in enumerate(model_names):
                self.weights[name] = max(0.01, weights_array[i])
            self.normalize_weights()
            
            # Вычисление гибридных рекомендаций с кэшированными результатами
            recalls = []
            for user_id in relevant_items.keys():
                candidate_scores = {}
                
                # Используем кэшированные рекомендации
                for model_name in model_names:
                    recs = cached_model_recs[model_name].get(user_id, [])
                    weight = self.weights[model_name]
                    
                    for rank, book_id in enumerate(recs, start=1):
                        score = weight / np.log2(rank + 1)
                        if book_id not in candidate_scores:
                            candidate_scores[book_id] = 0.0
                        candidate_scores[book_id] += score
                
                # Получение топ-рекомендаций
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                rec_list = [bid for bid, _ in sorted_candidates[:k]]
                
                # Вычисление recall
                recall = EvaluationMetrics.recall_at_k(rec_list, relevant_items[user_id], k)
                recalls.append(recall)
            
            return -np.mean(recalls)  # Минус потому что scipy минимизирует
        
        # Оптимизация
        if use_scipy:
            try:
                from scipy.optimize import minimize
                
                initial_weights = np.array([self.weights[name] for name in model_names])
                
                result = minimize(
                    compute_score,
                    initial_weights,
                    method='BFGS',
                    options={'maxiter': iterations, 'disp': verbose}
                )
                
                # Применение оптимизированных весов
                for i, name in enumerate(model_names):
                    self.weights[name] = max(0.01, result.x[i])
                self.normalize_weights()
                
                if verbose:
                    final_score = -result.fun
                    print(f"Оптимизация завершена. Финальный Recall@{k}: {final_score:.4f}")
            except ImportError:
                print("WARNING: scipy не установлен, переходим на coordinate descent")
                use_scipy = False
        
        # Fallback: coordinate descent если scipy недоступен
        if not use_scipy:
            best_score = float('-inf')
            best_weights = self.weights.copy()
            patience = 5
            no_improve = 0
            
            for iteration in range(iterations):
                improved = False
                
                # Пробовать улучшить каждый вес
                for idx, name in enumerate(model_names):
                    original_weight = self.weights[name]
                    current_score = -compute_score(np.array([self.weights[m] for m in model_names]))
                    
                    # Попытка увеличить вес
                    self.weights[name] = original_weight * 1.1
                    self.normalize_weights()
                    new_score = -compute_score(np.array([self.weights[m] for m in model_names]))
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Попытка уменьшить вес
                        self.weights[name] = original_weight * 0.9
                        self.normalize_weights()
                        new_score = -compute_score(np.array([self.weights[m] for m in model_names]))
                        
                        if new_score > current_score:
                            improved = True
                        else:
                            # Возврат оригинального веса
                            self.weights[name] = original_weight
                            self.normalize_weights()
                
                # Проверка на улучшение
                current_score = -compute_score(np.array([self.weights[m] for m in model_names]))
                if current_score > best_score:
                    best_score = current_score
                    best_weights = self.weights.copy()
                    no_improve = 0
                    if verbose:
                        print(f"  Итерация {iteration + 1}: Recall@{k} = {current_score:.4f}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"  Ранняя остановка на итерации {iteration + 1}")
                        break
            
            self.weights = best_weights
            self.normalize_weights()
            
            if verbose:
                print(f"Оптимизация завершена. Лучший Recall@{k}: {best_score:.4f}")
        
        return self.weights
