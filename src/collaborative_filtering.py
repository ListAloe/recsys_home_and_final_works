"""Collaborative Filtering - совместная фильтрация."""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances


class ItemBasedCF:
    """
    Item-based Collaborative Filtering.
    
    Рекомендует книги, похожие на те, которые пользователь положительно оценил.
    Сходство между книгами вычисляется на основе векторов предпочтений пользователей
    (столбцы матрицы взаимодействий).
    """
    
    def __init__(self, similarity_metric: str = 'cosine', k_neighbors: int = 10,
                 random_state: int = 42):
        """
        Инициализация модели Item-based CF.
        
        Args:
            similarity_metric: Метрика сходства ('cosine' или 'pearson')
            k_neighbors: Количество соседей для взвешивания предсказаний
            random_state: Seed для воспроизводимости
        """
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.item_similarity = None
        self.interaction_matrix = None
        self.book_ids = None
        self.user_ids = None
        self.book_id_to_idx = None
        self.user_id_to_idx = None
        self.train_data = None
    
    def fit(self, interaction_matrix: np.ndarray, book_ids: List[int],
            user_ids: List[int], train_df: pd.DataFrame) -> 'ItemBasedCF':
        """
        Обучение модели на матрице взаимодействий.
        
        Вычисляет матрицу попарных сходств между книгами на основе
        профилей предпочтений пользователей.
        
        Args:
            interaction_matrix: Матрица взаимодействий (пользователи x книги, dense)
            book_ids: Список ID книг (соответствует столбцам матрицы)
            user_ids: Список ID пользователей (соответствует строкам матрицы)
            train_df: Обучающие данные для справки
            
        Returns:
            self (для цепочки вызовов)
        """
        self.interaction_matrix = interaction_matrix.astype(np.float32)
        self.book_ids = book_ids
        self.user_ids = user_ids
        self.train_data = train_df.copy()
        
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        
        # Используем столбцы матрицы (каждый столбец = профиль предпочтений для одной книги)
        print("Вычисление матрицы сходства между книгами...")
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity: cos(x,y) = (x·y) / (||x|| ||y||)
            self.item_similarity = cosine_similarity(self.interaction_matrix.T)
        elif self.similarity_metric == 'pearson':
            # Pearson correlation: мера линейной зависимости
            self.item_similarity = self._pearson_similarity(self.interaction_matrix.T)
        else:
            raise ValueError(f"Неподдерживаемая метрика: {self.similarity_metric}")
        
        self.item_similarity = self.item_similarity.astype(np.float32)
        
        # Диагональ = 1 (книга идеально похожа на саму себя)
        np.fill_diagonal(self.item_similarity, 1.0)
        
        print(f"Item-based CF успешно обучена")
        print(f"  - Книг: {len(self.book_ids)}")
        print(f"  - Пользователей: {len(self.user_ids)}")
        print(f"  - Матрица сходства: {self.item_similarity.shape}")
        
        return self
    
    def _pearson_similarity(self, X: np.ndarray) -> np.ndarray:
        """Вычисление матрицы Pearson корреляции между столбцами (векторами).
        
        Pearson correlation = E[(x - x_mean)(y - y_mean)] / (std_x * std_y)
        
        Args:
            X: Матрица признаков (n_samples, n_features)
            
        Returns:
            Матрица корреляции (n_features, n_features) со значениями в [-1, 1]
        """
        # Центрирование данных (вычитание среднего)
        X_centered = X - X.mean(axis=0)
        
        # Вычисление стандартных отклонений
        X_std = np.std(X, axis=0)
        # Избегаем деления на ноль для признаков с нулевой дисперсией
        X_std[X_std == 0] = 1
        
        # Нормализация данных
        X_normalized = X_centered / X_std
        
        # Корреляция = (X_normalized.T @ X_normalized) / n_samples
        correlation = (X_normalized.T @ X_normalized) / X.shape[0]
        
        # Ограничиваем значения в диапазон [-1, 1] (числовые погрешности)
        return np.clip(correlation, -1, 1)
    
    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Предсказать рейтинг пользователя для книги.
        
        Args:
            user_id: ID пользователя
            book_id: ID книги
            
        Returns:
            Предсказанный рейтинг
        """
        if book_id not in self.book_id_to_idx:
            return 0.0
        
        if user_id not in self.user_id_to_idx:
            return 0.0
        
        user_idx = self.user_id_to_idx[user_id]
        book_idx = self.book_id_to_idx[book_id]
        
        # Получение рейтингов пользователя
        user_ratings = self.interaction_matrix[user_idx]
        
        # Работа которые пользователь оценил
        rated_books = np.where(user_ratings > 0)[0]
        
        if len(rated_books) == 0:
            return 0.0
        
        # Получение сходства между книгой и оцененными книгами
        similarities = self.item_similarity[book_idx, rated_books]
        
        # Взять топ k соседей
        top_k_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(top_k_indices) == 0:
            return 0.0
        
        top_similarities = similarities[top_k_indices]
        top_ratings = user_ratings[rated_books[top_k_indices]]
        
        # Взвешенное среднее
        if top_similarities.sum() == 0:
            return 0.0
        
        predicted_rating = np.sum(top_ratings * top_similarities) / top_similarities.sum()
        
        return float(np.clip(predicted_rating, 1.0, 5.0))
    
    def get_recommendations(self, user_id: int, top_n: int = 10,
                           exclude_seen: bool = True) -> List[int]:
        """
        Получить рекомендации для пользователя.
        
        Args:
            user_id: ID пользователя
            top_n: количество рекомендаций
            exclude_seen: исключать ли оценённые книги
            
        Returns:
            Список ID книг
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        user_ratings = self.interaction_matrix[user_idx]
        
        # Предсказать рейтинги для всех книг
        predictions = []
        
        for book_idx, book_id in enumerate(self.book_ids):
            if exclude_seen and user_ratings[book_idx] > 0:
                continue
            
            predicted_rating = self.predict_rating(user_id, book_id)
            if predicted_rating > 0:
                predictions.append((book_id, predicted_rating))
        
        # Сортировка по предсказанному рейтингу
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = [bid for bid, _ in predictions[:top_n]]
        
        return recommendations
    
    def get_all_recommendations(self, test_df: pd.DataFrame, top_n: int = 10,
                               batch_size: int = 100, verbose: bool = True) -> Dict[int, List[int]]:
        """
        Получить рекомендации для всех пользователей в тесте (пакетная обработка).
        
        Args:
            test_df: тестовые данные
            top_n: количество рекомендаций на пользователя
            batch_size: размер пакета для вывода прогресса
            verbose: показывать прогресс
            
        Returns:
            Словарь {user_id: [book_ids]}
        """
        recommendations = {}
        user_ids = test_df['user_id'].unique()
        total = len(user_ids)
        
        for batch_idx, user_id in enumerate(user_ids):
            recs = self.get_recommendations(user_id, top_n=top_n, exclude_seen=True)
            recommendations[user_id] = recs
            
            if verbose and (batch_idx + 1) % batch_size == 0:
                print(f"  Прогресс: {batch_idx + 1}/{total} пользователей")
        
        if verbose:
            print(f"  Рекомендации созданы для {total} пользователей")
        
        return recommendations
