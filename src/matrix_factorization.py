"""Matrix Factorization с использованием SVD и TruncatedSVD."""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, coo_matrix


class MatrixFactorization:
    """
    Matrix Factorization на основе TruncatedSVD.
    
    Факторизирует матрицу взаимодействий user x book на две матрицы:
    - U: распределение пользователей по скрытым факторам (n_users x n_factors)
    - V: распределение книг по скрытым факторам (n_books x n_factors)
    
    Рейтинг предсказывается как скалярное произведение: R_ui ~ U_u · V_i
    """
    
    def __init__(self, n_factors: int = 50, n_iter: int = 100, random_state: int = 42):
        """
        Инициализация модели Matrix Factorization.
        
        Args:
            n_factors: Количество латентных факторов
            n_iter: Количество итераций для SVD
            random_state: Seed для воспроизводимости
        """
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state
        self.U = None
        self.V = None
        self.svd = None
        self.interaction_matrix = None
        self.book_ids = None
        self.user_ids = None
        self.book_id_to_idx = None
        self.user_id_to_idx = None
        self.mean_rating = None
    
    def fit(self, interaction_matrix: np.ndarray, book_ids: List[int],
            user_ids: List[int], train_df: pd.DataFrame = None) -> 'MatrixFactorization':
        """
        Обучение модели факторизации матрицы.
        
        Применяет Truncated SVD для разложения матрицы взаимодействий,
        оставляя только n_factors основных компонент.
        
        Args:
            interaction_matrix: Матрица взаимодействий user x book (dense, n_users x n_books)
            book_ids: Список ID книг (соответствует столбцам матрицы)
            user_ids: Список ID пользователей (соответствует строкам матрицы)
            train_df: Обучающие данные (опционально, для статистики)
            
        Returns:
            self (для цепочки вызовов)
        """
        self.interaction_matrix = interaction_matrix.astype(np.float32)
        self.book_ids = book_ids
        self.user_ids = user_ids
        
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        
        # Вычисляем среднее для ненулевых значений (рейтинги)
        self.mean_rating = self.interaction_matrix[self.interaction_matrix > 0].mean()
        
        # Центрирование: вычитаем среднее
        centered_matrix = self.interaction_matrix.copy()
        centered_matrix[centered_matrix > 0] -= self.mean_rating
        
        # SVD эффективнее работает с разреженными матрицами
        sparse_matrix = csr_matrix(centered_matrix)
        
        print(f"Применение TruncatedSVD с {self.n_factors} факторами...")
        
        self.svd = TruncatedSVD(
            n_components=self.n_factors,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Fit & transform: U = sparse_matrix @ svd.components_^T
        self.U = self.svd.fit_transform(sparse_matrix)
        # V = svd.components_^T
        self.V = self.svd.components_.T
        
        self.U = self.U.astype(np.float32)
        self.V = self.V.astype(np.float32)
        
        # Статистика
        explained_variance = self.svd.explained_variance_ratio_.sum()
        
        print(f"Факторизация матрицы успешно обучена")
        print(f"  - Факторы: {self.n_factors}")
        print(f"  - Пользователей: {self.U.shape[0]}")
        print(f"  - Книг: {self.V.shape[0]}")
        print(f"  - Объясненная дисперсия: {explained_variance:.3f}")
        
        return self
    
    def predict_rating(self, user_id: int, book_id: int) -> float:
        """
        Предсказать рейтинг пользователя для книги.
        
        Args:
            user_id: ID пользователя
            book_id: ID книги
            
        Returns:
            Предсказанный рейтинг (1-5)
        """
        if user_id not in self.user_id_to_idx or book_id not in self.book_id_to_idx:
            return self.mean_rating if self.mean_rating else 3.0
        
        user_idx = self.user_id_to_idx[user_id]
        book_idx = self.book_id_to_idx[book_id]
        
        # Скалярное произведение между эмбеддингом пользователя и эмбеддингом товара
        prediction = np.dot(self.U[user_idx], self.V[book_idx])
        
        # Добавление среднего значения
        prediction += self.mean_rating
        
        # Клипировать в диапазон [1, 5]
        return float(np.clip(prediction, 1.0, 5.0))
    
    def get_recommendations(self, user_id: int, top_n: int = 10,
                           exclude_seen: bool = True,
                           interaction_matrix: np.ndarray = None) -> List[int]:
        """
        Получить рекомендации для пользователя.
        
        Args:
            user_id: ID пользователя
            top_n: количество рекомендаций
            exclude_seen: исключать ли оценённые книги
            interaction_matrix: матрица взаимодействий для отслеживания seen items
            
        Returns:
            Список ID книг
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        if interaction_matrix is None:
            interaction_matrix = self.interaction_matrix
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Предсказать рейтинги для всех книг
        predictions = np.dot(self.U[user_idx], self.V.T) + self.mean_rating
        predictions = np.clip(predictions, 1.0, 5.0)
        
        # Исключение оцененных книг
        if exclude_seen:
            seen_books = interaction_matrix[user_idx] > 0
            predictions[seen_books] = -np.inf
        
        # Получение топ-книг
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        recommendations = [self.book_ids[idx] for idx in top_indices if predictions[idx] > -np.inf]
        
        return recommendations[:top_n]
    
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
    
    def compute_rmse(self, test_df: pd.DataFrame, rating_col: str = 'rating', batch_size: int = 1000) -> float:
        """
        Вычислить RMSE на тестовом наборе (векторизованный расчёт).
        
        Args:
            test_df: тестовые данные
            rating_col: название колонки с рейтингом
            batch_size: размер пакета для обработки
            
        Returns:
            RMSE значение
        """
        predictions = []
        actuals = []
        
        # Пакетная обработка для лучшей производительности
        for start_idx in range(0, len(test_df), batch_size):
            end_idx = min(start_idx + batch_size, len(test_df))
            batch = test_df.iloc[start_idx:end_idx]
            
            batch_predictions = []
            for _, row in batch.iterrows():
                user_id = row['user_id']
                book_id = row['book_id']
                
                predicted_rating = self.predict_rating(user_id, book_id)
                batch_predictions.append(predicted_rating)
            
            predictions.extend(batch_predictions)
            actuals.extend(batch[rating_col].values)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return float(rmse)
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Получить embedding пользователя."""
        if user_id not in self.user_id_to_idx:
            return None
        
        user_idx = self.user_id_to_idx[user_id]
        return self.U[user_idx].copy()
    
    def get_item_embedding(self, book_id: int) -> Optional[np.ndarray]:
        """Получить embedding книги."""
        if book_id not in self.book_id_to_idx:
            return None
        
        book_idx = self.book_id_to_idx[book_id]
        return self.V[book_idx].copy()
