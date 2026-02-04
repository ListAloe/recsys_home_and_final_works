"""Content-based рекомендательная модель."""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class ContentBasedModel:
    """
    Content-based модель на основе TF-IDF и cosine similarity.
    
    Рекомендует книги, текстово похожие на те, которые пользователь
    положительно оценил. Использует TF-IDF векторизацию текстов книг.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2),
                 random_state: int = 42):
        """
        Инициализация content-based модели.
        
        Args:
            max_features: Максимальное количество TF-IDF признаков
            ngram_range: Диапазон n-грамм (юниграммы и биграммы)
            random_state: Seed для воспроизводимости
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.vectorizer = None
        self.tfidf_matrix = None
        self.book_profiles = None
        self.book_ids = None
        self.book_id_to_idx = None
        self.train_data = None
    
    def fit(self, book_profiles_df: pd.DataFrame, train_df: pd.DataFrame,
            profile_col: str = 'text_profile', rating_col: str = 'rating',
            rating_threshold: float = 4.0) -> 'ContentBasedModel':
        """
        Обучение модели на текстовых профилях книг.
        
        Создает TF-IDF матрицу из текстовых профилей книг для последующих
        расчетов сходства между книгами.
        
        Args:
            book_profiles_df: DataFrame с текстовыми профилями книг
            train_df: Обучающие данные с рейтингами
            profile_col: Названия колонки с текстовым профилем книги
            rating_col: Названия колонки с рейтингом
            rating_threshold: Порог для считания книги релевантной
            
        Returns:
            self (для цепочки вызовов)
        """
        self.train_data = train_df.copy()
        self.book_profiles = book_profiles_df.copy()
        
        self.book_ids = book_profiles_df['book_id'].values
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        # Извлечение текстов для векторизации
        texts = self.book_profiles[profile_col].fillna('')
        
        # Инициализация и обучение TF-IDF векторизатора
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=1,
            max_df=0.95,  # Исключаем слишком частые слова
            stop_words='english'
        )
        
        # Преобразование текстов в разреженную TF-IDF матрицу
        self.tfidf_matrix = self.vectorizer.fit_transform(texts).astype(np.float32)
        
        # Информация об обученной модели
        sparsity = 1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])
        
        print(f"Content-based модель обучена")
        print(f"  - Книг: {len(self.book_ids)}")
        print(f"  - TF-IDF признаков: {self.tfidf_matrix.shape[1]}")
        print(f"  - Разреженность матрицы: {sparsity:.3f}")
        
        return self
    
    def get_similar_books(self, book_id: int, top_n: int = 10,
                         min_similarity: float = 0.0) -> List[Tuple[int, float]]:
        """
        Получить похожие на заданную книгу.
        
        Args:
            book_id: ID книги
            top_n: количество похожих книг
            min_similarity: минимальное сходство
            
        Returns:
            Список кортежей (book_id, similarity_score)
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted yet.")
        
        if book_id not in self.book_id_to_idx:
            return []
        
        idx = self.book_id_to_idx[book_id]
        
        # Вычисление cosine similarity
        similarities = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()
        
        # Исключение самой книги (индекс idx имеет similarity = 1)
        similarities[idx] = -1
        
        # Поиск топ похожих
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= min_similarity:
                results.append((self.book_ids[idx], float(sim)))
        
        return results
    
    def get_recommendations(self, user_id: int, train_df: pd.DataFrame = None,
                           top_n: int = 10, min_avg_similarity: float = 0.1,
                           rating_threshold: float = 4.0) -> List[int]:
        """
        Получить рекомендации для пользователя на основе его истории.
        
        Args:
            user_id: ID пользователя
            train_df: данные для определения истории (если None, используется self.train_data)
            top_n: количество рекомендаций
            min_avg_similarity: минимальное среднее сходство
            rating_threshold: порог релевантности
            
        Returns:
            Список ID книг
        """
        if train_df is None:
            train_df = self.train_data
        
        if train_df is None:
            raise ValueError("No training data available.")
        
        # Получение положительно оцененных книг пользователя
        user_positive = train_df[
            (train_df['user_id'] == user_id) &
            (train_df['rating'] >= rating_threshold)
        ]['book_id'].unique()
        
        if len(user_positive) == 0:
            return []
        
        # Поиск похожих книг для каждой положительной оценки
        candidate_scores = {}
        
        for book_id in user_positive:
            similar = self.get_similar_books(book_id, top_n=20, min_similarity=0.0)
            
            for similar_book_id, similarity in similar:
                if similar_book_id not in candidate_scores:
                    candidate_scores[similar_book_id] = []
                candidate_scores[similar_book_id].append(similarity)
        
        # Вычисление среднего сходства и сортировка
        candidate_avg_scores = {
            book_id: np.mean(scores)
            for book_id, scores in candidate_scores.items()
            if np.mean(scores) >= min_avg_similarity
        }
        
        # Исключение книг, которые пользователь уже видел
        user_seen = set(train_df[train_df['user_id'] == user_id]['book_id'])
        candidate_avg_scores = {
            bid: score for bid, score in candidate_avg_scores.items()
            if bid not in user_seen
        }
        
        # Сортировка по среднему сходству и брание топ
        sorted_candidates = sorted(
            candidate_avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = [bid for bid, _ in sorted_candidates[:top_n]]
        
        return recommendations
    
    def get_all_recommendations(self, test_df: pd.DataFrame, top_n: int = 10,
                               rating_threshold: float = 4.0, batch_size: int = 100,
                               verbose: bool = True) -> Dict[int, List[int]]:
        """
        Получить рекомендации для всех пользователей в тесте (пакетная обработка).
        
        Args:
            test_df: тестовые данные (для определения пользователей)
            top_n: количество рекомендаций на пользователя
            rating_threshold: порог релевантности
            batch_size: размер пакета для прогресса-бара
            verbose: показывать прогресс
            
        Returns:
            Словарь {user_id: [book_ids]}
        """
        recommendations = {}
        user_ids = test_df['user_id'].unique()
        total = len(user_ids)
        
        for batch_idx, user_id in enumerate(user_ids):
            recs = self.get_recommendations(
                user_id,
                train_df=self.train_data,
                top_n=top_n,
                rating_threshold=rating_threshold
            )
            recommendations[user_id] = recs
            
            if verbose and (batch_idx + 1) % batch_size == 0:
                print(f"  Progress: {batch_idx + 1}/{total} users")
        
        if verbose:
            print(f"  Рекомендации созданы для {total} пользователей")
        
        return recommendations
