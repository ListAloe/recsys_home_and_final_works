"""Модель на основе популярности."""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class PopularityModel:
    """
    Базовая бенчмарк модель на основе глобальной популярности.
    
    Рекомендует наиболее популярные (высокооцененные) книги,
    которые пользователь еще не скотрел. Служит хорошим бенчмарком
    для сравнения с более сложными моделями.
    """
    
    def __init__(self, min_ratings: int = 1, random_state: int = 42):
        """
        Инициализация модели популярности.
        
        Args:
            min_ratings: Минимальное количество оценок для включения книги
            random_state: Seed для воспроизводимости
        """
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.book_stats = None
        self.train_data = None
    
    def fit(self, train_df: pd.DataFrame, rating_col: str = 'rating',
            min_rating_threshold: float = 4.0) -> 'PopularityModel':
        """
        Обучение модели на обучающем наборе.
        
        Вычисляет агреги:
        - mean_rating: средняя оценка для каждой книги
        - count: количество оценок (мера популярности)
        - std: стандартное отклонение оценок
        
        Книги ранжируются по среднему рейтингу (как основной критерий)
        и количеству оценок (как tie-breaker).
        
        Args:
            train_df: Обучающие данные с рейтингами
            rating_col: Название колонки рейтинга
            min_rating_threshold: Порог релевантности (для связи с fit, не используется)
            
        Returns:
            self (для цепочки вызовов)
        """
        self.train_data = train_df.copy()
        
        stats = []
        for book_id in train_df['book_id'].unique():
            book_ratings = train_df[train_df['book_id'] == book_id][rating_col]
            
            n_ratings = len(book_ratings)
            # Фильтруем книги с недостаточным количеством оценок
            if n_ratings < self.min_ratings:
                continue
            
            stats.append({
                'book_id': book_id,
                'mean_rating': book_ratings.mean(),
                'count': n_ratings,
                'std': book_ratings.std() if n_ratings > 1 else 0.0
            })
        
        self.book_stats = pd.DataFrame(stats)
        
        # Основной критерий: средняя оценка (по убыванию)
        # Вторичный критерий: количество оценок (по убыванию) - tie-breaker
        self.book_stats = self.book_stats.sort_values(
            by=['mean_rating', 'count'],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        print(f"Модель популярности обучена на {len(self.book_stats)} книгах")
        
        return self
    
    def get_recommendations(self, user_id: int, top_n: int = 10,
                           exclude_seen: bool = True) -> List[int]:
        """
        Получение рекомендаций на основе популярности.
        
        Возвращает top_n самых популярных книг (отсортированных по рейтингу и кол-ву оценок),
        исключая уже оцененные пользователем.
        
        Args:
            user_id: ID пользователя
            top_n: Количество рекомендаций
            exclude_seen: Исключать ли книги, которые пользователь уже оценивал
            
        Returns:
            Список ID книг, отсортированный по популярности (убывание)
        """
        if self.book_stats is None or self.train_data is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        recommendations = self.book_stats['book_id'].values.copy()
        
        if exclude_seen:
            user_seen = set(
                self.train_data[self.train_data['user_id'] == user_id]['book_id']
            )
            recommendations = [bid for bid in recommendations if bid not in user_seen]
        
        # Возвращаем top_n элементов (преобразуем в список если нужно)
        return recommendations[:top_n] if isinstance(recommendations, list) else recommendations[:top_n].tolist()
    
    def get_all_recommendations(self, test_df: pd.DataFrame, top_n: int = 10,
                               exclude_train: bool = True,
                               train_df: pd.DataFrame = None,
                               batch_size: int = 100,
                               verbose: bool = False) -> Dict[int, List[int]]:
        """
        Получить рекомендации для всех пользователей в тесте (пакетная обработка).
        
        Args:
            test_df: тестовые данные (для определения пользователей)
            top_n: количество рекомендаций на пользователя
            exclude_train: исключать ли обучающие взаимодействия
            train_df: обучающие данные (нужны если exclude_train=True)
            batch_size: размер пакета для вывода прогресса
            verbose: показывать прогресс
            
        Returns:
            Словарь {user_id: [book_ids]}
        """
        recommendations = {}
        user_ids = test_df['user_id'].unique()
        total = len(user_ids)
        
        for batch_idx, user_id in enumerate(user_ids):
            recs = self.get_recommendations(user_id, top_n=top_n, exclude_seen=exclude_train)
            recommendations[user_id] = recs
            
            if verbose and (batch_idx + 1) % batch_size == 0:
                print(f"  Прогресс: {batch_idx + 1}/{total} пользователей")
        
        if verbose:
            print(f"  Рекомендации созданы для {total} пользователей")
        
        return recommendations
    
    def get_popularity_score(self, book_id: int) -> float:
        """Получить оценку популярности книги."""
        if self.book_stats is None:
            raise ValueError("Model not fitted yet.")
        
        match = self.book_stats[self.book_stats['book_id'] == book_id]
        if match.empty:
            return 0.0
        
        # Нормализация до [0, 1]
        mean_rating = match['mean_rating'].values[0]
        return (mean_rating - 1.0) / 4.0  # Рейтинг от 1 до 5
