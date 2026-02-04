"""Загрузка и управление данными."""

from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np


class DataLoader:
    """Загрузчик данных для рекомендательной системы книг."""
    
    def __init__(self, data_dir: str):
        """
        Инициализация загрузчика.
        
        Args:
            data_dir: путь к папке с CSV файлами
        """
        self.data_dir = data_dir
        self.ratings = None
        self.books = None
        self.tags = None
        self.book_tags = None
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузить все файлы данных.
        
        Returns:
            Словарь с DataFrame'ами
        """
        self.ratings = self._load_csv('ratings.csv')
        self.books = self._load_csv('books.csv')
        self.tags = self._load_csv('tags.csv')
        self.book_tags = self._load_csv('book_tags.csv')
        
        # Валидация
        self._validate_data()
        
        return {
            'ratings': self.ratings,
            'books': self.books,
            'tags': self.tags,
            'book_tags': self.book_tags
        }
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Загрузка CSV файла из директории данных."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"{filename}: {df.shape[0]:,} строк, {df.shape[1]} колонок")
        return df
    
    def _validate_data(self) -> None:
        """Проверка целостности загруженных данных.
        
        Валидирует:
        - Наличие обязательных колонок
        - Целостность ссылок между таблицами
        """
        # Проверка ratings
        if 'user_id' not in self.ratings.columns or 'book_id' not in self.ratings.columns:
            raise ValueError("ratings.csv должен содержать user_id и book_id")
        
        # Проверка books
        if 'book_id' not in self.books.columns:
            raise ValueError("books.csv должен содержать book_id")
        
        # Проверка ссылок между таблицами
        unknown_books = set(self.ratings['book_id']) - set(self.books['book_id'])
        if unknown_books:
            print(f"Внимание: {len(unknown_books)} книг из ratings не найдены в books")
        
        print("Данные валидны")
    
    def get_interaction_matrix(self, ratings_df: pd.DataFrame = None,
                               rating_col: str = 'rating',
                               min_rating: int = 1) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
        """
        Создание матрицы взаимодействий (user x book).
        
        Преобразует таблицу рейтингов в плотную матрицу, где:
        - Строки: пользователи
        - Столбцы: книги
        - Значения: рейтинги
        
        Args:
            ratings_df: DataFrame с рейтингами. Если None, используется self.ratings
            rating_col: колонка с рейтингом
            min_rating: минимальный рейтинг для включения в матрицу
            
        Returns:
            Кортеж (interaction_matrix, user_to_idx, book_to_idx)
        """
        if ratings_df is None:
            ratings_df = self.ratings
        
        # Фильтрация по минимальному рейтингу
        ratings_df = ratings_df[ratings_df[rating_col] >= min_rating].copy()
        
        # Создание маппингов для индексации
        user_ids = sorted(ratings_df['user_id'].unique())
        book_ids = sorted(ratings_df['book_id'].unique())
        
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        book_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
        
        idx_to_book = {idx: bid for bid, idx in book_to_idx.items()}
        
        # Создание матрицы
        n_users = len(user_ids)
        n_books = len(book_ids)
        matrix = np.zeros((n_users, n_books), dtype=np.float32)
        
        # Заполнение матрицы
        for _, row in ratings_df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            book_idx = book_to_idx[row['book_id']]
            matrix[user_idx, book_idx] = row[rating_col]
        
        print(f"Матрица взаимодействий: {matrix.shape}, заполненность: {(matrix > 0).sum() / matrix.size * 100:.2f}%")
        
        return matrix, user_to_idx, idx_to_book
    
    def get_book_info(self) -> pd.DataFrame:
        """Получить информацию о книгах."""
        return self.books.copy()
    
    def get_tags_for_books(self) -> Dict[int, list]:
        """
        Получить теги для каждой книги.
        
        Returns:
            Словарь {book_id: [tag_ids]}
        """
        book_tags_dict = {}
        for _, row in self.book_tags.iterrows():
            book_id = row['book_id']
            tag_id = row['tag_id']
            
            if book_id not in book_tags_dict:
                book_tags_dict[book_id] = []
            book_tags_dict[book_id].append(tag_id)
        
        return book_tags_dict
    
    def get_tag_names(self) -> Dict[int, str]:
        """Получить маппинг tag_id -> tag_name."""
        return dict(zip(self.tags['tag_id'], self.tags['tag_name']))
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить базовую статистику."""
        return {
            'n_users': self.ratings['user_id'].nunique(),
            'n_books': self.ratings['book_id'].nunique(),
            'n_interactions': len(self.ratings),
            'n_unique_books_total': self.books['book_id'].nunique(),
            'sparsity': 1 - (len(self.ratings) / 
                           (self.ratings['user_id'].nunique() * 
                            self.ratings['book_id'].nunique())),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
        }
