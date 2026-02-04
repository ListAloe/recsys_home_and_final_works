"""Предобработка данных."""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessor:
    """Класс для предобработки данных."""
    
    def __init__(self, books_df: pd.DataFrame, tags_df: pd.DataFrame,
                 book_tags_df: pd.DataFrame):
        """
        Инициализация препроцессора.
        
        Args:
            books_df: DataFrame с информацией о книгах
            tags_df: DataFrame с информацией о тегах
            book_tags_df: DataFrame с связями книга-теги
        """
        self.books = books_df.copy()
        self.tags = tags_df.copy()
        self.book_tags = book_tags_df.copy()
        self.tag_map = None  # Маппинг: tag_id -> tag_name
    
    def build_tag_map(self) -> Dict[int, str]:
        """Создание маппинга tag_id -> tag_name."""
        self.tag_map = dict(zip(self.tags['tag_id'], self.tags['tag_name']))
        return self.tag_map
    
    def create_book_profiles(self, text_col: str = 'original_title',
                             include_tags: bool = True) -> pd.DataFrame:
        """
        Создание текстовых профилей книг для контент-модели.
        
        Объединяет название книги и её теги в единый текстовый профиль.
        
        Args:
            text_col: название колонки с текстом книги
            include_tags: включать ли теги в профиль
            
        Returns:
            DataFrame с колонками: book_id, text_profile
        """
        if self.tag_map is None:
            self.build_tag_map()

        profiles = []

        # Определение названия колонки в таблице связей книга-теги
        book_tags_book_col = None
        if 'book_id' in self.book_tags.columns:
            book_tags_book_col = 'book_id'
        elif 'goodreads_book_id' in self.book_tags.columns:
            book_tags_book_col = 'goodreads_book_id'

        # Создание маппинга для преобразования goodreads_book_id -> book_id
        goodreads_to_book = {}
        if book_tags_book_col == 'goodreads_book_id' and 'goodreads_book_id' in self.books.columns:
            goodreads_to_book = dict(zip(self.books['goodreads_book_id'], self.books['book_id']))

        for _, row in self.books.iterrows():
            book_id = row['book_id']
            # Извлечение названия книги
            title = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""

            # Извлечение тегов из таблицы связей
            tags_text = ""
            if include_tags and book_tags_book_col is not None:
                book_tags_ids = []
                
                if book_tags_book_col == 'book_id':
                    # Прямая ссылка по book_id
                    book_tags_ids = self.book_tags[self.book_tags['book_id'] == book_id]['tag_id'].tolist()
                else:
                    # Ссылка через goodreads_book_id
                    goodreads_id = row.get('goodreads_book_id') if 'goodreads_book_id' in row.index else None
                    
                    if goodreads_id is not None:
                        book_tags_ids = self.book_tags[self.book_tags['goodreads_book_id'] == goodreads_id]['tag_id'].tolist()
                    else:
                        # Попытка найти через маппинг
                        for gr_id, b_id in goodreads_to_book.items():
                            if b_id == book_id:
                                goodreads_id = gr_id
                                break
                        
                        if goodreads_id is not None:
                            book_tags_ids = self.book_tags[self.book_tags['goodreads_book_id'] == goodreads_id]['tag_id'].tolist()

                # Объединение названий тегов в текст
                tags_text = ' '.join([
                    self.tag_map.get(tag_id, '')
                    for tag_id in book_tags_ids
                    if tag_id in self.tag_map
                ])

            profile = f"{title} {tags_text}".strip()

            profiles.append({
                'book_id': book_id,
                'text_profile': profile
            })

        profiles_df = pd.DataFrame(profiles)
        return profiles_df
    
    def extract_tfidf_features(self, texts: List[str], max_features: int = 1000,
                               min_df: int = 1, max_df: float = 0.95) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Извлечение TF-IDF признаков из текстов.
        
        Args:
            texts: список текстов для векторизации
            max_features: максимальное количество признаков
            min_df: минимальное количество документов с термом
            max_df: максимальная доля документов с термом (от 0 до 1)
            
        Returns:
            Кортеж (tfidf_matrix, vectorizer) для использования на новых текстах
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),  # Используем унграммы и биграммы
            stop_words='english'
        )
        
        # Обучение векторизатора и преобразование текстов
        tfidf_matrix = vectorizer.fit_transform(texts)
        sparsity = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
        print(f"TF-IDF матрица: {tfidf_matrix.shape}, плотность: {sparsity * 100:.2f}%")
        
        return tfidf_matrix.astype(np.float32), vectorizer
    
    def clean_ratings(self, ratings_df: pd.DataFrame,
                      min_rating: float = 1,
                      max_rating: float = 5) -> pd.DataFrame:
        """
        Очистка и валидация рейтингов.
        
        Удаляет пропущенные значения и фильтрует выбросы.
        
        Args:
            ratings_df: DataFrame с рейтингами
            min_rating: минимальный допустимый рейтинг
            max_rating: максимальный допустимый рейтинг
            
        Returns:
            Очищенный DataFrame
        """
        df = ratings_df.copy()
        
        # Удаление пропусков
        df = df.dropna(subset=['user_id', 'book_id', 'rating'])
        
        # Фильтр по диапазону рейтингов
        df = df[(df['rating'] >= min_rating) & (df['rating'] <= max_rating)]
        
        # Удаление дубликатов (оставление первого/последнего по времени если есть)
        if 'date_added' in df.columns:
            df = df.sort_values('date_added')
        
        df = df.drop_duplicates(subset=['user_id', 'book_id'], keep='last')
        
        print(f"Рейтинги очищены: {len(df):,} взаимодействий")
        
        return df
    
    def create_user_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создать признаки пользователей.
        
        Признаки:
        - avg_rating: средний рейтинг
        - rating_count: количество оценок
        - rating_std: стандартное отклонение рейтингов
        - activity: активность (новая -> 0, активная -> 1)
        
        Args:
            ratings_df: DataFrame с рейтингами
            
        Returns:
            DataFrame с признаками пользователей
        """
        user_features = []
        
        for user_id in ratings_df['user_id'].unique():
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]['rating']
            
            n_ratings = len(user_ratings)
            avg_rating = user_ratings.mean()
            rating_std = user_ratings.std() if n_ratings > 1 else 0
            
            # Активность: количество оценок
            activity = min(1.0, n_ratings / 100)
            
            user_features.append({
                'user_id': user_id,
                'avg_rating': avg_rating,
                'rating_count': n_ratings,
                'rating_std': rating_std,
                'activity': activity
            })
        
        return pd.DataFrame(user_features)
    
    def create_item_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создать признаки книг.
        
        Признаки:
        - popularity: количество оценок
        - mean_rating: средний рейтинг
        - rating_std: стандартное отклонение
        - variance: дисперсия оценок
        
        Args:
            ratings_df: DataFrame с рейтингами
            
        Returns:
            DataFrame с признаками книг
        """
        item_features = []
        
        for book_id in ratings_df['book_id'].unique():
            book_ratings = ratings_df[ratings_df['book_id'] == book_id]['rating']
            
            n_ratings = len(book_ratings)
            mean_rating = book_ratings.mean()
            rating_std = book_ratings.std() if n_ratings > 1 else 0
            
            popularity = min(1.0, n_ratings / 100)
            
            item_features.append({
                'book_id': book_id,
                'popularity': popularity,
                'mean_rating': mean_rating,
                'rating_std': rating_std,
                'n_ratings': n_ratings
            })
        
        return pd.DataFrame(item_features)
