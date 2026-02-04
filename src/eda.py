"""Exploratory Data Analysis - анализ данных."""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    """Класс для исследовательского анализа данных (Exploratory Data Analysis).
    
    Предоставляет методы для получения базовой статистики и визуализации
    распределений рейтингов, активности пользователей и популярности книг.
    """
    
    def __init__(self, ratings_df: pd.DataFrame, books_df: pd.DataFrame,
                 tags_df: pd.DataFrame, book_tags_df: pd.DataFrame):
        """
        Инициализация EDA анализатора.
        
        Args:
            ratings_df: DataFrame с рейтингами (user_id, book_id, rating)
            books_df: DataFrame с информацией о книгах
            tags_df: DataFrame с информацией о тегах
            book_tags_df: DataFrame со связями book-tag
        """
        self.ratings = ratings_df
        self.books = books_df
        self.tags = tags_df
        self.book_tags = book_tags_df
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Получить базовую статистику о датасете.
        
        Returns:
            Словарь со следующей статистикой:
            - n_users: количество уникальных пользователей
            - n_books_rated: количество оцененных книг
            - n_books_total: всего книг в датасете
            - n_interactions: всего рейтингов
            - n_tags: количество уникальных тегов
            - sparsity: разреженность матрицы (доля нулей)
            - avg_rating, median_rating, std_rating: статистика по оценкам
        """
        return {
            'n_users': self.ratings['user_id'].nunique(),
            'n_books_rated': self.ratings['book_id'].nunique(),
            'n_books_total': self.books['book_id'].nunique(),
            'n_interactions': len(self.ratings),
            'n_tags': self.tags['tag_id'].nunique(),
            'sparsity': 1 - (len(self.ratings) / 
                           (self.ratings['user_id'].nunique() * 
                            self.books['book_id'].nunique())),
            'avg_rating': self.ratings['rating'].mean(),
            'median_rating': self.ratings['rating'].median(),
            'std_rating': self.ratings['rating'].std(),
        }
    
    def plot_rating_distribution(self, figsize: Tuple[int, int] = (10, 5),
                                  title: str = 'Distribution of Ratings') -> None:
        """Визуализация распределения оценок.
        
        Показывает, сколько раз каждая оценка была выставлена в датасете.
        Помогает понять распределение оценок (средние ли пользователи или экстремальные).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        self.ratings['rating'].value_counts().sort_index().plot(
            kind='bar', ax=ax, color='steelblue', edgecolor='black'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_user_activity(self, figsize: Tuple[int, int] = (12, 5),
                           bins: int = 50,
                           title: str = 'User Activity Distribution') -> None:
        """Визуализация активности пользователей (количество выставленных оценок).
        
        Показывает распределение пользователей по активности:
        - Слева (линейная шкала): распределение активных пользователей
        - Справа (логарифмическая шкала): видны "редкие" пользователи с малым кол-вом оценок
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        user_counts = self.ratings.groupby('user_id').size()
        
        ax1.hist(user_counts, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Гистограмма кол-ва оценок пользователя', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Ratings', fontsize=11)
        ax1.set_ylabel('Number of Users', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Помогает увидеть хвост распределения
        ax2.hist(user_counts, bins=bins, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_title('Активность пользователей (логарифмическая шкала)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Ratings', fontsize=11)
        ax2.set_ylabel('Number of Users (log)', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Активность пользователей:")
        print(f"  - Среднее оценок на пользователя: {user_counts.mean():.1f}")
        print(f"  - Медиана оценок на пользователя: {user_counts.median():.0f}")
        print(f"  - Минимум: {user_counts.min()}, Максимум: {user_counts.max()}")
    
    def plot_book_popularity(self, figsize: Tuple[int, int] = (12, 5),
                             bins: int = 50,
                             title: str = 'Book Popularity Distribution') -> None:
        """Визуализация популярности книг (количество полученных оценок)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        book_counts = self.ratings.groupby('book_id').size()
        
        ax1.hist(book_counts, bins=bins, color='green', edgecolor='black', alpha=0.7)
        ax1.set_title('Histogram of Book Ratings Count', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Ratings', fontsize=11)
        ax1.set_ylabel('Number of Books', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Log-масштаб
        ax2.hist(book_counts, bins=bins, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_title('Book Ratings Count (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Ratings', fontsize=11)
        ax2.set_ylabel('Number of Books (log)', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Среднее оценок на книгу: {book_counts.mean():.1f}")
        print(f"Медиана оценок на книгу: {book_counts.median():.0f}")
        print(f"Мин: {book_counts.min()}, Макс: {book_counts.max()}")
    
    def plot_top_books(self, top_n: int = 15, figsize: Tuple[int, int] = (12, 6),
                       title: str = 'Top Books by Number of Ratings') -> None:
        """Топ книг по количеству оценок."""
        top_books = self.ratings.groupby('book_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        top_books.columns = ['count', 'mean_rating']
        top_books = top_books.sort_values('count', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(top_books))
        ax.barh(x, top_books['count'], color='steelblue', edgecolor='black')
        ax.set_yticks(x)
        ax.set_yticklabels([f"Book {bid}" for bid in top_books.index], fontsize=10)
        ax.set_xlabel('Number of Ratings', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_tags(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 6),
                      title: str = 'Top Tags by Frequency') -> None:
        """Топ тегов по частоте."""
        tag_counts = self.book_tags.groupby('tag_id').size().sort_values(ascending=False).head(top_n)
        
        # Маппинг tag_id -> название тега
        tag_names = dict(zip(self.tags['tag_id'], self.tags['tag_name']))
        tag_labels = [tag_names.get(tid, f"Tag {tid}") for tid in tag_counts.index]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(range(len(tag_counts)), tag_counts.values, color='purple', edgecolor='black')
        ax.set_yticks(range(len(tag_counts)))
        ax.set_yticklabels(tag_labels, fontsize=10)
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_long_tail(self, percentile_q1: float = 25, percentile_q3: float = 75) -> Dict[str, Any]:
        """
        Анализ long tail эффекта с динамическими порогами.
        
        Args:
            percentile_q1: нижний квартиль для определения границ (по умолчанию 25%)
            percentile_q3: верхний квартиль для определения границ (по умолчанию 75%)
        
        Returns:
            Словарь с анализом long tail эффекта
        """
        book_counts = self.ratings.groupby('book_id').size().sort_values(ascending=False)
        
        # Вычисление динамических порогов на основе квартилей
        q1 = book_counts.quantile(percentile_q1 / 100)
        median = book_counts.median()
        q3 = book_counts.quantile(percentile_q3 / 100)
        
        # Категоризация книг по динамическим порогам
        very_popular = (book_counts >= q3).sum()
        popular = ((book_counts >= median) & (book_counts < q3)).sum()
        medium = ((book_counts >= q1) & (book_counts < median)).sum()
        tail = (book_counts < q1).sum()
        
        # Для tail берём книги ниже Q1
        tail_threshold = q1
        tail_interactions = book_counts[book_counts < tail_threshold].sum()
        
        return {
            'very_popular_q3_plus': {
                'count': very_popular,
                'threshold': int(q3)
            },
            'popular_median_q3': {
                'count': popular,
                'threshold_low': int(median),
                'threshold_high': int(q3)
            },
            'medium_q1_median': {
                'count': medium,
                'threshold_low': int(q1),
                'threshold_high': int(median)
            },
            'long_tail_lt_q1': {
                'count': tail,
                'threshold': int(tail_threshold)
            },
            'long_tail_percentage': tail / len(book_counts) * 100,
            'long_tail_interactions_percentage': tail_interactions / book_counts.sum() * 100,
            'thresholds_used': {
                'q1_25pct': int(q1),
                'median_50pct': int(median),
                'q3_75pct': int(q3)
            }
        }
    
    def analyze_sparsity_issues(self, cold_start_percentile: float = 25,
                               popularity_bias_percentile: float = 90) -> Dict[str, Any]:
        """
        Анализ проблем разреженности и cold start с динамическими порогами.
        
        Args:
            cold_start_percentile: используется медиана * коэффициент вместо жесткого значения 3
            popularity_bias_percentile: верхний процентиль для определения популярных книг (по умолчанию 90%)
        
        Returns:
            Словарь с анализом проблем разреженности
        """
        user_counts = self.ratings.groupby('user_id').size()
        book_counts = self.ratings.groupby('book_id').size()
        
        # Динамический порог для cold start на основе медианы
        # Берём значение, которое составляет ~25% от медианы (т.е. очень активные пользователи)
        user_median = user_counts.median()
        book_median = book_counts.median()
        
        # Cold start: берём пользователей/книги с количеством оценок ниже 1/4 медианы
        # Это более разумный порог, чем жесткое 3
        cold_start_threshold_users = max(1, int(user_median * 0.25))
        cold_start_threshold_books = max(1, int(book_median * 0.25))
        
        cold_start_users = (user_counts <= cold_start_threshold_users).sum()
        cold_start_books = (book_counts <= cold_start_threshold_books).sum()
        
        # Popularity bias: используем процентиль вместо жесткого 10%
        popularity_percentile = popularity_bias_percentile / 100
        n_popular_books = max(1, int(len(book_counts) * (1 - popularity_percentile)))
        top_percent_books = book_counts.nlargest(n_popular_books).sum()
        popularity_bias = top_percent_books / book_counts.sum() * 100
        
        popularity_threshold = book_counts.nlargest(n_popular_books).min()
        
        return {
            'cold_start_users': {
                'count': cold_start_users,
                'percentage': cold_start_users / len(user_counts) * 100,
                'threshold': cold_start_threshold_users,
                'threshold_note': f'<= 25% от медианы ({int(user_median)} оценок)'
            },
            'cold_start_books': {
                'count': cold_start_books,
                'percentage': cold_start_books / len(book_counts) * 100,
                'threshold': cold_start_threshold_books,
                'threshold_note': f'<= 25% от медианы ({int(book_median)} оценок)'
            },
            'popularity_bias': {
                'percentage': popularity_bias,
                'n_popular_books': n_popular_books,
                'popularity_threshold': int(popularity_threshold),
                'percentile_used': popularity_bias_percentile
            },
            'matrix_sparsity': 1 - len(self.ratings) / (len(user_counts) * len(book_counts)),
            'data_statistics': {
                'user_median_ratings': int(user_median),
                'book_median_ratings': int(book_median),
                'user_mean_ratings': int(user_counts.mean()),
                'book_mean_ratings': int(book_counts.mean())
            }
        }
    
    @staticmethod
    def format_stats(stats_dict: Dict[str, Any], title: str = None) -> None:
        """
        Форматированный вывод статистики с автоматическим определением типов.
        
        Args:
            stats_dict: словарь со статистикой
            title: опциональное название секции
        """
        if title:
            print(f"{title}")
        
        for key, value in stats_dict.items():
            if isinstance(value, float):
                # Если это вероятность (0-1) или процент, форматировать соответственно
                if value <= 1.0 and 'sparsity' in key.lower() or 'bias' in key.lower() and 'percentage' not in key:
                    # Это вероятность (0-1)
                    print(f"  {key}: {value:.4f}")
                elif 0 <= value <= 100 and ('percentage' in key.lower() or 'pct' in key.lower() or 'ratio' in key.lower()):
                    # Это процент
                    print(f"  {key}: {value:.2f}%")
                else:
                    # Это обычное число
                    print(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                # Целые числа с разделителями тысяч
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
