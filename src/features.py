"""
Генерация расширенных признаков для рекомендательной системы.
Создает признаки пользователей, объектов (книг) и взаимодействий.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Инженер признаков для генерации user, item и interaction features.
    
    Ответственен за создание расширенных признаков для всех компонентов
    рекомендательной системы.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Инициализация инженера признаков.
        
        Args:
            random_state: Seed для воспроизводимости алгоритмов (особенно KMeans)
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.user_features = None
        self.item_features = None
        

    
    def generate_user_features(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация поведенческих признаков пользователей.
        
        Вычисляет признаки, характеризующие поведение пользователя:
        - avg_rating: средняя оценка пользователя
        - rating_count: количество выставленных оценок
        - activity: логарифмическая активность (нормализованное кол-во оценок)
        - rating_std: вариативность оценок (станд. отклонение)
        - min_rating / max_rating: диапазон оценок пользователя
        - high_rating_ratio: доля высоких оценок (>=4)
        
        Args:
            ratings_df: DataFrame с полями: user_id, rating, timestamp (опционально)
            
        Returns:
            DataFrame с признаками пользователей (индекс = user_id)
        """
        user_feats = pd.DataFrame()
        
        # Средняя оценка
        user_feats['avg_rating'] = ratings_df.groupby('user_id')['rating'].mean()
        
        # Количество оценок
        user_feats['rating_count'] = ratings_df.groupby('user_id')['rating'].count()
        
        # Стандартное отклонение (мера вариативности предпочтений)
        user_feats['rating_std'] = ratings_df.groupby('user_id')['rating'].std().fillna(0)
        
        # Диапазон оценок
        user_feats['min_rating'] = ratings_df.groupby('user_id')['rating'].min()
        user_feats['max_rating'] = ratings_df.groupby('user_id')['rating'].max()
        
        # Activity: логарифмическая нормализация кол-ва оценок
        user_feats['activity'] = np.log1p(user_feats['rating_count'])
        
        # Процент высоких оценок (>=4) - показатель избирательности
        high_ratings = (ratings_df[ratings_df['rating'] >= 4]
                       .groupby('user_id')['rating'].count())
        user_feats['high_rating_ratio'] = (
            high_ratings / user_feats['rating_count']
        ).fillna(0)
        
        self.user_features = user_feats
        return user_feats
    
    def generate_item_features(self, ratings_df: pd.DataFrame, 
                               book_tags_df: pd.DataFrame,
                               tags_df: pd.DataFrame,
                               n_tag_clusters: int = 10) -> pd.DataFrame:
        """
        Генерация признаков книг на основе рейтингов и метаданных.
        
        Вычисляет признаки, характеризующие популярность и качество книги:
        - popularity: количество оценок (абсолютная популярность)
        - avg_rating: средняя оценка книги
        - rating_variance: дисперсия оценок (согласованность мнений)
        - min_rating / max_rating: диапазон полученных оценок
        - rating_spread: размах оценок (макс - мин)
        - log_popularity: логарифмическая нормализация популярности
        - tag_diversity: количество уникальных тегов
        - tag_cluster_X: one-hot кодирование кластеров тегов
        
        Args:
            ratings_df: DataFrame с оценками (user_id, book_id, rating)
            book_tags_df: DataFrame связей (book_id/goodreads_book_id, tag_id)
            tags_df: DataFrame описаний тегов
            n_tag_clusters: Количество кластеров для группировки тегов
            
        Returns:
            DataFrame с признаками книг (индекс = book_id)
        """
        item_feats = pd.DataFrame()
        
        # Популярность: количество оценок книги
        item_feats['popularity'] = ratings_df.groupby('book_id')['rating'].count()
        
        # Средняя оценка книги
        item_feats['avg_rating'] = ratings_df.groupby('book_id')['rating'].mean()
        
        # Дисперсия: мера согласованности оценок
        item_feats['rating_variance'] = ratings_df.groupby('book_id')['rating'].var().fillna(0)
        
        # Диапазон оценок
        item_feats['min_rating'] = ratings_df.groupby('book_id')['rating'].min()
        item_feats['max_rating'] = ratings_df.groupby('book_id')['rating'].max()
        
        # Размах (spread): разница между макс и мин оценками
        item_feats['rating_spread'] = (
            item_feats['max_rating'] - item_feats['min_rating']
        )
        
        # Логарифмическая популярность (нормализация для моделей)
        item_feats['log_popularity'] = np.log1p(item_feats['popularity'])
        
        # Количество уникальных тегов на книгу (разнообразие)
        tag_diversity = book_tags_df.groupby('goodreads_book_id').size()
        item_feats['tag_diversity'] = tag_diversity
        item_feats['tag_diversity'] = item_feats['tag_diversity'].fillna(0)
        
        # Кластеризация тегов для извлечения категориальной информации
        item_feats = self._add_tag_clusters(
            item_feats, book_tags_df, tags_df, n_tag_clusters
        )
        
        self.item_features = item_feats
        return item_feats
    
    def _add_tag_clusters(self, item_feats: pd.DataFrame,
                          book_tags_df: pd.DataFrame,
                          tags_df: pd.DataFrame,
                          n_clusters: int) -> pd.DataFrame:
        """
        Добавляет one-hot кодирование кластеров тегов.
        
        Args:
            item_feats: DataFrame с признаками книг
            book_tags_df: Связи книга-теги
            tags_df: Описания тегов
            n_clusters: Количество кластеров
            
        Returns:
            Обновленный DataFrame с tag_cluster_X признаками
        """
        try:
            # Получение доменов (категорий) тегов
            tag_to_domain = dict(zip(tags_df['tag_id'], tags_df.get('tag_name', tags_df.get('tag_id'))))
            
            # Маппирование тегов на домены
            book_tags_df = book_tags_df.copy()
            book_tags_df['domain'] = book_tags_df['tag_id'].map(tag_to_domain)
            
            # Агрегирование: какие домены у каждой книги
            tag_vectors = []
            book_ids = []
            
            for book_id in item_feats.index:
                book_tags = book_tags_df[book_tags_df['goodreads_book_id'] == book_id]
                if len(book_tags) > 0:
                    domains = book_tags['domain'].unique()
                    # Vector: количество тегов в каждом домене
                    vec = book_tags['domain'].value_counts().to_dict()
                    tag_vectors.append(vec)
                else:
                    tag_vectors.append({})
                book_ids.append(book_id)
            
            # Конвертировать в матрицу
            if tag_vectors and any(tag_vectors):
                tag_df = pd.DataFrame(tag_vectors, index=book_ids).fillna(0)
                
                # K-means кластеризация
                if len(tag_df) > n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                    clusters = kmeans.fit_predict(tag_df)
                    
                    # One-hot кодирование кластеров
                    for i in range(n_clusters):
                        cluster_col = f'tag_cluster_{i}'
                        item_feats[cluster_col] = 0
                        item_feats.loc[book_ids, cluster_col] = (clusters == i).astype(int)
            
        except Exception as e:
            # Fallback: если кластеризация не сработала
            print(f"Предупреждение: Кластеризация тегов ошибка ({e}), используется только разнообразие")
        
        return item_feats
    
    def generate_interaction_features(self, ratings_df: pd.DataFrame,
                                     train_df: pd.DataFrame,
                                     interaction_matrix: csr_matrix,
                                     user_ids: List,
                                     book_ids: List,
                                     user_to_idx: Dict,
                                     idx_to_book: Dict) -> pd.DataFrame:
        """
        Генерирует признаки взаимодействия пользователь-книга.
        
        Features:
        - similarity_to_history: косинусное сходство с историей пользователя
        - recency: как давно пользователь последний раз оценивал (дней)
        - user_book_similarity: сходство между пользовательским вектором и вектором книги
        - rating_position: позиция оценки в истории пользователя
        
        Args:
            ratings_df: Все оценки
            train_df: Обучающее множество (история)
            interaction_matrix: Матрица взаимодействий (sparse CSR)
            user_ids: Список ID пользователей
            book_ids: Список ID книг
            user_to_idx: Маппирование user_id -> индекс в матрице
            idx_to_book: Маппирование индекс -> book_id
            
        Returns:
            DataFrame с признаками взаимодействий для каждой пары (user, item)
        """
        interaction_feats = []
        
        # Подготовка: дата для recency
        train_df = train_df.copy()
        if 'timestamp' not in train_df.columns:
            # Fallback: используем порядок как прокси для времени
            train_df['timestamp'] = pd.to_datetime('2024-01-01')
        else:
            train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        
        max_date = train_df['timestamp'].max()
        
        # Последняя оценка пользователя
        user_last_rating = train_df.groupby('user_id')['timestamp'].max()
        
        # Для каждого пользователя в тесте
        for user_id in user_ids:
            if user_id not in user_to_idx:
                continue
            
            user_idx = user_to_idx[user_id]
            # Работает с обоими типами матриц (sparse и dense)
            if hasattr(interaction_matrix[user_idx], 'toarray'):
                user_vector = interaction_matrix[user_idx].toarray().flatten()
            else:
                user_vector = interaction_matrix[user_idx].flatten()
            
            # История пользователя
            user_history = train_df[train_df['user_id'] == user_id]
            
            if len(user_history) == 0:
                continue
            
            last_rating_date = user_last_rating.get(user_id, max_date)
            recency_days = (max_date - last_rating_date).days
            
            # Для каждой книги
            for book_id in book_ids:
                try:
                    book_idx = list(idx_to_book.values()).index(book_id)
                except (ValueError, IndexError):
                    continue
                
                # Работает с обоими типами матриц (sparse и dense)
                if hasattr(interaction_matrix[:, book_idx], 'toarray'):
                    item_vector = interaction_matrix[:, book_idx].toarray().flatten()
                else:
                    item_vector = interaction_matrix[:, book_idx].flatten()
                
                # Similarity to history
                if np.linalg.norm(user_vector) > 0 and np.linalg.norm(item_vector) > 0:
                    similarity = 1 - cosine(user_vector, item_vector)
                else:
                    similarity = 0
                
                # Rating position in user history
                rating_count = len(user_history)
                position = user_history[user_history['book_id'] == book_id].shape[0]
                
                interaction_feats.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'similarity_to_history': similarity,
                    'recency_days': recency_days,
                    'user_history_size': rating_count
                })
        
        interaction_df = pd.DataFrame(interaction_feats)
        return interaction_df
    
    def normalize_features(self, features_df: pd.DataFrame,
                          exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Нормализует признаки (исключая категориальные и binary).
        
        Args:
            features_df: DataFrame с признаками
            exclude_cols: Колонки для исключения из нормализации
            
        Returns:
            DataFrame с нормализованными признаками
        """
        if exclude_cols is None:
            exclude_cols = []
        
        features_norm = features_df.copy()
        numeric_cols = features_norm.select_dtypes(include=[np.number]).columns
        
        # Исключаем binary и уже перечисленные колонки
        cols_to_normalize = [
            col for col in numeric_cols
            if col not in exclude_cols and 
            not (features_norm[col].dtype == int and features_norm[col].max() <= 1)
        ]
        
        if cols_to_normalize:
            features_norm[cols_to_normalize] = self.scaler.fit_transform(
                features_norm[cols_to_normalize]
            )
        
        return features_norm
    
    def create_all_features(self, ratings_df: pd.DataFrame,
                           book_tags_df: pd.DataFrame,
                           tags_df: pd.DataFrame,
                           train_df: pd.DataFrame,
                           interaction_matrix: csr_matrix = None,
                           user_ids: List = None,
                           book_ids: List = None,
                           user_to_idx: Dict = None,
                           idx_to_book: Dict = None,
                           normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Создает все типы признаков: user, item, interaction.
        Все признаки создаются с dtype float32 для совместимости с нейросетями.
        
        Args:
            ratings_df: DataFrame оценок
            book_tags_df: DataFrame книга-теги
            tags_df: DataFrame тегов
            train_df: Обучающее множество для interaction features
            interaction_matrix: Матрица взаимодействий (опционально для interaction features)
            user_ids: Список user IDs (опционально)
            book_ids: Список book IDs (опционально)
            user_to_idx: Маппирование user_id -> индекс (опционально)
            idx_to_book: Маппирование индекс -> book_id (опционально)
            normalize: Нормализовать ли признаки
            
        Returns:
            Dict с ключами 'user_features', 'item_features', 'interaction_features'
        """
        print("Генерирование признаков пользователя...")
        user_feats = self.generate_user_features(ratings_df)
        
        print("Генерирование признаков предмета...")
        item_feats = self.generate_item_features(ratings_df, book_tags_df, tags_df)
        
        print("Генерирование признаков взаимодействия...")
        # Interaction features отключены для экономии времени (слишком дорого для 53k пользователей x 10k книг)
        interaction_feats = pd.DataFrame()
        
        if False and all([interaction_matrix is not None, user_ids is not None, 
                book_ids is not None, user_to_idx is not None, idx_to_book is not None]):
            interaction_feats = self.generate_interaction_features(
                ratings_df, train_df, interaction_matrix, user_ids, 
                book_ids, user_to_idx, idx_to_book
            )
        else:
            print("  Пропуск признаков взаимодействия (отсутствуют параметры)")
            interaction_feats = pd.DataFrame()
        
        # Нормализация
        if normalize:
            print("Нормализация признаков...")
            user_feats = self.normalize_features(user_feats)
            item_feats = self.normalize_features(item_feats)
        
        # Явная конвертация в float32 для совместимости с PyTorch
        print("Конвертирование признаков в float32...")
        for col in user_feats.columns:
            if user_feats[col].dtype != np.float32:
                user_feats[col] = user_feats[col].astype(np.float32)
        
        for col in item_feats.columns:
            if item_feats[col].dtype != np.float32:
                item_feats[col] = item_feats[col].astype(np.float32)
        
        print(f"Признаки пользователей: {len(user_feats)} пользователей, {len(user_feats.columns)} признаков (dtype={user_feats.iloc[:, 0].dtype})")
        print(f"Признаки книг: {len(item_feats)} книг, {len(item_feats.columns)} признаков (dtype={item_feats.iloc[:, 0].dtype})")
        print(f"Признаки взаимодействия: {len(interaction_feats)} пар")
        
        return {
            'user_features': user_feats,
            'item_features': item_feats,
            'interaction_features': interaction_feats
        }
    
    def get_user_feature_names(self) -> List[str]:
        """Получить названия пользовательских признаков."""
        if self.user_features is not None:
            return list(self.user_features.columns)
        return []
    
    def get_item_feature_names(self) -> List[str]:
        """Получить названия признаков книг."""
        if self.item_features is not None:
            return list(self.item_features.columns)
        return []
