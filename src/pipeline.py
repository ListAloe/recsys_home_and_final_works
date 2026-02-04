"""Сквозной pipeline для рекомендательной системы."""

from typing import Dict, Any, List, Tuple, Optional
import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.utils import temporal_split
from src.eda import EDA
from src.popularity_model import PopularityModel
from src.content_model import ContentBasedModel
from src.collaborative_filtering import ItemBasedCF
from src.matrix_factorization import MatrixFactorization
from src.neural_model import NeuralRecommender
from src.hybrid_system import HybridRecommender
from src.evaluation import ModelEvaluator, EvaluationMetrics


class RecommendationPipeline:
    """
    Полный pipeline для обучения и оценки рекомендательной системы.
    
    Объединяет все этапы рекомендвации:
    1. Загрузка данных из CSV файлов
    2. Предобработка и валидация
    3. Временное разделение (train/test)
    4. Создание матрицы взаимодействий
    5. Обучение всех моделей
    6. Гибридизация моделей
    7. Оценка производительности
    
    Использование:
        pipeline = RecommendationPipeline('data/')
        pipeline.run() - запустить полный pipeline
    """
    
    def __init__(self, data_dir: str, random_state: int = 42):
        """
        Инициализация pipeline.
        
        Args:
            data_dir: Путь к папке с CSV файлами данных
            random_state: Seed для воспроизводимости всех случайных операций
        """
        self.data_dir = data_dir
        self.random_state = random_state
        
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.data_loader = None
        self.preprocessor = None
        self.evaluator = None
        
        self.ratings = None
        self.books = None
        self.tags = None
        self.book_tags = None
        self.train_df = None
        self.test_df = None
        self.book_profiles = None
        
        self.popularity_model = None
        self.content_model = None
        self.cf_model = None
        self.mf_model = None
        self.neural_model = None
        self.hybrid_model = None
        
        self.interaction_matrix = None
        self.book_ids = None
        self.user_ids = None
        self.book_id_to_idx = None
    
    def load_data(self) -> 'RecommendationPipeline':
        """Загрузка всех данных из CSV файлов и валидация.
        
        Returns:
            self (для цепочки вызовов)
        """
        print("=" * 50)
        print("ЭТАП 1: Загрузка данных")
        print("=" * 50)
        
        self.data_loader = DataLoader(self.data_dir)
        data = self.data_loader.load_all()
        
        self.ratings = data['ratings']
        self.books = data['books']
        self.tags = data['tags']
        self.book_tags = data['book_tags']
        
        # Получение и вывод статистики
        stats = self.data_loader.get_stats()
        print("\nСтатистика датасета:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        
        return self
    
    def preprocess(self) -> 'RecommendationPipeline':
        """Предобработка данных."""
        print("ЭТАП 2: Предобработка")
        
        # Очистить рейтинги
        self.ratings = self.ratings.dropna(subset=['user_id', 'book_id', 'rating'])
        self.ratings = self.ratings.drop_duplicates(
            subset=['user_id', 'book_id'],
            keep='last'
        )
        
        # Инициализация preprocessor
        self.preprocessor = DataPreprocessor(
            self.books, self.tags, self.book_tags
        )
        
        # Создание текстовых профилей книг
        print("\nСоздание профилей книг...")
        self.book_profiles = self.preprocessor.create_book_profiles(
            text_col='original_title',
            include_tags=True
        )
        print(f"Профили книг созданы: {len(self.book_profiles)} книг")
        
        return self
    
    def split_data(self, train_ratio: float = 0.8) -> 'RecommendationPipeline':
        """Temporal split данных."""
        print("ЭТАП 3: Временной split")
        
        self.train_df, self.test_df = temporal_split(
            self.ratings,
            user_col='user_id',
            item_col='book_id',
            rating_col='rating',
            train_ratio=train_ratio
        )
        
        print(f"\nОбучающий набор: {len(self.train_df):,} взаимодействий, {self.train_df['user_id'].nunique():,} пользователей")
        print(f"Тестовый набор: {len(self.test_df):,} взаимодействий, {self.test_df['user_id'].nunique():,} пользователей")
        
        # Создание матрицы взаимодействий
        self.interaction_matrix, user_to_idx, idx_to_book = self.data_loader.get_interaction_matrix(
            self.train_df
        )
        
        self.user_ids = list(user_to_idx.keys())
        self.book_ids = sorted(idx_to_book.values())
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        return self
    
    def run_eda(self) -> 'RecommendationPipeline':
        """Exploratory Data Analysis."""
        print("\n" + "=" * 60)
        print("ЭТАП 4: Анализ данных (EDA)")
        print("=" * 60)
        
        eda = EDA(self.train_df, self.books, self.tags, self.book_tags)
        
        # Базовая статистика
        print("\nБазовая статистика:")
        stats = eda.get_basic_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Long tail анализ
        print("\nАнализ длинного хвоста:")
        long_tail = eda.analyze_long_tail()
        for key, value in long_tail.items():
            if 'percentage' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:,}")
        
        # Sparsity и cold-start
        print("\nПроблемы разреженности и холодного старта:")
        sparsity = eda.analyze_sparsity_issues()
        for key, value in sparsity.items():
            if 'percentage' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        
        return self
    
    def train_models(self, neural_epochs: int = 5, neural_batch_size: int = 32) -> 'RecommendationPipeline':
        """Обучить все модели."""
        print("\n" + "=" * 60)
        print("ЭТАП 5: Обучение моделей")
        print("=" * 60)
        
        # Popularity Model
        print("\n1. Модель популярности...")
        self.popularity_model = PopularityModel(min_ratings=1)
        self.popularity_model.fit(self.train_df)
        
        # Content-Based Model
        print("\n2. Content-Based модель...")
        self.content_model = ContentBasedModel(max_features=1000)
        self.content_model.fit(
            self.book_profiles,
            self.train_df,
            profile_col='text_profile'
        )
        
        # Collaborative Filtering
        print("\n3. Item-Based CF модель...")
        self.cf_model = ItemBasedCF(similarity_metric='cosine', k_neighbors=10)
        self.cf_model.fit(
            self.interaction_matrix,
            self.book_ids,
            self.user_ids,
            self.train_df
        )
        
        # Matrix Factorization (SVD)
        print("\n4. Факторизация матриц (SVD)...")
        self.mf_model = MatrixFactorization(n_factors=50, n_iter=100)
        self.mf_model.fit(
            self.interaction_matrix,
            self.book_ids,
            self.user_ids,
            self.train_df
        )
        
        # Neural Model
        print("\n5. Нейросетевая Two-Tower модель...")
        self.neural_model = NeuralRecommender(
            n_users=len(self.user_ids),
            n_items=len(self.book_ids),
            embedding_dim=64,
            hidden_dims=[128, 64],
            device='cpu',
            learning_rate=0.001
        )
        self.neural_model.fit(
            self.train_df,
            self.user_ids,
            self.book_ids,
            epochs=neural_epochs,
            batch_size=neural_batch_size,
            verbose=True
        )
        
        return self
    
    def evaluate_models(self, k_values: List[int] = None) -> pd.DataFrame:
        """Оценить отдельные модели."""
        print("ЭТАП 6: Оценка моделей")
        
        if k_values is None:
            k_values = [5, 10]
        
        self.evaluator = ModelEvaluator(
            self.test_df,
            self.train_df,
            rating_threshold=4
        )
        
        # Получение рекомендаций от каждой модели
        print("\nГенерирование рекомендаций...")
        
        models_recs = {
            'Popularity': self.popularity_model.get_all_recommendations(self.test_df, top_n=20),
            'Content-Based': self.content_model.get_all_recommendations(self.test_df, top_n=20),
            'Item-Based CF': self.cf_model.get_all_recommendations(self.test_df, top_n=20),
            'SVD': self.mf_model.get_all_recommendations(self.test_df, top_n=20),
            'Neural': self.neural_model.get_all_recommendations(self.test_df, top_n=20, train_df=self.train_df)
        }
        
        # Сравнить модели
        print("\nСравнение моделей...")
        comparison_df = self.evaluator.compare_models(models_recs, k_values=k_values)
        
        print("\nРезультаты сравнения моделей:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def build_hybrid(self) -> 'RecommendationPipeline':
        """Построить гибридную систему."""
        print("ЭТАП 7: Построение гибридной системы")
        
        self.hybrid_model = HybridRecommender()
        
        # Добавление моделей с начальными весами
        self.hybrid_model.add_model('Popularity', self.popularity_model, weight=0.1)
        self.hybrid_model.add_model('Content', self.content_model, weight=0.2)
        self.hybrid_model.add_model('CF', self.cf_model, weight=0.2)
        self.hybrid_model.add_model('SVD', self.mf_model, weight=0.25)
        self.hybrid_model.add_model('Neural', self.neural_model, weight=0.25)
        
        self.hybrid_model.set_fallback_model(self.popularity_model)
        self.hybrid_model.normalize_weights()
        
        print("\nНачальные веса:")
        for name, weight in self.hybrid_model.weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return self
    
    def optimize_hybrid(self, k: int = 10, iterations: int = 30) -> Dict[str, float]:
        """Оптимизировать веса гибридной системы."""
        print("ЭТАП 8: Оптимизация весов гибридной системы")
        
        optimized_weights = self.hybrid_model.optimize_weights(
            self.test_df,
            self.train_df,
            rating_threshold=4,
            k=k,
            learning_rate=0.05,
            iterations=iterations
        )
        
        print("\nОптимизированные веса:")
        for name, weight in optimized_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return optimized_weights
    
    def evaluate_hybrid(self, k_values: List[int] = None) -> pd.DataFrame:
        """Оценить гибридную систему."""
        print("ЭТАП 9: Оценка гибридной системы")
        
        if k_values is None:
            k_values = [5, 10]
        
        evaluator = ModelEvaluator(
            self.test_df,
            self.train_df,
            rating_threshold=4
        )
        
        # Получение рекомендаций от гибридной модели
        print("\nГенерирование рекомендаций гибридной системы...")
        hybrid_recs = self.hybrid_model.get_all_recommendations(
            self.test_df,
            top_n=20,
            train_df=self.train_df
        )
        
        # Оценить
        print("\nРезультаты оценки:")
        results = evaluator.evaluate_model(hybrid_recs, k_values=k_values)
        
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        return results
    
    def run_full_pipeline(self, neural_epochs: int = 5,
                         k_values: List[int] = None) -> Dict[str, Any]:
        """
        Запустить полный pipeline.
        
        Args:
            neural_epochs: количество эпох для нейросети
            k_values: значения K для метрик
            
        Returns:
            Словарь с результатами всех этапов
        """
        if k_values is None:
            k_values = [5, 10]
        
        # Выполнение всех этапов
        self.load_data()
        self.preprocess()
        self.split_data()
        self.run_eda()
        self.train_models(neural_epochs=neural_epochs)
        
        # Оценика отдельных моделей
        comparison_df = self.evaluate_models(k_values=k_values)
        
        # Гибридная система
        self.build_hybrid()
        self.optimize_hybrid()
        
        # Финальная оценка гибридной системы
        hybrid_results = self.evaluate_hybrid(k_values=k_values)
        
        print("\n" + "=" * 60)
        print("PIPELINE ЗАВЕРШЕН")
        print("=" * 60)
        
        return {
            'models_comparison': comparison_df,
            'hybrid_results': hybrid_results,
            'models': {
                'popularity': self.popularity_model,
                'content': self.content_model,
                'cf': self.cf_model,
                'mf': self.mf_model,
                'neural': self.neural_model,
                'hybrid': self.hybrid_model,
            }
        }
