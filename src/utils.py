"""Вспомогательные функции и утилиты."""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def temporal_split(df: pd.DataFrame, user_col: str = 'user_id',
                   item_col: str = 'book_id', rating_col: str = 'rating',
                   time_col: str = None, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Временной split данных с сохранением временной последовательности.
    
    Для каждого пользователя отдельно выполняет split:
    - Первые 80% рейтингов идут в обучение
    - Последние 20% идут в тест
    
    Это реалистичный способ оценки, так как модель предсказывает
    будущие рейтинги на основе истории.
    
    Args:
        df: DataFrame с рейтингами
        user_col: Название колонки с user_id
        item_col: Название колонки с book_id
        rating_col: Названи колонки с рейтингом (опционально)
        time_col: Названн колонки с временем (игнорируется, используется порядок строк)
        train_ratio: Доля обучающего набора (по умолчанию 0.8 -> 80/20 split)
        
    Returns:
        Кортеж (train_df, test_df) - обучающий и тестовый наборы
    """
    train_dfs = []
    test_dfs = []
    
    # Для каждого пользователя отдельно
    for user_id in df[user_col].unique():
        user_data = df[df[user_col] == user_id].copy()
        user_data = user_data.reset_index(drop=True)
        
        # Вычисляем точку разделения
        n_samples = len(user_data)
        split_idx = int(n_samples * train_ratio)
        
        # Разделяем на обучение (раньше) и тест (позже)
        train_dfs.append(user_data.iloc[:split_idx])
        test_dfs.append(user_data.iloc[split_idx:])
    
    # Объединяем все user'ов
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df


def get_train_items(train_df: pd.DataFrame, user_id: int,
                    item_col: str = 'book_id') -> set:
    """Получение множества всех книг, которые пользователь оценивал в обучении."""
    return set(train_df[train_df['user_id'] == user_id][item_col].unique())


def remove_train_items(recommendations: List[int], train_items: set) -> List[int]:
    """Удаление из списка рекомендаций книг, которые пользователь уже видел."""
    return [item for item in recommendations if item not in train_items]


def get_relevant_test_items(test_df: pd.DataFrame, user_id: int,
                            rating_threshold: int = 4,
                            item_col: str = 'book_id',
                            rating_col: str = 'rating') -> set:
    """
    Получение множества релевантных книг для пользователя в тестовом наборе.
    
    Релевантной считается книга с rating >= rating_threshold.
    
    Args:
        test_df: Тестовый набор данных
        user_id: ID пользователя
        rating_threshold: Минимальный рейтинг для считания книги релевантной
        item_col: Названн колонки с book_id
        rating_col: Названн колонки с рейтингом
        
    Returns:
        Множество ID релевантных книг
    """
    user_test = test_df[test_df['user_id'] == user_id]
    relevant = user_test[user_test[rating_col] >= rating_threshold][item_col].unique()
    return set(relevant)


def standardize_features(X: np.ndarray, scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Стандартизация признаков (нормализация к нулевому среднему и единичной дисперсии).
    
    Применяет: X_scaled = (X - mean) / std
    
    Args:
        X: Матрица признаков (n_samples, n_features)
        scaler: Готовый scaler для применения. Если None, создает новый из X
        
    Returns:
        Кортеж (X_scaled, scaler) для последующего использования в других наборах
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                fill_value: float = 0.0) -> np.ndarray:
    """
    Безопасное деление с обработкой ноль-деления.
    
    Args:
        numerator: Числитель (делимое)
        denominator: Знаменатель (делитель)
        fill_value: Значение, которое используется при делении на ноль
        
    Returns:
        Результат деления с fill_value где знаменатель == 0
    """
    result = np.divide(numerator, denominator,
                       out=np.full_like(numerator, fill_value, dtype=float),
                       where=denominator != 0)
    return result
